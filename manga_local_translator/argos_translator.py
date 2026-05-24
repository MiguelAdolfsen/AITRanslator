from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from .logging_utils import shorten
from .translation_rules import (
    normalize_japanese_for_translation,
    postprocess_translation,
    prepare_source_for_translation,
    translate_known_phrase,
)
from .translator_base import Translator

logger = logging.getLogger(__name__)


class ArgosTranslator(Translator):
    def __init__(
        self,
        source_code: str = "ja",
        target_code: str = "en",
        glossary_path: Path | None = None,
    ) -> None:
        super().__init__(glossary_path)
        logger.info("Initializing Argos translator: %s -> %s", source_code, target_code)
        try:
            import argostranslate.translate
        except ImportError as exc:
            logger.exception("argostranslate import failed")
            raise RuntimeError(
                "argostranslate is not installed. Run: pip install -r requirements.txt"
            ) from exc

        installed_languages = argostranslate.translate.get_installed_languages()
        logger.debug(
            "Installed Argos languages: %s",
            ", ".join(f"{lang.code}:{lang.name}" for lang in installed_languages),
        )
        source = next((lang for lang in installed_languages if lang.code == source_code), None)
        target = next((lang for lang in installed_languages if lang.code == target_code), None)
        if source is None or target is None or source.get_translation(target) is None:
            raise RuntimeError(
                "Argos Japanese to English model is not installed. "
                "Run: python -m manga_local_translator.install_argos ja en"
            )

        self._cache: dict[str, str] = {}
        self._source_code = source_code
        self._target_code = target_code
        self._disabled = False
        self._timeout_seconds = 5
        logger.info("Argos translator ready")

    def translate(self, text: str) -> str:
        if text not in self._cache:
            logger.debug("Argos translating text=%s", shorten(text))
            self._cache[text] = self._translate_normalized(text)
            logger.debug("Argos translated result=%s", shorten(self._cache[text]))
        else:
            logger.debug("Argos translation cache hit for text=%s", shorten(text))
        return self._cache[text]

    def _translate_normalized(self, text: str) -> str:
        normalized = normalize_japanese_for_translation(text)
        prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
        phrase = translate_known_phrase(normalized, self._glossary) or translate_known_phrase(prepared, self._glossary)
        if phrase is not None:
            logger.debug("Phrasebook translation: source=%s result=%s", shorten(normalized), shorten(phrase))
            return phrase
        if self._disabled:
            logger.warning("Argos translation is disabled after an earlier failure; using fallback for text=%s", shorten(text))
            return ""
        translated = translate_argos_subprocess(
            prepared,
            source_code=self._source_code,
            target_code=self._target_code,
            timeout_seconds=self._timeout_seconds,
        )
        if not translated.strip():
            self._disabled = True
            logger.warning("Disabling Argos for this run after failed translation for text=%s", shorten(normalized))
        return postprocess_translation(normalized, prepared, translated, self._glossary)


def translate_argos_subprocess(
    text: str,
    *,
    source_code: str,
    target_code: str,
    timeout_seconds: int,
) -> str:
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "manga_local_translator.argos_worker",
                source_code,
                target_code,
            ],
            input=json.dumps({"text": text}, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("Argos subprocess timed out after %d seconds for text=%s", timeout_seconds, shorten(text))
        return ""

    if completed.returncode != 0:
        logger.error("Argos subprocess failed: returncode=%s stderr=%s", completed.returncode, shorten(completed.stderr))
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        logger.error(
            "Argos subprocess returned invalid JSON: stdout=%s stderr=%s",
            shorten(completed.stdout),
            shorten(completed.stderr),
        )
        return ""

    if not payload.get("ok"):
        logger.error("Argos subprocess error for text=%s error=%s", shorten(text), payload.get("error"))
        return ""
    return str(payload.get("text", ""))
