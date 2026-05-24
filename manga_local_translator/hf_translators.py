from __future__ import annotations

import logging
import os
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path

from .logging_utils import shorten
from .translation_rules import (
    choose_contextual_translation,
    clean_translation_unit,
    normalize_japanese_for_translation,
    postprocess_translation,
    prepare_source_for_translation,
    should_use_context_translation,
    split_japanese_translation_units,
    translate_known_phrase,
)
from .translator_base import Translator

logger = logging.getLogger(__name__)


MADLAD_MODEL_NAME = "google/madlad400-3b-mt"
OPUS_MODEL_NAME = "Helsinki-NLP/opus-mt-ja-en"


class MadladTranslator(Translator):
    def __init__(
        self,
        model_name: str = MADLAD_MODEL_NAME,
        target_tag: str = "<2en>",
        glossary_path: Path | None = None,
    ) -> None:
        super().__init__(glossary_path)
        logger.info("Initializing MADLAD translator: model=%s target=%s", model_name, target_tag)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError as exc:
            logger.exception("MADLAD dependencies are missing")
            raise RuntimeError(
                "MADLAD translator dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        self._cache: dict[str, str] = {}
        self._target_tag = target_tag
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with local_hf_cache_only():
            try:
                self._tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "MADLAD translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_madlad"
                ) from exc
        model_kwargs = {"low_cpu_mem_usage": True}
        if self._device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        with local_hf_cache_only():
            try:
                self._model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    local_files_only=True,
                    **model_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    "MADLAD translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_madlad"
                ) from exc
        self._model.to(self._device)
        self._model.eval()
        logger.info("MADLAD translator ready on %s", self._device)

    def translate(self, text: str) -> str:
        if text not in self._cache:
            logger.debug("MADLAD translating text=%s", shorten(text))
            self._cache[text] = self._translate_normalized(text)
            logger.debug("MADLAD translated result=%s", shorten(self._cache[text]))
        else:
            logger.debug("MADLAD translation cache hit for text=%s", shorten(text))
        return self._cache[text]

    def _translate_normalized(self, text: str) -> str:
        normalized = normalize_japanese_for_translation(text)
        prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
        phrase = translate_known_phrase(normalized, self._glossary) or translate_known_phrase(prepared, self._glossary)
        if phrase is not None:
            logger.debug("Phrasebook translation: source=%s result=%s", shorten(normalized), shorten(phrase))
            return phrase

        prompt = f"{self._target_tag} {prepared}"
        with local_hf_cache_only():
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=True,
                )
        translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return postprocess_translation(normalized, prepared, translated, self._glossary)


class OpusTranslator(Translator):
    def __init__(self, model_name: str = OPUS_MODEL_NAME, glossary_path: Path | None = None) -> None:
        super().__init__(glossary_path)
        logger.info("Initializing OPUS translator: model=%s", model_name)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            logger.exception("OPUS dependencies are missing")
            raise RuntimeError(
                "OPUS translator dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        self._cache: dict[str, str] = {}
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with local_hf_cache_only():
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "OPUS translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_opus"
                ) from exc
            try:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "OPUS translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_opus"
                ) from exc
        self._model.to(self._device)
        self._model.eval()
        logger.info("OPUS translator ready on %s", self._device)

    def translate(self, text: str) -> str:
        if text not in self._cache:
            logger.debug("OPUS translating text=%s", shorten(text))
            self._cache[text] = self._translate_normalized(text)
            logger.debug("OPUS translated result=%s", shorten(self._cache[text]))
        else:
            logger.debug("OPUS translation cache hit for text=%s", shorten(text))
        return self._cache[text]

    def translate_with_context(
        self,
        text: str,
        *,
        before: str | None = None,
        after: str | None = None,
        before_contexts: tuple[str, ...] = (),
        after_contexts: tuple[str, ...] = (),
        visual_facts: tuple[str, ...] = (),
    ) -> str:
        _ = before_contexts, after_contexts, visual_facts
        normalized = normalize_japanese_for_translation(text)
        if not should_use_context_translation(normalized, before=before, after=after):
            return self.translate(text)

        context_units = [
            normalize_japanese_for_translation(value)
            for value in (before, text, after)
            if value
        ]
        joined = "\u3002".join(context_units)
        logger.debug("OPUS translating with context: text=%s joined=%s", shorten(text), shorten(joined))
        contextual = self._translate_normalized(joined)
        focused = self.translate(text)
        return choose_contextual_translation(
            focused=focused,
            contextual=contextual,
            source_text=normalized,
            before_count=1 if before else 0,
            source_unit_count=max(1, len(split_japanese_translation_units(normalized))),
        )

    def _translate_normalized(self, text: str) -> str:
        normalized = normalize_japanese_for_translation(text)
        prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
        phrase = translate_known_phrase(normalized, self._glossary) or translate_known_phrase(prepared, self._glossary)
        if phrase is not None:
            logger.debug("Phrasebook translation: source=%s result=%s", shorten(normalized), shorten(phrase))
            return phrase

        units = split_japanese_translation_units(prepared)
        if len(units) > 1:
            translations = [self._translate_unit(unit) for unit in units]
            translated = " ".join(translation for translation in translations if translation).strip()
            return postprocess_translation(normalized, prepared, translated, self._glossary)

        translated = self._translate_unit(prepared)
        return postprocess_translation(normalized, prepared, translated, self._glossary)

    def _translate_unit(self, text: str) -> str:
        unit = clean_translation_unit(text)
        phrase = translate_known_phrase(unit, self._glossary)
        if phrase is not None:
            return phrase

        with local_hf_cache_only():
            inputs = self._tokenizer(unit, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=5,
                    do_sample=False,
                    early_stopping=True,
                )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


@contextmanager
def local_hf_cache_only():
    previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    captured_stderr = StringIO()
    try:
        with redirect_stderr(captured_stderr):
            yield
    finally:
        restore_env_var("TRANSFORMERS_OFFLINE", previous_transformers_offline)
        restore_env_var("HF_HUB_OFFLINE", previous_hf_offline)
        captured = captured_stderr.getvalue().strip()
        if captured:
            logger.debug("Suppressed local Hugging Face load stderr: %s", shorten(captured))


def restore_env_var(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
