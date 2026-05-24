from __future__ import annotations

import logging
from pathlib import Path

from .logging_utils import shorten
from .translation_rules import load_translation_glossary

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, glossary_path: Path | None = None) -> None:
        self._glossary = load_translation_glossary(glossary_path)

    def translate(self, text: str) -> str:
        raise NotImplementedError

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
        _ = before, after, before_contexts, after_contexts, visual_facts
        return self.translate(text)


class NoopTranslator(Translator):
    def __init__(self, glossary_path: Path | None = None) -> None:
        super().__init__(glossary_path)

    def translate(self, text: str) -> str:
        logger.debug("Noop translation used for text=%s", shorten(text))
        return text
