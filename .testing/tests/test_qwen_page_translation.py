from __future__ import annotations

import unittest

from manga_local_translator.qwen_translator import QwenTranslator
from manga_local_translator.qwen_types import QwenGenerationSettings, QwenVerificationDecision
from manga_local_translator.translation_rules import TranslationGlossary


class FakeQwenTranslator(QwenTranslator):
    def __init__(self, raw_response: str) -> None:
        self._raw_response = raw_response
        self._cache = {}
        self._page_cache = {}
        self._debug = {}
        self._baseline_translator = None
        self._glossary_path = None
        self._glossary = TranslationGlossary((), {}, ())
        self._ollama_model_name = "fake-qwen"
        self._backend = "fake"

    def _baseline_translate(self, text: str) -> str:
        return "Baseline."

    def _run_prompt(self, prompt: str, *, settings: QwenGenerationSettings) -> str:
        self.last_prompt = prompt
        return self._raw_response

    def _translate_qwen(
        self,
        text: str,
        *,
        before: str | None,
        after: str | None,
        before_contexts: tuple[str, ...] = (),
        after_contexts: tuple[str, ...] = (),
    ) -> tuple[str, dict[str, object]]:
        return f"Fallback for {text}", {"qwen_used": True, "qwen_rejected": False, "qwen_final": f"Fallback for {text}"}

    def _verify_qwen_translation(
        self,
        *,
        source_text: str,
        before: str | None,
        after: str | None,
        before_contexts: tuple[str, ...] = (),
        after_contexts: tuple[str, ...] = (),
        baseline: str,
        candidate: str,
    ) -> QwenVerificationDecision:
        return QwenVerificationDecision(True, "candidate", "candidate is direct", candidate)


class QwenPageTranslationTests(unittest.TestCase):
    def test_page_translation_accepts_valid_items_and_falls_back_for_bad_items(self) -> None:
        translator = FakeQwenTranslator(
            '{"translations":['
            '{"id":1,"translation":"Hello!","confidence":0.9,"reason":"target"},'
            '{"id":2,"translation":"母","confidence":0.9,"reason":"bad"},'
            '{"id":4,"translation":"First duplicate.","confidence":0.9,"reason":"bad"},'
            '{"id":4,"translation":"Second duplicate.","confidence":0.9,"reason":"bad"},'
            '{"id":99,"translation":"Extra.","confidence":0.9,"reason":"extra"}'
            ']}'
        )
        items = [
            {"id": 1, "text": "\u3053\u3093\u306b\u3061\u306f"},
            {"id": 2, "text": "\u6bcd"},
            {"id": 3, "text": "\u307e\u3063\u3066"},
            {"id": 4, "text": "\u3088\u3057"},
        ]

        translated = translator.translate_page(items, previous_page_context="\u524d")

        self.assertEqual(translated[1], "Hello!")
        self.assertEqual(translated[2], "Fallback for \u6bcd")
        self.assertEqual(translated[3], "Fallback for \u307e\u3063\u3066")
        self.assertEqual(translated[4], "Fallback for \u3088\u3057")
        self.assertIn("Previous page final bubble context", translator.last_prompt)
        self.assertTrue(translator.debug_info_for("\u6bcd")["qwen_page_rejected"])
        self.assertEqual(translator.debug_info_for("\u307e\u3063\u3066")["qwen_page_reject_reason"], "missing_page_translation")
        self.assertEqual(translator.debug_info_for("\u3088\u3057")["qwen_page_reject_reason"], "duplicate_page_id")


if __name__ == "__main__":
    unittest.main()
