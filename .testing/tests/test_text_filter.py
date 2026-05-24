from __future__ import annotations

import unittest

from manga_local_translator.detect_types import TextBlock
from manga_local_translator.text_filter import (
    fallback_translation,
    filter_text_blocks,
    suspected_bad_translation,
    unusable_translation_reason,
)


def block(text: str, *, detector: str = "ctd", box: tuple[int, int, int, int] = (10, 10, 40, 40)) -> TextBlock:
    return TextBlock(text=text, box=box, confidence=80.0, detector=detector)


class TextFilterTests(unittest.TestCase):
    def test_filter_keeps_common_short_ctd_reactions(self) -> None:
        kept, skipped = filter_text_blocks([block("くっ"), block("A")], width=500, height=500)

        self.assertEqual([item.text for item in kept], ["くっ"])
        self.assertEqual(skipped[0]["reason"], "non_japanese_text")

    def test_filter_skips_symbol_only_and_oversized_noise(self) -> None:
        huge_noise = block("あ", box=(0, 0, 400, 400))
        kept, skipped = filter_text_blocks([block("!!!"), huge_noise], width=500, height=500)

        self.assertEqual(kept, [])
        self.assertEqual([item["reason"] for item in skipped], ["punctuation_or_symbols_only", "single_character_fragment"])

    def test_unusable_translation_and_fallback_rules(self) -> None:
        self.assertEqual(unusable_translation_reason("母", "", translator_name="qwen"), "empty_translation")
        self.assertEqual(unusable_translation_reason("母", "母", translator_name="qwen"), "non_english_translation")
        self.assertIsNone(unusable_translation_reason("母", "Mom", translator_name="qwen"))
        self.assertEqual(fallback_translation("母", reason="empty_translation"), "...")
        self.assertEqual(fallback_translation("母", reason="mojibake_translation"), "母")

    def test_suspected_bad_translation_catches_debug_or_garbage_outputs(self) -> None:
        self.assertTrue(suspected_bad_translation(""))
        self.assertTrue(suspected_bad_translation("..."))
        self.assertTrue(suspected_bad_translation("<think>draft</think>"))
        self.assertTrue(suspected_bad_translation("one-two-three-four-five-six"))
        self.assertFalse(suspected_bad_translation("I am glad."))


if __name__ == "__main__":
    unittest.main()
