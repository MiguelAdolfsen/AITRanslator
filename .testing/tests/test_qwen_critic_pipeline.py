import unittest

from manga_local_translator.pipeline import (
    apply_qwen_fallback_translations,
    english_context_for_block,
    is_probable_standalone_japanese_name_or_credit,
    is_probable_short_sfx_or_reaction,
    qwen_critic_trigger_reasons,
    qwen_critic_should_trigger_repair,
    should_try_qwen_fallback,
)
from manga_local_translator.config import PipelineConfig
from manga_local_translator.detect_types import TextBlock
from manga_local_translator.qwen_types import QwenCriticDecision


class QwenCriticPipelineTests(unittest.TestCase):
    def test_english_context_uses_nearby_translated_lines(self) -> None:
        blocks = [
            TextBlock("一", (0, 0, 10, 10), 90),
            TextBlock("二", (20, 0, 30, 10), 90),
            TextBlock("三", (40, 0, 50, 10), 90),
            TextBlock("四", (60, 0, 70, 10), 90),
        ]
        page_order = [
            {"source_text": block.text, "box": block.box, "page_order": index}
            for index, block in enumerate(blocks, start=1)
        ]
        translations = {"一": "One.", "二": "Two.", "三": "Three.", "四": "Four."}

        before, after = english_context_for_block(blocks[2], translations, page_order, window=2)

        self.assertEqual(before, ("One.", "Two."))
        self.assertEqual(after, ("Four.",))

    def test_critic_triggers_on_awkward_family_passive(self) -> None:
        reasons = qwen_critic_trigger_reasons(
            "\u7236\u3082\u6bcd\u3082\u304d\u3089\u3044",
            "Father and Mother are hated!!",
            {"qwen_used": True},
        )

        self.assertIn("accepted_line_review", reasons)
        self.assertIn("risk_source_terms", reasons)
        self.assertIn("awkward_english_pattern", reasons)

    def test_critic_reviews_normal_accepted_dialogue(self) -> None:
        reasons = qwen_critic_trigger_reasons(
            "\u304d\u307f\u304c\u3042\u305f\u3089\u3057\u304f\u306f\u3044\u3063\u305f\u3057\u3093\u3058\u3093\u304f\u3093\u3060\u306d",
            "You're the new recruit, right?",
            {"qwen_used": True},
        )

        self.assertEqual(reasons, ("accepted_line_review",))

    def test_critic_does_not_review_rejected_or_unusable_lines(self) -> None:
        self.assertEqual(
            qwen_critic_trigger_reasons("\u6bcd", "...", {"qwen_rejected": True}),
            (),
        )
        self.assertEqual(
            qwen_critic_trigger_reasons("\u6bcd", "", {"qwen_used": True}),
            (),
        )
        self.assertEqual(
            qwen_critic_trigger_reasons("\u3046\u3080", "Hmm.", {"qwen_used": False, "qwen_reason": "phrasebook"}),
            (),
        )

    def test_critic_does_not_review_short_sfx_or_reactions(self) -> None:
        self.assertTrue(is_probable_short_sfx_or_reaction("\u304f\u3063"))
        self.assertTrue(is_probable_short_sfx_or_reaction("\u306a\u3063\u2026!?"))
        self.assertEqual(
            qwen_critic_trigger_reasons("\u304f\u3063", "Ngh.", {"qwen_used": True}),
            (),
        )

    def test_critic_does_not_review_short_kanji_name_or_credit(self) -> None:
        self.assertTrue(is_probable_standalone_japanese_name_or_credit("\u9060\u85e4\u9054\u54c9"))
        self.assertEqual(
            qwen_critic_trigger_reasons("\u9060\u85e4\u9054\u54c9", "Endo Tatsuya", {"qwen_repair_used": True}),
            (),
        )

    def test_critic_flag_triggers_fallback(self) -> None:
        self.assertTrue(
            should_try_qwen_fallback(
                "\u6bcd",
                "Mom.",
                {"qwen_critic_should_repair": True},
            )
        )

    def test_critic_style_only_issue_does_not_trigger_repair(self) -> None:
        self.assertFalse(
            qwen_critic_should_trigger_repair(
                QwenCriticDecision(False, "medium", ("awkward_literal", "tone_mismatch"), "style issue")
            )
        )
        self.assertTrue(
            qwen_critic_should_trigger_repair(
                QwenCriticDecision(False, "medium", ("omitted_term",), "missing target term")
            )
        )

    def test_fallback_uses_guided_repair_when_critic_flagged(self) -> None:
        class FakeFallback:
            def __init__(self) -> None:
                self.guided_calls = []
                self.debug = {}

            def repair_translation_with_guidance(self, text, **kwargs):
                self.guided_calls.append((text, kwargs))
                self.debug[text] = {
                    "qwen_model": "fake-q8",
                    "qwen_guided_repair": True,
                    "qwen_repair_accepted": True,
                }
                return "Guided repair."

            def debug_info_for(self, text):
                return self.debug.get(text, {})

        block = TextBlock("対象", (10, 10, 20, 20), 90)
        before = TextBlock("前", (30, 10, 40, 20), 90)
        after = TextBlock("後", (0, 10, 5, 20), 90)
        page_order = [
            {"source_text": before.text, "box": before.box, "page_order": 1, "context_after": block.text},
            {"source_text": block.text, "box": block.box, "page_order": 2, "context_before": before.text, "context_after": after.text},
            {"source_text": after.text, "box": after.box, "page_order": 3, "context_before": block.text},
        ]
        translations = {before.text: "Before.", block.text: "Current bad.", after.text: "After."}
        contexts = {
            block.text: {
                "qwen_critic_should_repair": True,
                "qwen_critic_issues": ["omitted_term"],
                "qwen_critic_reason": "missing term",
                "qwen_baseline": "Baseline.",
            }
        }
        fake = FakeFallback()

        attempted, accepted = apply_qwen_fallback_translations(
            [block],
            translations,
            contexts,
            page_order,
            fake,
            PipelineConfig(translator="qwen"),
        )

        self.assertEqual((attempted, accepted), (1, 1))
        self.assertEqual(translations[block.text], "Guided repair.")
        self.assertTrue(contexts[block.text]["qwen_fallback_guided_by_critic"])
        self.assertEqual(contexts[block.text]["qwen_fallback_before_translations"], ["Before."])
        self.assertEqual(contexts[block.text]["qwen_fallback_after_translations"], ["After."])
        self.assertEqual(fake.guided_calls[0][1]["critic_issues"], ("omitted_term",))


if __name__ == "__main__":
    unittest.main()
