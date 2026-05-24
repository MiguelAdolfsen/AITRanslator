import unittest

from manga_local_translator.pipeline import qwen_critic_trigger_reasons, should_try_qwen_fallback


class QwenCriticPipelineTests(unittest.TestCase):
    def test_critic_triggers_on_awkward_family_passive(self) -> None:
        reasons = qwen_critic_trigger_reasons(
            "\u7236\u3082\u6bcd\u3082\u304d\u3089\u3044",
            "Father and Mother are hated!!",
            {"qwen_used": True},
        )

        self.assertIn("risk_source_terms", reasons)
        self.assertIn("awkward_english_pattern", reasons)

    def test_critic_does_not_review_rejected_or_unusable_lines(self) -> None:
        self.assertEqual(
            qwen_critic_trigger_reasons("\u6bcd", "...", {"qwen_rejected": True}),
            (),
        )
        self.assertEqual(
            qwen_critic_trigger_reasons("\u6bcd", "", {"qwen_used": True}),
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


if __name__ == "__main__":
    unittest.main()
