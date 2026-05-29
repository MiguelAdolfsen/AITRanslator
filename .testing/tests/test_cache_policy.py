from __future__ import annotations

import unittest
from pathlib import Path

from manga_local_translator.cache_policy import render_only_translation_stages, translation_cache_stage
from manga_local_translator.config import PipelineConfig


class CachePolicyTests(unittest.TestCase):
    def test_qwen_cache_stage_includes_mode_and_review_state(self) -> None:
        stage = translation_cache_stage(
            PipelineConfig(
                translator="qwen",
                qwen_mode="page",
                qwen_critic_model_path=Path("critic-model"),
                qwen_fallback_model_path=Path("fallback-model"),
                vision_facts_enabled=True,
            ),
            "hybrid",
        )

        self.assertEqual(stage, "hybrid-page-evidence-vision-facts-critic")

    def test_plain_translator_cache_stage_is_original_stage_name(self) -> None:
        self.assertEqual(translation_cache_stage(PipelineConfig(translator="opus"), "primary"), "primary")

    def test_render_only_translation_stages_prefers_final_available_stage(self) -> None:
        stages = render_only_translation_stages(
            PipelineConfig(
                translator="qwen",
                qwen_mode="line",
                qwen_critic_model_path=Path("critic-model"),
                qwen_fallback_model_path=Path("fallback-model"),
            )
        )

        self.assertEqual(
            stages,
            [
                "hybrid-line-evidence-critic",
                "critic-line-evidence-critic",
                "primary-line-evidence-critic",
            ],
        )

    def test_render_only_translation_stages_includes_fallback_names_when_flags_are_absent(self) -> None:
        stages = render_only_translation_stages(PipelineConfig(translator="qwen", qwen_mode="line"))

        self.assertEqual(stages, ["primary-line", "hybrid-line", "critic-line"])


if __name__ == "__main__":
    unittest.main()
