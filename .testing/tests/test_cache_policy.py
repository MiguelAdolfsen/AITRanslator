from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from manga_local_translator.cache_policy import (
    render_only_translation_stages,
    translation_cache_plan,
    translation_cache_stage,
)
from manga_local_translator.config import PipelineConfig
from manga_local_translator.page_types import PreparedPage


def make_page(name: str) -> PreparedPage:
    return PreparedPage(
        image_path=Path(f"{name}.png"),
        output_path=Path(f"out/{name}.png"),
        image_bgr=object(),
        width=100,
        height=100,
        raw_blocks=[],
        render_blocks=[],
        skipped_blocks=[],
        grouping_report=[],
        page_order_report=[],
    )


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

    def test_cache_plan_names_all_translation_stages(self) -> None:
        plan = translation_cache_plan(
            PipelineConfig(
                translator="qwen",
                qwen_mode="page",
                qwen_critic_model_path=Path("critic-model"),
                qwen_fallback_model_path=Path("fallback-model"),
            )
        )

        self.assertEqual(plan.primary_stage, "primary-page-evidence-critic")
        self.assertEqual(plan.critic_stage, "critic-page-evidence-critic")
        self.assertEqual(plan.hybrid_stage, "hybrid-page-evidence-critic")
        self.assertEqual(plan.final_stage, "hybrid-page-evidence-critic")

    def test_cache_plan_keeps_resume_lookup_order(self) -> None:
        plan = translation_cache_plan(
            PipelineConfig(
                translator="qwen",
                qwen_mode="line",
                qwen_critic_model_path=Path("critic-model"),
                qwen_fallback_model_path=Path("fallback-model"),
            )
        )

        self.assertEqual(
            plan.resume_lookup_order,
            (
                "hybrid-line-evidence-critic",
                "critic-line-evidence-critic",
                "primary-line-evidence-critic",
            ),
        )

    def test_cache_plan_builds_page_paths(self) -> None:
        plan = translation_cache_plan(PipelineConfig(translator="qwen", qwen_mode="line"))

        self.assertEqual(
            plan.prepared_cache_path(Path("work"), 2, Path("page one.png")),
            Path("work") / "0002-page_one.prepared.json",
        )
        self.assertEqual(
            plan.translation_cache_path(Path("work"), 2, Path("page one.png"), "primary"),
            Path("work") / "0002-page_one.primary-line.json",
        )

    def test_cache_plan_finds_first_existing_render_only_translation_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = translation_cache_plan(
                PipelineConfig(
                    translator="qwen",
                    qwen_mode="line",
                    qwen_critic_model_path=Path("critic-model"),
                    qwen_fallback_model_path=Path("fallback-model"),
                )
            )
            work_dir = Path(tmp)
            image_path = Path("page one.png")
            expected = plan.translation_cache_path(work_dir, 2, image_path, "critic")
            expected.write_text("{}", encoding="utf-8")

            self.assertEqual(
                plan.find_render_only_translation_cache(work_dir, 2, image_path),
                expected,
            )

    def test_cache_plan_returns_none_when_render_only_translation_cache_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = translation_cache_plan(PipelineConfig(translator="qwen", qwen_mode="line"))

            self.assertIsNone(
                plan.find_render_only_translation_cache(Path(tmp), 2, Path("page one.png"))
            )

    def test_cache_plan_selects_hybrid_resume_jobs_from_mixed_cache_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = translation_cache_plan(
                PipelineConfig(
                    translator="qwen",
                    qwen_mode="line",
                    qwen_critic_model_path=Path("critic-model"),
                    qwen_fallback_model_path=Path("fallback-model"),
                )
            )
            work_dir = Path(tmp)
            pages = [make_page("page1"), make_page("page2"), make_page("page3"), make_page("page4")]
            loaded: list[tuple[str, Path]] = []

            for stage in ("hybrid", "critic"):
                plan.translation_cache_path(work_dir, 1, pages[0].image_path, stage).write_text("{}", encoding="utf-8")
            plan.translation_cache_path(work_dir, 2, pages[1].image_path, "critic").write_text("{}", encoding="utf-8")
            plan.translation_cache_path(work_dir, 3, pages[2].image_path, "primary").write_text("{}", encoding="utf-8")

            jobs = plan.select_hybrid_resume_cache_jobs(
                work_dir,
                pages,
                resume=True,
                load_translation_cache=lambda page, path: loaded.append((page.image_path.name, path)),
            )

            self.assertEqual(jobs.hybrid_cached_pages, {1})
            self.assertEqual(jobs.critic_cached_pages, {2})
            self.assertEqual(
                jobs.primary_jobs,
                [
                    (4, pages[3], plan.translation_cache_path(work_dir, 4, pages[3].image_path, "primary")),
                ],
            )
            self.assertEqual(
                loaded,
                [
                    ("page1.png", plan.translation_cache_path(work_dir, 1, pages[0].image_path, "hybrid")),
                    ("page2.png", plan.translation_cache_path(work_dir, 2, pages[1].image_path, "critic")),
                    ("page3.png", plan.translation_cache_path(work_dir, 3, pages[2].image_path, "primary")),
                ],
            )


if __name__ == "__main__":
    unittest.main()
