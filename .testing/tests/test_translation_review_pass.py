from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from manga_local_translator.cache_policy import TranslationCachePlan, translation_cache_plan
from manga_local_translator.config import PipelineConfig
from manga_local_translator.detect_types import TextBlock
from manga_local_translator.page_types import PreparedPage
from manga_local_translator.translation_review_pass import TranslationReviewPass


def make_page(name: str) -> PreparedPage:
    block = TextBlock(
        "対象",
        (0, 0, 10, 10),
        90,
        metadata={"line_id": f"{name}_001"},
    )
    return PreparedPage(
        image_path=Path(f"{name}.png"),
        output_path=Path(f"out/{name}.png"),
        image_bgr=object(),
        width=100,
        height=100,
        raw_blocks=[block],
        render_blocks=[block],
        skipped_blocks=[],
        grouping_report=[],
        page_order_report=[{"source_text": block.text, "box": block.box, "page_order": 1, "line_id": f"{name}_001"}],
        translations={f"{name}_001": "Current."},
        translation_contexts={f"{name}_001": {"qwen_used": True}},
    )


class TranslationReviewPassTests(unittest.TestCase):
    def test_review_pass_runs_critic_and_fallback_and_writes_stage_caches(self) -> None:
        pages = [make_page("page1"), make_page("page2")]
        config = PipelineConfig(
            translator="qwen",
            qwen_mode="line",
            qwen_critic_model_path=Path("critic-model"),
            qwen_fallback_model_path=Path("fallback-model"),
        )
        cache_plan = translation_cache_plan(config)
        saved: list[Path] = []
        released: list[object] = []
        translators = {"critic-model": object(), "fallback-model": object()}

        def build_translator(name: str, **kwargs):
            self.assertEqual(name, "qwen")
            return translators[str(kwargs["qwen_model_path"])]

        with (
            patch("manga_local_translator.translation_review_pass.apply_translation_evidence_to_pages", return_value=object()),
            patch("manga_local_translator.translation_review_pass.apply_qwen_critic_reviews", side_effect=[(2, 1), (1, 1)]),
            patch("manga_local_translator.translation_review_pass.apply_qwen_fallback_translations", side_effect=[(3, 2), (1, 1)]),
        ):
            summary = TranslationReviewPass(
                config=config,
                cache_plan=cache_plan,
                work_dir=Path("work"),
                translator_factory=build_translator,
                cache_writer=lambda _page, path: saved.append(path),
                release_translator=lambda translator: released.append(translator),
            ).run(
                pages,
                primary_jobs=[],
                hybrid_cached_pages=set(),
                critic_cached_pages=set(),
            )

        self.assertEqual(summary.critic_attempted, 3)
        self.assertEqual(summary.critic_flagged, 2)
        self.assertEqual(summary.fallback_attempted, 4)
        self.assertEqual(summary.fallback_accepted, 3)
        self.assertEqual(
            saved,
            [
                Path("work") / "0001-page1.critic-line-evidence-critic.json",
                Path("work") / "0002-page2.critic-line-evidence-critic.json",
                Path("work") / "0001-page1.hybrid-line-evidence-critic.json",
                Path("work") / "0002-page2.hybrid-line-evidence-critic.json",
            ],
        )
        self.assertEqual(released, [translators["critic-model"], translators["fallback-model"]])

    def test_review_pass_owns_cat_retry_and_releases_primary_translator(self) -> None:
        pages = [make_page("page1"), make_page("page2")]
        primary_translator = object()
        primary_paths = [Path("work/page1.primary.json"), Path("work/page2.primary.json")]
        saved: list[Path] = []
        released: list[object] = []

        with patch("manga_local_translator.translation_review_pass.retry_cat_failures", side_effect=[(1, 1), (0, 0)]):
            summary = TranslationReviewPass(
                config=PipelineConfig(translator="cat"),
                cache_plan=TranslationCachePlan(
                    primary_stage="primary",
                    critic_stage="critic",
                    hybrid_stage="hybrid",
                    resume_lookup_order=("hybrid", "primary"),
                    render_only_lookup_order=("primary", "hybrid"),
                    final_stage="primary",
                ),
                work_dir=Path("work"),
                translator_factory=lambda *_args, **_kwargs: self.fail("no Qwen translator should be built"),
                cache_writer=lambda _page, path: saved.append(path),
                release_translator=lambda translator: released.append(translator),
            ).run(
                pages,
                primary_jobs=[
                    (1, pages[0], primary_paths[0]),
                    (2, pages[1], primary_paths[1]),
                ],
                hybrid_cached_pages=set(),
                critic_cached_pages=set(),
                primary_translator=primary_translator,
            )

        self.assertEqual(summary.cat_retry_attempted, 1)
        self.assertEqual(summary.cat_retry_accepted, 1)
        self.assertEqual(saved, [primary_paths[0]])
        self.assertEqual(released, [primary_translator])


if __name__ == "__main__":
    unittest.main()
