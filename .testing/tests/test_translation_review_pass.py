from __future__ import annotations

import tempfile
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
    def test_review_pass_selects_resume_jobs_and_runs_primary_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pages = [make_page("page1"), make_page("page2"), make_page("page3"), make_page("page4")]
            config = PipelineConfig(
                translator="qwen",
                qwen_mode="line",
                qwen_critic_model_path=Path("critic-model"),
                qwen_fallback_model_path=Path("fallback-model"),
                resume=True,
            )
            cache_plan = translation_cache_plan(config)
            work_dir = Path(tmp)
            for stage in ("hybrid", "critic"):
                cache_plan.translation_cache_path(work_dir, 1, pages[0].image_path, stage).write_text("{}", encoding="utf-8")
            cache_plan.translation_cache_path(work_dir, 2, pages[1].image_path, "critic").write_text("{}", encoding="utf-8")
            cache_plan.translation_cache_path(work_dir, 3, pages[2].image_path, "primary").write_text("{}", encoding="utf-8")
            saved: list[Path] = []
            released: list[object] = []
            translated: list[str] = []
            loaded: list[Path] = []
            translators = {
                "primary": object(),
                "critic-model": object(),
                "fallback-model": object(),
            }

            def build_translator(name: str, **kwargs):
                if kwargs.get("qwen_model_path") == Path("critic-model"):
                    return translators["critic-model"]
                if kwargs.get("qwen_model_path") == Path("fallback-model"):
                    return translators["fallback-model"]
                return translators["primary"]

            with (
                patch("manga_local_translator.translation_review_page.TranslationReviewPage.apply_translation_evidence_to_pages", return_value=object()),
                patch("manga_local_translator.translation_review_page.TranslationReviewPage.run_qwen_critic", side_effect=[(1, 1), (1, 0)]),
                patch("manga_local_translator.translation_review_page.TranslationReviewPage.run_qwen_fallback", side_effect=[(1, 1), (1, 0), (1, 1)]),
            ):
                summary = TranslationReviewPass(
                    config=config,
                    cache_plan=cache_plan,
                    work_dir=work_dir,
                    translator_factory=build_translator,
                    primary_page_translator=lambda page, translator, _config: translated.append(page.image_path.name),
                    cache_writer=lambda _page, path: saved.append(path),
                    cache_loader=lambda _page, path: loaded.append(path),
                    release_translator=lambda translator: released.append(translator),
                ).run(pages)

            self.assertEqual(translated, ["page4.png"])
            self.assertEqual(
                loaded,
                [
                    cache_plan.translation_cache_path(work_dir, 1, pages[0].image_path, "hybrid"),
                    cache_plan.translation_cache_path(work_dir, 2, pages[1].image_path, "critic"),
                    cache_plan.translation_cache_path(work_dir, 3, pages[2].image_path, "primary"),
                ],
            )
            self.assertEqual(summary.critic_attempted, 2)
            self.assertEqual(summary.critic_flagged, 1)
            self.assertEqual(summary.fallback_attempted, 3)
            self.assertEqual(summary.fallback_accepted, 2)
            self.assertIn(cache_plan.translation_cache_path(work_dir, 4, pages[3].image_path, "primary"), saved)
            self.assertIn(cache_plan.translation_cache_path(work_dir, 3, pages[2].image_path, "critic"), saved)
            self.assertIn(cache_plan.translation_cache_path(work_dir, 4, pages[3].image_path, "hybrid"), saved)
            self.assertEqual(
                released,
                [translators["primary"], translators["critic-model"], translators["fallback-model"]],
            )

    def test_review_pass_skips_work_for_fully_hybrid_cached_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pages = [make_page("page1"), make_page("page2")]
            config = PipelineConfig(
                translator="qwen",
                qwen_mode="line",
                qwen_critic_model_path=Path("critic-model"),
                qwen_fallback_model_path=Path("fallback-model"),
                resume=True,
            )
            cache_plan = translation_cache_plan(config)
            work_dir = Path(tmp)
            for page_index, page in enumerate(pages, start=1):
                cache_plan.translation_cache_path(work_dir, page_index, page.image_path, "hybrid").write_text("{}", encoding="utf-8")
            pages[0].translation_contexts["page1_001"]["qwen_fallback_attempted"] = True
            pages[0].translation_contexts["page1_001"]["qwen_fallback_accepted"] = True
            loaded: list[str] = []

            with patch("manga_local_translator.translation_review_page.TranslationReviewPage.apply_translation_evidence_to_pages", return_value=object()):
                summary = TranslationReviewPass(
                    config=config,
                    cache_plan=cache_plan,
                    work_dir=work_dir,
                    translator_factory=lambda *_args, **_kwargs: self.fail("no translators should be built"),
                    primary_page_translator=lambda *_args: self.fail("no primary translation should run"),
                    cache_writer=lambda *_args: self.fail("no caches should be written"),
                    cache_loader=lambda page, _path: loaded.append(page.image_path.name),
                    release_translator=lambda *_args: self.fail("no translators should be released"),
                ).run(pages)

            self.assertEqual(loaded, ["page1.png", "page2.png"])
            self.assertEqual(summary.critic_attempted, 0)
            self.assertEqual(summary.critic_flagged, 0)
            self.assertEqual(summary.fallback_attempted, 1)
            self.assertEqual(summary.fallback_accepted, 1)

    def test_review_pass_runs_primary_and_review_work_for_uncached_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pages = [make_page("page1"), make_page("page2")]
            config = PipelineConfig(
                translator="qwen",
                qwen_mode="line",
                qwen_critic_model_path=Path("critic-model"),
                qwen_fallback_model_path=Path("fallback-model"),
                resume=False,
            )
            cache_plan = translation_cache_plan(config)
            work_dir = Path(tmp)
            translated: list[str] = []
            saved: list[Path] = []
            released: list[object] = []
            translators = {
                "primary": object(),
                "critic-model": object(),
                "fallback-model": object(),
            }

            def build_translator(_name: str, **kwargs):
                if kwargs.get("qwen_model_path") == Path("critic-model"):
                    return translators["critic-model"]
                if kwargs.get("qwen_model_path") == Path("fallback-model"):
                    return translators["fallback-model"]
                return translators["primary"]

            with (
                patch("manga_local_translator.translation_review_page.TranslationReviewPage.apply_translation_evidence_to_pages", return_value=object()),
                patch("manga_local_translator.translation_review_page.TranslationReviewPage.run_qwen_critic", side_effect=[(1, 1), (1, 1)]),
                patch("manga_local_translator.translation_review_page.TranslationReviewPage.run_qwen_fallback", side_effect=[(1, 1), (1, 1)]),
            ):
                summary = TranslationReviewPass(
                    config=config,
                    cache_plan=cache_plan,
                    work_dir=work_dir,
                    translator_factory=build_translator,
                    primary_page_translator=lambda page, _translator, _config: translated.append(page.image_path.name),
                    cache_writer=lambda _page, path: saved.append(path),
                    release_translator=lambda translator: released.append(translator),
                ).run(pages)

            self.assertEqual(translated, ["page1.png", "page2.png"])
            self.assertEqual(summary.critic_attempted, 2)
            self.assertEqual(summary.critic_flagged, 2)
            self.assertEqual(summary.fallback_attempted, 2)
            self.assertEqual(summary.fallback_accepted, 2)
            self.assertEqual(
                released,
                [translators["primary"], translators["critic-model"], translators["fallback-model"]],
            )
            self.assertEqual(
                saved,
                [
                    cache_plan.translation_cache_path(work_dir, 1, pages[0].image_path, "primary"),
                    cache_plan.translation_cache_path(work_dir, 2, pages[1].image_path, "primary"),
                    cache_plan.translation_cache_path(work_dir, 1, pages[0].image_path, "critic"),
                    cache_plan.translation_cache_path(work_dir, 2, pages[1].image_path, "critic"),
                    cache_plan.translation_cache_path(work_dir, 1, pages[0].image_path, "hybrid"),
                    cache_plan.translation_cache_path(work_dir, 2, pages[1].image_path, "hybrid"),
                ],
            )

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
            patch("manga_local_translator.translation_review_page.TranslationReviewPage.apply_translation_evidence_to_pages", return_value=object()),
            patch("manga_local_translator.translation_review_page.TranslationReviewPage.run_qwen_critic", side_effect=[(2, 1), (1, 1)]),
            patch("manga_local_translator.translation_review_page.TranslationReviewPage.run_qwen_fallback", side_effect=[(3, 2), (1, 1)]),
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

        with patch("manga_local_translator.translation_review_page.TranslationReviewPage.run_cat_retry", side_effect=[(1, 1), (0, 0)]):
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
