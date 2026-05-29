from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from manga_local_translator.cache_policy import translation_cache_plan
from manga_local_translator.config import PipelineConfig
from manga_local_translator.page_types import PreparedPage
from manga_local_translator.pipeline import process_folder, process_folder_qwen_hybrid_batch


def _page(image_path: Path, output_path: Path, source_text: str) -> PreparedPage:
    return PreparedPage(
        image_path=image_path,
        output_path=output_path,
        image_bgr=None,
        width=1,
        height=1,
        raw_blocks=[],
        render_blocks=[],
        skipped_blocks=[],
        grouping_report=[],
        page_order_report=[{"source_text": source_text}],
    )


class PipelineFlowTests(unittest.TestCase):
    def test_standard_flow_uses_prepared_page_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            images = [input_dir / "001.png", input_dir / "002.png"]
            translator = object()
            seen_contexts: list[str | None] = []

            def prepare_side_effect(
                image_path: Path,
                output_path: Path,
                _config: PipelineConfig,
                *,
                chapter_context_before: str | None = None,
            ) -> PreparedPage:
                seen_contexts.append(chapter_context_before)
                return _page(image_path, output_path, image_path.stem)

            with (
                patch("manga_local_translator.pipeline.iter_images", return_value=images),
                patch("manga_local_translator.translate.build_translator", return_value=translator),
                patch("manga_local_translator.pipeline.prepare_page_for_translation", side_effect=prepare_side_effect) as prepare,
                patch("manga_local_translator.pipeline.translate_prepared_page") as translate,
                patch("manga_local_translator.pipeline.render_prepared_page") as render,
            ):
                process_folder(
                    input_dir,
                    output_dir,
                    PipelineConfig(detector="mock", ocr_engine="mock", translator="none"),
                )

            self.assertEqual(seen_contexts, [None, "001"])
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(translate.call_count, 2)
            self.assertEqual(render.call_count, 2)
            self.assertIs(translate.call_args_list[0].args[1], translator)
            self.assertIs(translate.call_args_list[1].args[1], translator)
            self.assertIs(render.call_args_list[0].args[0], translate.call_args_list[0].args[0])
            self.assertIs(render.call_args_list[1].args[0], translate.call_args_list[1].args[0])

    def test_render_only_missing_translation_cache_lists_expected_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            work_dir = root / "work"
            input_dir.mkdir()
            work_dir.mkdir()
            image_path = input_dir / "001.png"
            config = PipelineConfig(
                detector="mock",
                ocr_engine="mock",
                translator="qwen",
                qwen_mode="line",
                render_only=True,
                work_dir=work_dir,
            )
            translation_cache_plan(config).prepared_cache_path(work_dir, 1, image_path).write_text("{}", encoding="utf-8")

            with patch("manga_local_translator.pipeline.iter_images", return_value=[image_path]):
                with self.assertRaisesRegex(RuntimeError, "expected one of: primary-line, hybrid-line, critic-line"):
                    process_folder(input_dir, output_dir, config)

    def test_hybrid_flow_delegates_review_work_to_review_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            work_dir = root / "work"
            input_dir.mkdir()
            work_dir.mkdir()
            image_paths = [input_dir / f"00{index}.png" for index in range(1, 5)]
            config = PipelineConfig(
                detector="mock",
                ocr_engine="mock",
                translator="qwen",
                qwen_mode="line",
                qwen_critic_model_path=Path("critic-model"),
                qwen_fallback_model_path=Path("fallback-model"),
                resume=True,
                work_dir=work_dir,
            )
            cache_plan = translation_cache_plan(config)
            for stage in ("hybrid", "critic"):
                cache_plan.translation_cache_path(work_dir, 1, image_paths[0], stage).write_text("{}", encoding="utf-8")
            cache_plan.translation_cache_path(work_dir, 2, image_paths[1], "critic").write_text("{}", encoding="utf-8")
            cache_plan.translation_cache_path(work_dir, 3, image_paths[2], "primary").write_text("{}", encoding="utf-8")
            captured: dict[str, object] = {}

            class FakeTranslationReviewPass:
                def __init__(self, **_kwargs) -> None:
                    pass

                def run(self, pages, **kwargs):
                    captured["pages"] = pages
                    captured.update(kwargs)
                    return SimpleNamespace(fallback_attempted=0, fallback_accepted=0)

            def prepare_side_effect(image_path: Path, output_path: Path, _config: PipelineConfig, **_kwargs) -> PreparedPage:
                return _page(image_path, output_path, image_path.stem)

            with (
                patch("manga_local_translator.pipeline.prepare_page_for_translation", side_effect=prepare_side_effect),
                patch("manga_local_translator.translate.build_translator") as build_translator,
                patch("manga_local_translator.pipeline.translate_prepared_page") as translate,
                patch("manga_local_translator.pipeline.render_prepared_page"),
                patch("manga_local_translator.pipeline.TranslationReviewPass", FakeTranslationReviewPass),
            ):
                process_folder_qwen_hybrid_batch(input_dir, output_dir, image_paths, config)

            self.assertEqual(set(captured), {"pages"})
            self.assertEqual(len(captured["pages"]), 4)
            self.assertEqual(build_translator.call_count, 0)
            self.assertEqual(translate.call_count, 0)


if __name__ == "__main__":
    unittest.main()
