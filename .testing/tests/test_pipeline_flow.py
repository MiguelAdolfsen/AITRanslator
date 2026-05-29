from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from manga_local_translator.config import PipelineConfig
from manga_local_translator.page_types import PreparedPage
from manga_local_translator.pipeline import process_folder


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


if __name__ == "__main__":
    unittest.main()
