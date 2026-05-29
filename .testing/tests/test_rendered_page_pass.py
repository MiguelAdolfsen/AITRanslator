from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from manga_local_translator.config import PipelineConfig
from manga_local_translator.detect_types import TextBlock
from manga_local_translator.page_types import PreparedPage
from manga_local_translator.rendered_page_pass import render_prepared_page


def make_page() -> PreparedPage:
    block = TextBlock("source", (1, 2, 30, 40), 91.0, "ctd", metadata={"line_id": "line-001"})
    return PreparedPage(
        image_path=Path("page.png"),
        output_path=Path("out.png"),
        image_bgr=object(),
        width=100,
        height=100,
        raw_blocks=[block],
        render_blocks=[block],
        skipped_blocks=[],
        grouping_report=[],
        page_order_report=[{"source_text": block.text, "box": block.box, "page_order": 1, "line_id": "line-001"}],
        translations={"line-001": "A translation that may need vision repair."},
        translation_contexts={"line-001": {}},
    )


class RenderedPagePassTests(unittest.TestCase):
    def test_vision_acceptance_triggers_second_fit_and_compaction_pass(self) -> None:
        page = make_page()
        layout = object()
        fit = object()
        fit_calls: list[str] = []
        compact_calls: list[str] = []

        def plan_text_fits(*_args, **_kwargs):
            fit_calls.append("fit")
            return [fit]

        def compact_translations_for_render(*_args, **_kwargs):
            compact_calls.append("compact")

        def apply_vision_repair(**kwargs):
            kwargs["translation_contexts"]["line-001"]["vision_accepted"] = True
            return object()

        with (
            patch("manga_local_translator.rendered_page_pass.plan_render_layouts", return_value=[layout]),
            patch("manga_local_translator.rendered_page_pass.plan_text_fits", side_effect=plan_text_fits),
            patch("manga_local_translator.rendered_page_pass.compact_translations_for_render", side_effect=compact_translations_for_render),
            patch("manga_local_translator.rendered_page_pass.apply_vision_repair", side_effect=apply_vision_repair),
            patch("manga_local_translator.rendered_page_pass.blocks_for_erasing", return_value=page.render_blocks),
            patch("manga_local_translator.rendered_page_pass.erase_text", return_value="cleaned"),
            patch("manga_local_translator.rendered_page_pass.render_translations", return_value="rendered"),
            patch("manga_local_translator.rendered_page_pass.write_image_with_fallback", return_value=page.output_path),
        ):
            actual_output = render_prepared_page(page, PipelineConfig())

        self.assertEqual(actual_output, page.output_path)
        self.assertEqual(fit_calls, ["fit", "fit", "fit", "fit"])
        self.assertEqual(compact_calls, ["compact", "compact"])

    def test_skip_render_writes_translation_only_debug_report_when_debug_enabled(self) -> None:
        page = make_page()

        with patch("manga_local_translator.rendered_page_pass.write_debug_report") as write_debug_report:
            actual_output = render_prepared_page(page, PipelineConfig(skip_render=True, debug=True))

        self.assertEqual(actual_output, page.output_path)
        self.assertEqual(write_debug_report.call_count, 1)
        render_layouts = write_debug_report.call_args.args[9]
        render_fits = write_debug_report.call_args.args[10]
        self.assertEqual(render_layouts[0].render_box, page.render_blocks[0].box)
        self.assertEqual(render_fits, [None])


if __name__ == "__main__":
    unittest.main()
