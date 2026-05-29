from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
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
    def test_render_uses_translation_after_compaction_stabilizes(self) -> None:
        page = make_page()
        page.translations["line-001"] = "First wording"
        layout = object()
        rendered_translations: dict[str, str] | None = None

        def plan_text_fits(_image, _blocks, translations, **_kwargs):
            text = translations["line-001"]
            return [SimpleNamespace(font_size=6 if text != "Final wording" else 14, clipped=text != "Final wording")]

        def compact_translations_for_render(_image, _blocks, translations, **_kwargs):
            if translations["line-001"] == "First wording":
                translations["line-001"] = "Second wording"
                return 1
            if translations["line-001"] == "Second wording":
                translations["line-001"] = "Final wording"
                return 1
            return 0

        def render_translations(_image, _blocks, translations, **_kwargs):
            nonlocal rendered_translations
            rendered_translations = dict(translations)
            return "rendered"

        with (
            patch("manga_local_translator.rendered_page_pass.plan_render_layouts", return_value=[layout]),
            patch("manga_local_translator.rendered_page_pass.plan_text_fits", side_effect=plan_text_fits),
            patch("manga_local_translator.rendered_page_pass.compact_translations_for_render", side_effect=compact_translations_for_render),
            patch("manga_local_translator.rendered_page_pass.apply_vision_repair", return_value=None),
            patch("manga_local_translator.rendered_page_pass.blocks_for_erasing", return_value=page.render_blocks),
            patch("manga_local_translator.rendered_page_pass.erase_text", return_value="cleaned"),
            patch("manga_local_translator.rendered_page_pass.render_translations", side_effect=render_translations),
            patch("manga_local_translator.rendered_page_pass.write_image_with_fallback", return_value=page.output_path),
        ):
            render_prepared_page(page, PipelineConfig())

        self.assertEqual(rendered_translations, {"line-001": "Final wording"})

    def test_stale_vision_acceptance_context_does_not_trigger_post_vision_compaction(self) -> None:
        page = make_page()
        page.translations["line-001"] = "Original wording"
        page.translation_contexts["line-001"] = {"vision_accepted": True}
        layout = object()
        compact_calls = 0
        rendered_translations: dict[str, str] | None = None

        def plan_text_fits(*_args, **_kwargs):
            return [SimpleNamespace(font_size=6, clipped=True)]

        def compact_translations_for_render(_image, _blocks, translations, **_kwargs):
            nonlocal compact_calls
            compact_calls += 1
            if compact_calls == 1:
                translations["line-001"] = "First compacted"
                return 1
            if compact_calls == 3:
                translations["line-001"] = "Stale post vision compacted"
                return 1
            return 0

        def render_translations(_image, _blocks, translations, **_kwargs):
            nonlocal rendered_translations
            rendered_translations = dict(translations)
            return "rendered"

        with (
            patch("manga_local_translator.rendered_page_pass.plan_render_layouts", return_value=[layout]),
            patch("manga_local_translator.rendered_page_pass.plan_text_fits", side_effect=plan_text_fits),
            patch("manga_local_translator.rendered_page_pass.compact_translations_for_render", side_effect=compact_translations_for_render),
            patch("manga_local_translator.rendered_page_pass.apply_vision_repair", return_value=None),
            patch("manga_local_translator.rendered_page_pass.blocks_for_erasing", return_value=page.render_blocks),
            patch("manga_local_translator.rendered_page_pass.erase_text", return_value="cleaned"),
            patch("manga_local_translator.rendered_page_pass.render_translations", side_effect=render_translations),
            patch("manga_local_translator.rendered_page_pass.write_image_with_fallback", return_value=page.output_path),
        ):
            render_prepared_page(page, PipelineConfig())

        self.assertEqual(rendered_translations, {"line-001": "First compacted"})

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

    def test_debug_report_uses_final_fits_after_vision_repair_replans(self) -> None:
        page = make_page()
        page.translations["line-001"] = "Original wording"
        layout = object()
        stale_fit = object()
        final_fit = object()

        def plan_text_fits(_image, _blocks, translations, **_kwargs):
            if translations["line-001"] == "Vision repaired wording":
                return [final_fit]
            return [stale_fit]

        def apply_vision_repair(**kwargs):
            kwargs["translation_contexts"]["line-001"]["vision_accepted"] = True
            kwargs["translations"]["line-001"] = "Vision repaired wording"
            return object()

        with (
            patch("manga_local_translator.rendered_page_pass.plan_render_layouts", return_value=[layout]),
            patch("manga_local_translator.rendered_page_pass.plan_text_fits", side_effect=plan_text_fits),
            patch("manga_local_translator.rendered_page_pass.compact_translations_for_render", return_value=0),
            patch("manga_local_translator.rendered_page_pass.apply_vision_repair", side_effect=apply_vision_repair),
            patch("manga_local_translator.rendered_page_pass.erase_text", return_value="cleaned"),
            patch("manga_local_translator.rendered_page_pass.render_translations", return_value="rendered"),
            patch("manga_local_translator.rendered_page_pass.write_image_with_fallback", return_value=page.output_path),
            patch("manga_local_translator.rendered_page_pass.write_debug_report") as write_debug_report,
            patch("manga_local_translator.rendered_page_pass.write_debug_image"),
        ):
            render_prepared_page(page, PipelineConfig(debug=True))

        debug_render_fits = write_debug_report.call_args.args[10]
        self.assertEqual(debug_render_fits, [final_fit])
        self.assertNotEqual(debug_render_fits, [stale_fit])

    def test_debug_report_render_and_overlay_use_same_final_lifecycle_state(self) -> None:
        page = make_page()
        page.translations["line-001"] = "First wording"
        final_layouts = [object()]
        stale_fit = object()
        final_fit = object()
        actual_output_path = Path("actual-output.png")

        def plan_text_fits(_image, _blocks, translations, **_kwargs):
            if translations["line-001"] == "Final wording":
                return [final_fit]
            return [stale_fit]

        def compact_translations_for_render(_image, _blocks, translations, **_kwargs):
            if translations["line-001"] == "First wording":
                translations["line-001"] = "Final wording"
                return 1
            return 0

        with (
            patch("manga_local_translator.rendered_page_pass.plan_render_layouts", return_value=final_layouts),
            patch("manga_local_translator.rendered_page_pass.plan_text_fits", side_effect=plan_text_fits),
            patch("manga_local_translator.rendered_page_pass.compact_translations_for_render", side_effect=compact_translations_for_render),
            patch("manga_local_translator.rendered_page_pass.apply_vision_repair", return_value=None),
            patch("manga_local_translator.rendered_page_pass.blocks_for_erasing", return_value=page.render_blocks),
            patch("manga_local_translator.rendered_page_pass.erase_text", return_value="cleaned"),
            patch("manga_local_translator.rendered_page_pass.render_translations", return_value="rendered") as render_translations,
            patch("manga_local_translator.rendered_page_pass.write_image_with_fallback", return_value=actual_output_path),
            patch("manga_local_translator.rendered_page_pass.write_debug_report") as write_debug_report,
            patch("manga_local_translator.rendered_page_pass.write_debug_image") as write_debug_image,
        ):
            returned_output_path = render_prepared_page(page, PipelineConfig(debug=True))

        self.assertEqual(returned_output_path, actual_output_path)
        self.assertIs(write_debug_report.call_args.args[9], final_layouts)
        self.assertEqual(write_debug_report.call_args.args[10], [final_fit])
        self.assertEqual(write_debug_report.call_args.args[5]["line-001"], "Final wording")
        self.assertIs(render_translations.call_args.kwargs["render_layouts"], final_layouts)
        self.assertIs(render_translations.call_args.args[2], page.translations)
        self.assertEqual(render_translations.call_args.args[2]["line-001"], "Final wording")
        self.assertIs(write_debug_image.call_args.args[4], final_layouts)
        self.assertEqual(write_debug_image.call_args.args[5], actual_output_path)

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
