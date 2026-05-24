from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from manga_local_translator.render import (
    estimate_start_font_size,
    expand_box,
    fallback_render_box,
    inset_text_box,
    is_long_translation_from_narrow_vertical_source,
    measure_fitted_text,
    trim_render_box_away_from_source,
    render_text_length,
    wrap_text,
    split_long_word,
    merge_tiny_word_chunks,
    RenderLayout,
    TextFit,
)
from manga_local_translator.debug_report import render_layout_warnings


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (240, 180), "white")
        self.draw = ImageDraw.Draw(self.image)

    def test_estimate_uses_render_box_capacity_for_short_text(self) -> None:
        estimate = estimate_start_font_size(
            "What the heck?",
            (0, 0, 82, 50),
            source_box=(10, 10, 30, 32),
            base_font_size=28,
        )

        self.assertGreaterEqual(estimate, 10)

    def test_measure_fitted_text_no_longer_forces_tiny_short_translation(self) -> None:
        fit = measure_fitted_text(
            self.draw,
            "What the heck?",
            (0, 0, 82, 50),
            font_path=None,
            base_font_size=28,
            source_box=(10, 10, 30, 32),
        )

        self.assertFalse(fit.clipped)
        self.assertGreaterEqual(fit.font_size, 8)
        self.assertGreaterEqual(fit.attempted_font_size, fit.font_size)

    def test_long_translation_wraps_without_clipping_in_tall_bubble(self) -> None:
        fit = measure_fitted_text(
            self.draw,
            "This is the command room of the hideout.",
            (0, 0, 92, 120),
            font_path=None,
            base_font_size=28,
            source_box=(20, 20, 45, 95),
        )

        self.assertFalse(fit.clipped)
        self.assertGreaterEqual(fit.font_size, 8)
        self.assertGreater(fit.line_count, 1)
        self.assertEqual(fit.line_count, len(fit.lines))

    def test_split_long_word_keeps_chunks_inside_width(self) -> None:
        from manga_local_translator.render import load_font, text_width

        font = load_font(None, 12)
        chunks = split_long_word(self.draw, "supercalifragilistic", font, 45)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(text_width(self.draw, chunk, font) <= 45 for chunk in chunks))

    def test_render_text_length_discounts_spaces_and_punctuation(self) -> None:
        self.assertLess(render_text_length("What... the heck?!"), len("What... the heck?!"))

    def test_inset_text_box_keeps_safe_margin_for_bubbles(self) -> None:
        self.assertEqual(inset_text_box((10, 20, 130, 140)), (20, 30, 120, 130))
        self.assertEqual(inset_text_box((10, 20, 20, 28)), (10, 20, 20, 28))

    def test_large_fallback_render_box_stays_inside_detected_source_box(self) -> None:
        source = (320, 31, 474, 140)
        expanded = (294, 28, 501, 142)

        self.assertEqual(fallback_render_box(source, expanded), source)

    def test_small_fallback_render_box_can_still_expand_for_readability(self) -> None:
        source = (91, 218, 126, 249)
        expanded = (70, 214, 147, 252)

        self.assertEqual(fallback_render_box(source, expanded), expanded)

    def test_flat_horizontal_text_box_expands_height_for_readability(self) -> None:
        expanded = expand_box((388, 282, 461, 293), 500, 800, 2.2)

        self.assertGreaterEqual(expanded[3] - expanded[1], 45)
        self.assertGreater(expanded[2] - expanded[0], 120)

    def test_long_translation_from_narrow_vertical_source_caps_start_font(self) -> None:
        self.assertTrue(is_long_translation_from_narrow_vertical_source("Let's play! One person, three roles!", (2, 601, 27, 787)))

        estimate = estimate_start_font_size(
            "Let's play! One person, three roles!",
            (0, 596, 95, 791),
            source_box=(2, 601, 27, 787),
            base_font_size=28,
        )

        self.assertLessEqual(estimate, 16)

    def test_trim_render_box_away_from_other_source_below(self) -> None:
        trimmed = trim_render_box_away_from_source(
            (0, 596, 95, 791),
            (2, 601, 27, 787),
            (27, 695, 131, 752),
        )

        self.assertEqual(trimmed, (0, 596, 95, 690))

    def test_trim_render_box_keeps_box_when_source_overlaps(self) -> None:
        original = (0, 596, 95, 791)

        trimmed = trim_render_box_away_from_source(original, (2, 601, 27, 787), (10, 650, 20, 700))

        self.assertEqual(trimmed, original)

    def test_wrap_text_balances_lines_and_avoids_tiny_last_line(self) -> None:
        from manga_local_translator.render import load_font, text_width

        font = load_font(None, 14)
        text = "All right, come along, you lot!"
        lines = wrap_text(self.draw, text, font, 125)

        self.assertEqual(lines, ["All right, come", "along, you lot!"])
        widths = [text_width(self.draw, line, font) for line in lines]
        self.assertLess(max(widths) - min(widths), 45)

    def test_wrap_text_still_splits_overlong_words(self) -> None:
        from manga_local_translator.render import load_font, text_width

        font = load_font(None, 12)
        lines = wrap_text(self.draw, "supercalifragilistic word", font, 45)

        self.assertGreater(len(lines), 2)
        self.assertTrue(all(text_width(self.draw, line, font) <= 45 for line in lines))

    def test_merge_tiny_word_chunks_keeps_possessive_suffix_together(self) -> None:
        from manga_local_translator.render import load_font

        font = load_font(None, 28)
        chunks = merge_tiny_word_chunks(self.draw, ["Father'", "s"], font, 120)

        self.assertEqual(chunks, ["Father's"])

    def test_fit_prefers_cleaner_wrap_over_orphan_last_word(self) -> None:
        fit = measure_fitted_text(
            self.draw,
            "All right, come along, you lot! The candy shop on 2nd Street has been taken over by the bad guys! I'm going to save you.",
            (0, 0, 154, 109),
            font_path=None,
            base_font_size=28,
            source_box=(0, 0, 154, 109),
        )

        self.assertNotEqual(fit.lines[-1], "you.")
        self.assertEqual(fit.orphan_line_count, 0)
        self.assertNotIn("orphan_line", fit.wrap_warnings)
        self.assertGreater(fit.wrap_score, 0)
        self.assertGreater(len(fit.candidate_font_sizes), 1)

    def test_fit_allows_short_two_line_question_to_stay_readable(self) -> None:
        fit = measure_fitted_text(
            self.draw,
            "What's that?",
            (0, 0, 80, 46),
            font_path=None,
            base_font_size=28,
            source_box=(30, 4, 50, 43),
        )

        self.assertGreaterEqual(fit.font_size, 14)

    def test_fit_shrinks_before_splitting_overwide_words(self) -> None:
        fit = measure_fitted_text(
            self.draw,
            "I'm Director Kimerada.",
            (0, 0, 72, 164),
            font_path=None,
            base_font_size=28,
            source_box=(44, 77, 62, 234),
        )

        self.assertNotIn("or", fit.lines)
        self.assertTrue(all(len(line.strip(".,!?;:'\"")) > 2 for line in fit.lines))
        self.assertFalse(any("Kime" == line or "rada." == line for line in fit.lines))
        self.assertEqual(fit.split_word_count, 0)
        self.assertNotIn("split_word", fit.wrap_warnings)

    def test_fit_reports_split_word_when_unavoidable(self) -> None:
        fit = measure_fitted_text(
            self.draw,
            "supercalifragilistic",
            (0, 0, 34, 82),
            font_path=None,
            base_font_size=16,
            source_box=(0, 0, 34, 82),
        )

        self.assertGreater(fit.split_word_count, 0)
        self.assertIn("split_word", fit.wrap_warnings)

    def test_wrap_keeps_punctuation_attached(self) -> None:
        from manga_local_translator.render import load_font

        font = load_font(None, 16)
        lines = wrap_text(self.draw, "Wait! Don't go.", font, 68)

        self.assertFalse(any(line.strip() in {"!", ".", "?", ","} for line in lines))
        self.assertTrue(any("Wait!" in line for line in lines))

    def test_cramped_fit_reports_debug_warnings(self) -> None:
        fit = measure_fitted_text(
            self.draw,
            "This is far too much text for this tiny bubble.",
            (0, 0, 42, 34),
            font_path=None,
            base_font_size=28,
            source_box=(0, 0, 42, 34),
        )
        layout = RenderLayout(source_box=(0, 0, 42, 34), render_box=(0, 0, 42, 34))

        warnings = render_layout_warnings(layout, fit, image_width=100, image_height=100)

        self.assertTrue(fit.clipped or fit.font_size <= 8)
        self.assertTrue({"cramped_fit", "text_clipped", "tiny_font"} & set(warnings))

    def test_layout_warnings_include_geometry_qa_flags(self) -> None:
        layout = RenderLayout(source_box=(10, 10, 20, 20), render_box=(-1, 0, 120, 80), bubble_box=(5, 5, 60, 60))
        fit = TextFit(7, 5, "clipped", True, 90, 90, 40, 40, 12, 12, overflow_width=4, overflow_height=6)

        warnings = render_layout_warnings(layout, fit, image_width=100, image_height=100)

        self.assertIn("text_clipped", warnings)
        self.assertIn("tiny_font", warnings)
        self.assertIn("excessive_line_count", warnings)
        self.assertIn("text_outside_render_box", warnings)
        self.assertIn("render_box_outside_bubble", warnings)
        self.assertIn("render_box_outside_page", warnings)

    def test_layout_warnings_include_wrap_quality_flags(self) -> None:
        layout = RenderLayout(source_box=(10, 10, 90, 90), render_box=(10, 10, 90, 90))
        fit = TextFit(
            8,
            4,
            "fit",
            False,
            70,
            40,
            70,
            70,
            14,
            12,
            lines=("Go to", "the", "old", "shop"),
            wrap_warnings=("bad_wrap", "orphan_line", "split_word"),
            split_word_count=1,
            orphan_line_count=1,
        )

        warnings = render_layout_warnings(layout, fit, image_width=100, image_height=100)

        self.assertIn("bad_wrap", warnings)
        self.assertIn("orphan_line", warnings)
        self.assertIn("split_word", warnings)
        self.assertIn("cramped_fit", warnings)


if __name__ == "__main__":
    unittest.main()
