from __future__ import annotations

import unittest

from manga_local_translator.debug_report import add_render_collision_warnings, render_fit_to_debug_dict
from manga_local_translator.render import TextFit


class DebugReportTests(unittest.TestCase):
    def test_add_render_collision_warnings_records_overlapping_render_boxes(self) -> None:
        blocks = [
            {"page_order": 1, "source_text": "a", "render_box": (0, 0, 50, 50)},
            {"page_order": 2, "source_text": "b", "render_box": (25, 25, 75, 75)},
            {"page_order": 3, "source_text": "c", "render_box": (100, 100, 120, 120)},
        ]

        add_render_collision_warnings(blocks)

        self.assertIn("render_box_collision", blocks[0]["layout_warnings"])
        self.assertEqual(blocks[0]["render_collisions"][0]["page_order"], 2)
        self.assertEqual(blocks[0]["render_collisions"][0]["overlap_area"], 625)
        self.assertNotIn("render_collisions", blocks[2])

    def test_render_fit_debug_includes_wrap_diagnostics(self) -> None:
        fit = TextFit(
            10,
            2,
            "fit",
            False,
            50,
            24,
            70,
            70,
            14,
            12,
            lines=("Hello", "there."),
            wrap_score=12.5,
            wrap_warnings=("bad_wrap",),
            split_word_count=1,
            orphan_line_count=0,
            candidate_font_sizes=(14, 13, 12, 11, 10),
            compacted_for_render=True,
        )

        payload = render_fit_to_debug_dict(fit)

        self.assertEqual(payload["wrap_score"], 12.5)
        self.assertEqual(payload["wrap_warnings"], ["bad_wrap"])
        self.assertEqual(payload["split_word_count"], 1)
        self.assertEqual(payload["candidate_font_sizes"], [14, 13, 12, 11, 10])
        self.assertTrue(payload["compacted_for_render"])


if __name__ == "__main__":
    unittest.main()
