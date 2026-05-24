from __future__ import annotations

import unittest

from manga_local_translator.detect_types import TextBlock
from manga_local_translator.grouping import (
    build_page_order_report,
    context_for_block,
    group_text_blocks_for_translation,
    sort_blocks_in_reading_order,
)


def block(
    text: str,
    box: tuple[int, int, int, int],
    *,
    vertical: bool = True,
    detector: str = "ctd",
) -> TextBlock:
    return TextBlock(text=text, box=box, confidence=90.0, detector=detector, metadata={"vertical": vertical})


class GroupingTests(unittest.TestCase):
    def test_vertical_group_merges_right_to_left_members(self) -> None:
        right = block("右", (100, 10, 120, 80))
        left = block("左", (82, 12, 102, 82))

        grouped, report = group_text_blocks_for_translation([left, right])

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].text, "右左")
        self.assertEqual(grouped[0].metadata["translation_group_size"], 2)
        self.assertEqual([member["source_text"] for member in report[0]["members"]], ["右", "左"])

    def test_separate_speech_bubbles_are_not_grouped_when_far_apart(self) -> None:
        first = block("アーニャ", (300, 10, 330, 80))
        second = block("父", (20, 200, 50, 270))

        grouped, _report = group_text_blocks_for_translation([first, second])

        self.assertEqual([item.text for item in grouped], ["アーニャ", "父"])

    def test_page_order_uses_manga_right_to_left_rows_and_context_windows(self) -> None:
        top_right = block("一", (300, 10, 330, 70))
        top_left = block("二", (100, 12, 130, 72))
        bottom_right = block("三", (280, 160, 310, 220))

        report = build_page_order_report([top_left, bottom_right, top_right], width=400, height=300)

        self.assertEqual([item["source_text"] for item in report], ["一", "二", "三"])
        context = context_for_block(top_left, report)
        self.assertEqual(context["context_before"], "一")
        self.assertEqual(context["context_after"], "三")
        self.assertEqual(context["context_before_window"], ["一"])

    def test_sort_vertical_group_members_right_to_left_then_top_down(self) -> None:
        ordered = sort_blocks_in_reading_order(
            [
                block("left", (80, 50, 100, 90)),
                block("right-lower", (110, 60, 130, 100)),
                block("right-upper", (110, 10, 130, 40)),
            ]
        )

        self.assertEqual([item.text for item in ordered], ["right-upper", "right-lower", "left"])


if __name__ == "__main__":
    unittest.main()
