from __future__ import annotations

import unittest
from pathlib import Path

from manga_local_translator.detect_ocr import (
    ctd_line_union_crop_box,
    is_repeated_kana_sfx_ctd_block,
    manga_ocr_crop_box,
    normalize_ocr_text,
)
from manga_local_translator.detect_types import TextBlock


def block(
    text: str = "",
    box: tuple[int, int, int, int] = (100, 100, 150, 260),
    *,
    vertical: bool = True,
    detector: str = "ctd",
    lines: list[list[list[int]]] | None = None,
    language: str = "ja",
) -> TextBlock:
    metadata: dict[str, object] = {"vertical": vertical, "language": language}
    if lines is not None:
        metadata["lines"] = lines
    return TextBlock(text=text, box=box, confidence=1.0, detector=detector, metadata=metadata)


class DetectOcrHeuristicTests(unittest.TestCase):
    def test_no_exact_benchmark_rewrite_helpers_return(self) -> None:
        source = Path("manga_local_translator/detect_ocr.py").read_text(encoding="utf-8")

        forbidden_fragments = [
            "fix_common_phrase_ocr_text",
            "is_bottom_left_short_horizontal_ctd_block",
            "is_bottom_edge_short_horizontal_ctd_block",
            "is_small_horizontal_credit_ctd_block",
            "is_left_edge_vertical_metadata_ctd_block",
            "is_compact_horizontal_sfx_ctd_block",
            "友達しゃなく",
            "この文化祭",
            "休憩時間",
            "二つの告白",
        ]
        for fragment in forbidden_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, source)

    def test_normalize_ocr_text_only_removes_spacing(self) -> None:
        self.assertEqual(normalize_ocr_text("\u53cb \u9054\u3057\u3083\u306a\u304f"), "\u53cb\u9054\u3057\u3083\u306a\u304f")
        self.assertEqual(normalize_ocr_text("\u3053\u306e\u6587\u5316\u796d..."), "\u3053\u306e\u6587\u5316\u796d...")

    def test_line_union_crop_uses_ctd_line_bounds_for_vertical_multi_line_blocks(self) -> None:
        candidate = block(
            box=(100, 20, 185, 330),
            lines=[
                [[112, 145], [132, 145], [132, 300], [112, 300]],
                [[150, 140], [170, 140], [170, 305], [150, 305]],
            ],
        )

        self.assertEqual(
            ctd_line_union_crop_box(candidate, image_width=300, image_height=400, padding=10),
            (102, 130, 180, 315),
        )

    def test_line_union_crop_rejects_non_ctd_or_non_vertical_blocks(self) -> None:
        candidate = block(detector="visual", lines=[[[0, 0], [10, 0], [10, 20], [0, 20]]])
        self.assertIsNone(ctd_line_union_crop_box(candidate, image_width=300, image_height=400, padding=10))

        horizontal = block(vertical=False, lines=[[[0, 0], [10, 0], [10, 20], [0, 20]]])
        self.assertIsNone(ctd_line_union_crop_box(horizontal, image_width=300, image_height=400, padding=10))

    def test_extra_bottom_crop_expands_uncertain_vertical_text_without_hardcoded_source_text(self) -> None:
        candidate = block(
            box=(230, 750, 320, 931),
            lines=[[[233, 750], [320, 751], [317, 931], [230, 930]]],
            language="unknown",
        )

        self.assertEqual(manga_ocr_crop_box(candidate, 850, 1200, 10), (220, 740, 330, 1051))

    def test_compact_repeated_kana_sfx_filter_is_shape_and_script_based(self) -> None:
        self.assertTrue(
            is_repeated_kana_sfx_ctd_block(
                block("\u3042\u3042\u3042", box=(20, 20, 40, 75)),
                text="\u3042\u3042\u3042",
            )
        )
        self.assertFalse(
            is_repeated_kana_sfx_ctd_block(
                block("\u3046\u3093", box=(20, 20, 40, 75)),
                text="\u3046\u3093",
            )
        )
        self.assertFalse(
            is_repeated_kana_sfx_ctd_block(
                block("\u3042\u3042\u3042", box=(20, 20, 80, 160), vertical=False),
                text="\u3042\u3042\u3042",
            )
        )


if __name__ == "__main__":
    unittest.main()
