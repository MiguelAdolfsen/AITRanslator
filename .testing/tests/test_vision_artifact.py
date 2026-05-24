from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from manga_local_translator.detect_types import TextBlock
from manga_local_translator.vision_artifact import create_vision_artifact, label_plan


class VisionArtifactTests(unittest.TestCase):
    def test_numbered_artifact_preserves_dimensions_and_maps_blocks(self) -> None:
        image = np.full((120, 90, 3), 255, dtype=np.uint8)
        blocks = [
            TextBlock("一", (10, 10, 30, 40), 0.9, "ctd", {}),
            TextBlock("二", (50, 60, 80, 100), 0.8, "ctd", {}),
        ]
        page_order = [
            {"page_order": 1, "box": blocks[0].box, "source_text": blocks[0].text, "detector": "ctd"},
            {"page_order": 2, "box": blocks[1].box, "source_text": blocks[1].text, "detector": "ctd"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "page.png"

            artifact = create_vision_artifact(image, blocks, {blocks[0].text: "One", blocks[1].text: "Two"}, page_order, output, mode="numbered_page")

            self.assertTrue(artifact.image_path.exists())
            rendered = cv2.imread(str(artifact.image_path), cv2.IMREAD_COLOR)
            self.assertEqual(rendered.shape, image.shape)
            self.assertEqual([entry.number for entry in artifact.number_map], [1, 2])
            self.assertEqual(artifact.number_map[0].translated_text, "One")

    def test_tiny_region_label_moves_outside_with_leader_line(self) -> None:
        warnings: list[str] = []

        label_box, leader = label_plan(1, (10, 10, 24, 24), 120, 120, [], warnings, line_id="p1_001")

        self.assertIsNotNone(leader)
        self.assertEqual(label_box, (29, 2, 59, 32))
        self.assertIn("tiny_label_region:p1_001", warnings)
        self.assertTrue(all(0 <= value <= 119 for value in label_box))

    def test_clamped_label_emits_warning(self) -> None:
        warnings: list[str] = []

        label_box, leader = label_plan(
            1,
            (110, 4, 118, 12),
            120,
            120,
            [],
            warnings,
            line_id="p1_002",
        )

        self.assertIsNotNone(leader)
        self.assertTrue(all(0 <= value <= 119 for value in label_box))
        self.assertIn("label_clamped:p1_002", warnings)

    def test_overlapping_label_emits_warning(self) -> None:
        warnings: list[str] = []

        label_box, leader = label_plan(
            1,
            (50, 50, 60, 60),
            120,
            120,
            [(65, 40, 95, 70), (15, 40, 45, 70), (40, 15, 70, 45), (40, 65, 70, 95), (40, 40, 70, 70)],
            warnings,
            line_id="p1_003",
        )

        self.assertIsNotNone(leader)
        self.assertTrue(all(0 <= value <= 119 for value in label_box))
        self.assertIn("overlapping_label:p1_003", warnings)


if __name__ == "__main__":
    unittest.main()
