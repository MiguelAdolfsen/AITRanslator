from __future__ import annotations

import unittest

import cv2
import numpy as np

from manga_local_translator.detect_types import TextBlock
from manga_local_translator.erase import erase_text, uses_dark_background


class EraseTests(unittest.TestCase):
    def test_white_erase_removes_text_ink_without_wiping_bubble_outline(self) -> None:
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (140, 100), (0, 0, 0), thickness=2)
        cv2.putText(image, "HI", (62, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
        block = TextBlock("HI", (58, 38, 104, 76), 90.0, "test")

        cleaned = erase_text(image, [block], mode="white", padding=10)

        self.assertLess(cleaned[20, 80].mean(), 30, "top outline above text should remain black")
        self.assertLess(cleaned[20, 30].mean(), 30, "unrelated bubble outline should remain black")
        self.assertGreater(cleaned[60, 76].mean(), 245, "text ink should be erased to white")

    def test_white_erase_does_not_touch_unrelated_art_inside_padded_box(self) -> None:
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        cv2.line(image, (10, 10), (110, 10), (0, 0, 0), thickness=2)
        cv2.putText(image, "I", (55, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        block = TextBlock("I", (50, 25, 70, 48), 90.0, "test")

        cleaned = erase_text(image, [block], mode="white", padding=25)

        self.assertLess(cleaned[10, 60].mean(), 30)
        self.assertGreater(cleaned[35, 60].mean(), 245)

    def test_dark_background_text_uses_rectangular_white_erase(self) -> None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        cv2.putText(image, "HI", (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        block = TextBlock("HI", (35, 20, 82, 52), 90.0, "test")

        self.assertTrue(uses_dark_background(image, block.box))
        cleaned = erase_text(image, [block], mode="white", padding=4)

        self.assertGreater(cleaned[35, 55].mean(), 245)


if __name__ == "__main__":
    unittest.main()
