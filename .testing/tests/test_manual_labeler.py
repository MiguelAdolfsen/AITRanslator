import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labeler.manual_labeler import (  # noqa: E402
    duplicate_page_stems,
    normalize_boxes_for_image,
    parse_image_size,
    rescale_box,
)


class ManualLabelerTests(unittest.TestCase):
    def test_parse_image_size_rejects_invalid_values(self):
        self.assertEqual(parse_image_size([477, 800]), (477, 800))
        self.assertIsNone(parse_image_size(["bad", 800]))
        self.assertIsNone(parse_image_size([0, 800]))
        self.assertIsNone(parse_image_size([477]))

    def test_rescale_box_maps_saved_coordinates_to_current_image_size(self):
        self.assertEqual(rescale_box([100, 100, 200, 300], (500, 1000), (1000, 500)), [200, 50, 400, 150])

    def test_normalize_boxes_for_image_rescales_all_regions(self):
        label = {
            "text_regions": [
                {"region_id": "r001_001", "box": [100, 100, 200, 300]},
                {"region_id": "r001_002", "box": [450, 900, 550, 1100]},
            ]
        }

        changed = normalize_boxes_for_image(label, (500, 1000), (1000, 500))

        self.assertTrue(changed)
        self.assertEqual(label["text_regions"][0]["box"], [200, 50, 400, 150])
        self.assertEqual(label["text_regions"][1]["box"], [900, 450, 1000, 500])

    def test_duplicate_page_stems_detects_label_file_collisions(self):
        duplicates = duplicate_page_stems(
            [
                Path("pages/page_001.png"),
                Path("pages/page_001.jpg"),
                Path("pages/page_002.png"),
            ]
        )

        self.assertEqual(duplicates, {"page_001": ["page_001.png", "page_001.jpg"]})


if __name__ == "__main__":
    unittest.main()
