import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "source_extraction_autoresearch" / "scripts"
if not SCRIPTS_DIR.exists():
    raise unittest.SkipTest("source_extraction_autoresearch is a local ignored harness")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score import score_page  # noqa: E402


class SourceExtractionScoreTests(unittest.TestCase):
    def test_false_positive_details_include_classification_and_nearest_label(self):
        label = {
            "page_id": "page_001",
            "text_regions": [
                {
                    "region_id": "r001",
                    "box": [10, 10, 40, 80],
                    "source_text": "こんにちは",
                    "orientation": "vertical",
                    "kind": "dialogue",
                    "should_extract": True,
                    "group_id": "g001",
                    "page_order": 1,
                },
                {
                    "region_id": "r002",
                    "box": [60, 10, 100, 80],
                    "source_text": "広告",
                    "orientation": "vertical",
                    "kind": "metadata",
                    "should_extract": False,
                },
            ],
            "groups": [],
        }
        prediction = {
            "regions": [
                {
                    "pred_region_id": "p001",
                    "box": [10, 10, 40, 80],
                    "ocr_text": "こんにちは",
                    "normalized_ocr_text": "こんにちは",
                    "orientation": "vertical",
                    "group_id": "pg001",
                    "page_order": 1,
                },
                {
                    "pred_region_id": "p002",
                    "box": [60, 10, 100, 80],
                    "ocr_text": "広告",
                    "normalized_ocr_text": "広告",
                    "orientation": "vertical",
                    "group_id": "pg002",
                    "page_order": 2,
                },
            ],
            "groups": [],
            "timing": {"total_ms": 0.0},
        }

        metrics = score_page(label, prediction)

        self.assertEqual(metrics["destructive_false_positive_count"], 1)
        self.assertEqual(len(metrics["false_positive_details"]), 1)
        detail = metrics["false_positive_details"][0]
        self.assertEqual(detail["type"], "destructive_false_positive")
        self.assertEqual(detail["pred_region_id"], "p002")
        self.assertTrue(detail["overlaps_non_extractable"])
        self.assertEqual(detail["nearest_region_id"], "r002")
        self.assertEqual(detail["nearest_kind"], "metadata")
        self.assertFalse(detail["nearest_should_extract"])
        self.assertEqual(metrics["source_extraction_score"], metrics["source_extraction_quality_score"])

    def test_group_and_order_details_are_reported(self):
        label = {
            "page_id": "page_001",
            "text_regions": [
                {"region_id": "r001", "box": [10, 10, 40, 80], "source_text": "一", "orientation": "vertical", "kind": "dialogue", "should_extract": True, "group_id": "g001", "page_order": 1},
                {"region_id": "r002", "box": [50, 10, 80, 80], "source_text": "二", "orientation": "vertical", "kind": "dialogue", "should_extract": True, "group_id": "g002", "page_order": 2},
            ],
            "groups": [],
        }
        prediction = {
            "regions": [
                {"pred_region_id": "p001", "box": [10, 10, 40, 80], "ocr_text": "一", "normalized_ocr_text": "一", "orientation": "vertical", "group_id": "pg001", "page_order": 2},
                {"pred_region_id": "p002", "box": [50, 10, 80, 80], "ocr_text": "二", "normalized_ocr_text": "二", "orientation": "vertical", "group_id": "pg001", "page_order": 1},
            ],
            "groups": [{"pred_group_id": "pg001", "member_pred_region_ids": ["p001", "p002"]}],
            "timing": {"total_ms": 12.0},
        }

        metrics = score_page(label, prediction)

        self.assertEqual(metrics["overmerge_count"], 1)
        self.assertEqual(metrics["reading_order_error_count"], 1)
        self.assertEqual(metrics["group_failure_details"][0]["type"], "overmerge")
        self.assertEqual(metrics["reading_order_details"][0]["type"], "reading_order_inversion")
        self.assertEqual(metrics["page_score_breakdown"]["timing_tiebreaker"], 12.0)
        self.assertEqual(metrics["source_extraction_score"], metrics["source_extraction_quality_score"])


if __name__ == "__main__":
    unittest.main()
