from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from manga_local_translator.review_report import rows_from_report, write_csv_report, write_html_report


class ReviewReportTests(unittest.TestCase):
    def test_rows_detect_known_translation_issues(self) -> None:
        report = {
            "kept_blocks": [
                {
                    "page_order": 1,
                    "source_text": "\u3072\u307f\u3064\u305d\u3057\u304d\u3008PII2\u3009\u306e\u30dc\u30b9",
                    "translated_text": "I'm the boss of the secret organization.",
                    "qwen_baseline": "I'm the boss of PII2.",
                    "qwen_candidate": "I'm the boss of the secret organization.",
                    "qwen_verify_reason": "target",
                },
                {
                    "page_order": 2,
                    "source_text": "\u3084\u3081\u308d",
                    "translated_text": "Stop.",
                    "layout_warnings": ["tiny_font"],
                },
                {
                    "page_order": 3,
                    "source_text": "\u3042",
                    "translated_text": "one-two-three-four-five-six",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "7.ocr.json"
            path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

            rows = rows_from_report(path)

        self.assertIn("possibly_dropped_bracket_term", rows[0].issues)
        self.assertIn("layout:tiny_font", rows[1].issues)
        self.assertIn("malformed_hyphen_chain", rows[2].issues)

    def test_writes_html_and_csv(self) -> None:
        report = {
            "kept_blocks": [
                {
                    "page_order": 1,
                    "source_text": "\u30c6\u30b9\u30c8",
                    "translated_text": "Test",
                    "qwen_baseline": "Test",
                    "context_before": "前",
                    "context_after": "後",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            report_path = temp_path / "1.ocr.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            rows = rows_from_report(report_path)
            html_path = temp_path / "review.html"
            csv_path = temp_path / "review.csv"

            write_html_report(rows, html_path)
            write_csv_report(rows, csv_path)

            html_text = html_path.read_text(encoding="utf-8")
            csv_text = csv_path.read_text(encoding="utf-8")

        self.assertIn("Manga Translation Review", html_text)
        self.assertIn("Test", html_text)
        self.assertIn("source_text", csv_text)
        self.assertIn("Test", csv_text)


if __name__ == "__main__":
    unittest.main()
