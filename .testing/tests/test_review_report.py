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
                    "source_text": "ちちもははもきらいー！！",
                    "translated_text": "I don't like my mother!",
                    "qwen_baseline": "I hate my father and mother!",
                    "qwen_candidate": "I don't like my mother!",
                    "qwen_verify_reason": "target",
                },
                {
                    "page_order": 2,
                    "source_text": "はんぶんこしたぴーなつをおたがいたべる",
                    "translated_text": "If you do that, you can watch my organization.",
                    "layout_warnings": ["tiny_font"],
                },
                {
                    "page_order": 3,
                    "source_text": "ふたりとも〈アーニャ〉のりっぱなこぶんだ",
                    "translated_text": "You're both a good son of a bitch.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "7.ocr.json"
            path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

            rows = rows_from_report(path)

        self.assertIn("dropped_pair:father_mother", rows[0].issues)
        self.assertIn("known_bad_pattern:watch_my_organization", rows[1].issues)
        self.assertIn("layout:tiny_font", rows[1].issues)
        self.assertIn("known_bad_pattern:kobun_offensive", rows[2].issues)

    def test_writes_html_and_csv(self) -> None:
        report = {
            "kept_blocks": [
                {
                    "page_order": 1,
                    "source_text": "アーニャ",
                    "translated_text": "Anja",
                    "qwen_baseline": "Anya",
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
        self.assertIn("name_consistency:anya", html_text)
        self.assertIn("source_text", csv_text)
        self.assertIn("Anja", csv_text)


if __name__ == "__main__":
    unittest.main()
