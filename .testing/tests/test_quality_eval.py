from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from manga_local_translator.quality_eval import benchmark_variants, summarize_debug_reports, write_review_artifacts


class QualityEvalTests(unittest.TestCase):
    def test_benchmark_variants_can_pair_text_and_vision_runs(self) -> None:
        variants = benchmark_variants(["quality"], vision=False, compare_vision=True, vision_trigger="suspicious")

        self.assertEqual(
            variants,
            [
                ("quality", "quality:text", "quality-text", False),
                ("quality", "quality:vision-suspicious", "quality-vision-suspicious", True),
            ],
        )

    def test_summary_counts_vision_json_repair_and_reason_fields(self) -> None:
        report = {
            "kept_blocks": [
                {
                    "translated_text": "Mom",
                    "layout_warnings": ["tiny_font", "text_clipped"],
                    "vision_attempted": True,
                    "vision_accepted": False,
                    "vision_json_repair_attempted": True,
                    "vision_json_repair_used": True,
                    "vision_reject_reason": "invalid_visual_evidence",
                    "vision_facts_attempted": True,
                    "vision_facts_accepted": False,
                    "vision_facts_json_repair_attempted": True,
                    "vision_facts_json_repair_used": True,
                    "vision_facts_reject_reason": "contains_translation_field",
                    "qwen_critic_attempted": True,
                    "qwen_critic_flagged": True,
                    "qwen_critic_issues": ["awkward_literal"],
                    "qwen_critic_evidence_gate_reason": "no_local_repair_evidence",
                    "evidence_risk_flags": ["translation_evidence_failure"],
                    "evidence_repair_reasons": ["broken_english"],
                    "qwen_fallback_attempted": True,
                    "qwen_fallback_reject_reason": "fallback_evidence_failure:broken_english",
                },
                {
                    "translated_text": "Okay.",
                    "layout_warnings": ["tiny_font"],
                    "vision_attempted": True,
                    "vision_accepted": True,
                },
            ],
            "skipped_blocks": [],
            "fallback_blocks": [],
            "page_summary": {},
        }
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "1.ocr.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            rows = summarize_debug_reports(Path(temp), profile="quality")

        self.assertEqual(rows[0]["vision_json_repair_attempted"], 1)
        self.assertEqual(rows[0]["vision_json_repair_used"], 1)
        self.assertEqual(json.loads(rows[0]["vision_reject_reasons"]), {"invalid_visual_evidence": 1})
        self.assertEqual(rows[0]["vision_facts_json_repair_attempted"], 1)
        self.assertEqual(rows[0]["vision_facts_json_repair_used"], 1)
        self.assertEqual(json.loads(rows[0]["vision_facts_reject_reasons"]), {"contains_translation_field": 1})
        self.assertEqual(json.loads(rows[0]["layout_warning_types"]), {"text_clipped": 1, "tiny_font": 2})
        self.assertEqual(rows[0]["qwen_critic_attempted"], 1)
        self.assertEqual(rows[0]["qwen_critic_flagged"], 1)
        self.assertEqual(json.loads(rows[0]["qwen_critic_issue_types"]), {"awkward_literal": 1})
        self.assertEqual(rows[0]["evidence_risk_blocks"], 1)
        self.assertEqual(rows[0]["evidence_repair_reason_blocks"], 1)
        self.assertEqual(json.loads(rows[0]["evidence_risk_types"]), {"translation_evidence_failure": 1})
        self.assertEqual(json.loads(rows[0]["evidence_repair_reason_types"]), {"broken_english": 1})
        self.assertEqual(json.loads(rows[0]["qwen_critic_evidence_gate_reasons"]), {"no_local_repair_evidence": 1})
        self.assertEqual(json.loads(rows[0]["qwen_fallback_reject_reasons"]), {"fallback_evidence_failure:broken_english": 1})
        self.assertEqual(json.loads(rows[1]["vision_reject_reasons"]), {"invalid_visual_evidence": 1})
        self.assertEqual(json.loads(rows[1]["vision_facts_reject_reasons"]), {"contains_translation_field": 1})
        self.assertEqual(json.loads(rows[1]["layout_warning_types"]), {"text_clipped": 1, "tiny_font": 2})
        self.assertEqual(json.loads(rows[1]["qwen_critic_issue_types"]), {"awkward_literal": 1})
        self.assertEqual(json.loads(rows[1]["evidence_repair_reason_types"]), {"broken_english": 1})

    def test_summary_recurses_into_chapter_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_dir = root / "chapter-a"
            report_dir.mkdir()
            report_path = report_dir / "001.ocr.json"
            report_path.write_text(
                json.dumps(
                    {
                        "kept_blocks": [{"translated_text": "Hello."}],
                        "skipped_blocks": [],
                        "fallback_blocks": [],
                    }
                ),
                encoding="utf-8",
            )

            rows = summarize_debug_reports(root, profile="quality")

        self.assertEqual(rows[0]["page"], "chapter-a/001")
        self.assertEqual(rows[0]["kept_blocks"], 1)
        self.assertEqual(rows[1]["kept_blocks"], 1)

    def test_write_review_artifacts_creates_markdown_with_html_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / "001.ocr.json"
            report_path.write_text(
                json.dumps(
                    {
                        "kept_blocks": [
                            {
                                "page_order": 1,
                                "source_text": "\u6bcd",
                                "translated_text": "...",
                                "cat_rejected": True,
                                "cat_reject_reason": "cat_chatter",
                            }
                        ],
                        "skipped_blocks": [],
                        "fallback_blocks": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            write_review_artifacts(root)

            self.assertTrue((root / "translation_review.html").exists())
            self.assertTrue((root / "translation_review.csv").exists())
            markdown = (root / "translation_review.md").read_text(encoding="utf-8")

        self.assertIn("Manga Translation Human Review", markdown)
        self.assertIn("cat_chatter", markdown)


if __name__ == "__main__":
    unittest.main()
