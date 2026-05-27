from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from translation_quality_autoresearch.scripts.build_cases_from_quality_runs import infer_source_type, mine_cases


class BuildCasesFromQualityRunsTests(unittest.TestCase):
    def test_mines_deduped_contextual_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {
                "source_image": "page001.png",
                "output_image": "out/page001.png",
                "detector": "ctd",
                "ocr_engine": "manga-ocr",
                "translator": "cat",
                "qwen_mode": "block",
                "kept_blocks": [
                    {
                        "source_text": "早く行こう",
                        "translated_text": "Let's hurry.",
                        "box": [1, 2, 3, 20],
                        "page_order": 1,
                        "context_after": "待って",
                        "line_id": "001_001",
                    },
                    {
                        "source_text": "早く行こう",
                        "translated_text": "Let's hurry.",
                        "box": [1, 2, 3, 20],
                        "page_order": 2,
                    },
                ],
            }
            (root / "sample.ocr.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            rows = mine_cases(root, min_source_chars=1, max_cases=50, include_context=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case_id"], "generated_quality_00001")
            self.assertEqual(rows[0]["context_after"], ["待って"])
            self.assertTrue(rows[0]["metadata"]["needs_human_review"])
            self.assertEqual(rows[0]["metadata"]["draft_machine_translation"], "Let's hurry.")

    def test_infers_reaction_without_calling_short_hiragana_sfx(self) -> None:
        self.assertEqual(infer_source_type("はーい"), "reaction")
        self.assertEqual(infer_source_type("ビィ！"), "sfx")


if __name__ == "__main__":
    unittest.main()
