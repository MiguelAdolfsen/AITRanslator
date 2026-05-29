from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from manga_local_translator.config import PipelineConfig
from manga_local_translator.debug_report import write_debug_report
from manga_local_translator.detect_types import TextBlock
from manga_local_translator.line_identity import assign_render_line_ids, enrich_page_order_report, lookup_translation, set_translation
from manga_local_translator.render import RenderLayout, TextFit
from manga_local_translator.review_report import rows_from_report
from manga_local_translator.translation_review import apply_translation_evidence, english_context_for_block


class LineIdentityTests(unittest.TestCase):
    def test_duplicate_source_text_uses_distinct_line_ids(self) -> None:
        blocks = [
            TextBlock("\u6bcd", (10, 10, 30, 30), 90, "ctd"),
            TextBlock("\u6bcd", (60, 10, 80, 30), 90, "ctd"),
        ]
        page_order = [
            {"page_order": 1, "box": blocks[0].box, "source_text": blocks[0].text, "detector": "ctd"},
            {"page_order": 2, "box": blocks[1].box, "source_text": blocks[1].text, "detector": "ctd"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            output_path = Path(temp) / "page.png"
            blocks = assign_render_line_ids(blocks, page_order, output_path)
            page_order = enrich_page_order_report(page_order, blocks)
            translations: dict[str, str] = {}
            contexts: dict[str, dict[str, object]] = {}
            set_translation(translations, blocks[0], "Mother.")
            set_translation(translations, blocks[1], "Mom.")

            apply_translation_evidence(blocks, translations, contexts, None)
            before, after = english_context_for_block(blocks[1], translations, page_order)

        self.assertEqual(lookup_translation(translations, blocks[0]), "Mother.")
        self.assertEqual(lookup_translation(translations, blocks[1]), "Mom.")
        self.assertEqual(before, ("Mother.",))
        self.assertEqual(after, ())
        self.assertEqual(set(contexts), {"page_001", "page_002"})
        self.assertEqual(contexts["page_001"]["source_hash"], contexts["page_002"]["source_hash"])

    def test_debug_and_review_reports_expose_line_id_for_duplicate_text(self) -> None:
        blocks = [
            TextBlock("\u6bcd", (10, 10, 30, 30), 90, "ctd"),
            TextBlock("\u6bcd", (60, 10, 80, 30), 90, "ctd"),
        ]
        page_order = [
            {"page_order": 1, "box": blocks[0].box, "source_text": blocks[0].text, "detector": "ctd"},
            {"page_order": 2, "box": blocks[1].box, "source_text": blocks[1].text, "detector": "ctd"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            output_path = temp_path / "page.png"
            blocks = assign_render_line_ids(blocks, page_order, output_path)
            page_order = enrich_page_order_report(page_order, blocks)
            translations = {"page_001": "Mother.", "page_002": "Mom."}
            contexts = {
                "page_001": {"qwen_used": True},
                "page_002": {"qwen_used": True},
            }
            write_debug_report(
                output_path,
                output_path,
                PipelineConfig(debug=True),
                blocks,
                blocks,
                translations,
                [],
                [],
                [],
                [
                    RenderLayout(blocks[0].box, blocks[0].box),
                    RenderLayout(blocks[1].box, blocks[1].box),
                ],
                [
                    TextFit(12, 1, "fit", False, 20, 14, 20, 14, 12, 14),
                    TextFit(12, 1, "fit", False, 20, 14, 20, 14, 12, 14),
                ],
                100,
                100,
                2,
                page_order,
                contexts,
            )
            report_path = temp_path / "page.ocr.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows = rows_from_report(report_path)

        kept = report["kept_blocks"]
        self.assertEqual([block["line_id"] for block in kept], ["page_001", "page_002"])
        self.assertEqual([block["translated_text"] for block in kept], ["Mother.", "Mom."])
        self.assertEqual([row.line_id for row in rows], ["page_001", "page_002"])


if __name__ == "__main__":
    unittest.main()
