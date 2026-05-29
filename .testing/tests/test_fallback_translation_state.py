from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from manga_local_translator.config import PipelineConfig
from manga_local_translator.detect_types import TextBlock
from manga_local_translator.line_identity import lookup_context, lookup_translation, source_hash
from manga_local_translator.page_cache import load_translation_cache
from manga_local_translator.page_types import PreparedPage
from manga_local_translator.pipeline import refresh_translation_fallback_blocks, translate_prepared_page


def _page_with_block(block: TextBlock) -> PreparedPage:
    return PreparedPage(
        image_path=Path("page.png"),
        output_path=Path("out.png"),
        image_bgr=None,
        width=100,
        height=100,
        raw_blocks=[block],
        render_blocks=[block],
        skipped_blocks=[],
        grouping_report=[],
        page_order_report=[{"source_text": block.text, "box": block.box, "page_order": 1}],
    )


class EmptyTranslator:
    def translate_with_context(self, _text: str, **_kwargs) -> str:
        return ""


class FallbackTranslationStateTests(unittest.TestCase):
    def test_direct_translation_fallback_updates_translation_report_and_context_together(self) -> None:
        block = TextBlock("母", (1, 2, 30, 40), 91.0, "ctd", metadata={"line_id": "line-001"})
        page = _page_with_block(block)

        translate_prepared_page(page, EmptyTranslator(), PipelineConfig(translator="qwen"))

        self.assertEqual(lookup_translation(page.translations, block), "...")
        self.assertEqual(
            page.translation_fallback_blocks,
            [
                {
                    "detector": "ctd",
                    "box": (1, 2, 30, 40),
                    "confidence": 91.0,
                    "block_id": "",
                    "line_id": "line-001",
                    "source_hash": source_hash("母"),
                    "source_text": "母",
                    "status": "fallback",
                    "metadata": {"line_id": "line-001"},
                    "translated_text": "...",
                    "reason": "empty_translation",
                }
            ],
        )
        self.assertEqual(lookup_context(page.translation_contexts, block)["fallback_used"], True)
        self.assertEqual(lookup_context(page.translation_contexts, block)["fallback_reason"], "empty_translation")
        self.assertEqual(lookup_context(page.translation_contexts, block)["fallback_translation"], "...")

    def test_cache_load_rehydrates_fallback_translation_state_for_line_identity(self) -> None:
        block = TextBlock("母", (1, 2, 30, 40), 91.0, "ctd", metadata={"line_id": "line-001"})
        page = _page_with_block(block)

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "page.translation.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "version": 3,
                        "cache_kind": "translation",
                        "translations": {"母": ""},
                        "translation_contexts": {"母": {"translator": "qwen"}},
                        "translation_fallback_blocks": [
                            {
                                "box": [1, 2, 30, 40],
                                "source_text": "母",
                                "status": "fallback",
                                "translated_text": "...",
                                "reason": "empty_translation",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            load_translation_cache(page, cache_path)

        context = lookup_context(page.translation_contexts, block)
        self.assertEqual(lookup_translation(page.translations, block), "...")
        self.assertEqual(context["translator"], "qwen")
        self.assertEqual(context["fallback_used"], True)
        self.assertEqual(context["fallback_reason"], "empty_translation")
        self.assertEqual(context["fallback_translation"], "...")

    def test_final_refresh_uses_same_fallback_state_for_unusable_translations(self) -> None:
        block = TextBlock("母", (1, 2, 30, 40), 91.0, "ctd", metadata={"line_id": "line-001"})
        page = _page_with_block(block)
        page.translations["line-001"] = "母"
        page.translation_contexts["line-001"] = {"translator": "qwen"}

        refresh_translation_fallback_blocks(page, PipelineConfig(translator="qwen"))

        context = lookup_context(page.translation_contexts, block)
        self.assertEqual(lookup_translation(page.translations, block), "...")
        self.assertEqual(page.translation_fallback_blocks[0]["line_id"], "line-001")
        self.assertEqual(page.translation_fallback_blocks[0]["translated_text"], "...")
        self.assertEqual(page.translation_fallback_blocks[0]["reason"], "non_english_translation")
        self.assertEqual(context["translator"], "qwen")
        self.assertEqual(context["fallback_used"], True)
        self.assertEqual(context["fallback_reason"], "non_english_translation")
        self.assertEqual(context["fallback_source_translation"], "母")
        self.assertEqual(context["fallback_translation"], "...")


if __name__ == "__main__":
    unittest.main()
