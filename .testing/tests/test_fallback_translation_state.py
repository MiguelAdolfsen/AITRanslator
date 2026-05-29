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
from manga_local_translator.translated_state import TranslationReviewState, update_translated_line_state


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


class LineIdAwareTranslator:
    def translate_with_context(self, _text: str, **kwargs) -> str:
        return {
            "page_001": "Mother.",
            "page_002": "Mom.",
        }[str(kwargs["debug_id"])]


class FallbackTranslationStateTests(unittest.TestCase):
    def test_primary_translation_stores_duplicate_source_lines_independently(self) -> None:
        first = TextBlock("母", (10, 10, 30, 30), 91.0, "ctd", metadata={"line_id": "page_001"})
        second = TextBlock("母", (60, 10, 80, 30), 91.0, "ctd", metadata={"line_id": "page_002"})
        page = PreparedPage(
            image_path=Path("page.png"),
            output_path=Path("out.png"),
            image_bgr=None,
            width=100,
            height=100,
            raw_blocks=[first, second],
            render_blocks=[first, second],
            skipped_blocks=[],
            grouping_report=[],
            page_order_report=[
                {
                    "source_text": first.text,
                    "box": first.box,
                    "page_order": 1,
                    "line_id": "page_001",
                    "context_after": second.text,
                },
                {
                    "source_text": second.text,
                    "box": second.box,
                    "page_order": 2,
                    "line_id": "page_002",
                    "context_before": first.text,
                },
            ],
        )
        page.translation_contexts[first.text] = {"legacy_context": "shared bubble note"}

        translate_prepared_page(page, LineIdAwareTranslator(), PipelineConfig(translator="qwen"))

        state = TranslationReviewState.for_page(page)
        self.assertEqual(state.translation_for(first), "Mother.")
        self.assertEqual(state.translation_for(second), "Mom.")
        first_context = state.context_for(first)
        second_context = state.context_for(second)
        self.assertEqual(first_context["line_id"], "page_001")
        self.assertEqual(second_context["line_id"], "page_002")
        self.assertEqual(first_context["legacy_context"], "shared bubble note")
        self.assertEqual(second_context["legacy_context"], "shared bubble note")

    def test_translation_review_state_reads_updates_and_exports_line_identity_state(self) -> None:
        first = TextBlock("source", (1, 2, 30, 40), 91.0, "ctd", metadata={"line_id": "line-001"})
        second = TextBlock("source", (40, 2, 70, 40), 91.0, "ctd", metadata={"line_id": "line-002"})
        page = PreparedPage(
            image_path=Path("page.png"),
            output_path=Path("out.png"),
            image_bgr=None,
            width=100,
            height=100,
            raw_blocks=[first, second],
            render_blocks=[first, second],
            skipped_blocks=[],
            grouping_report=[],
            page_order_report=[
                {"source_text": first.text, "box": first.box, "page_order": 1},
                {"source_text": second.text, "box": second.box, "page_order": 2},
            ],
        )
        page.translations = {"source": "legacy translation", "line-001": "first translation"}
        page.translation_contexts = {
            "source": {"translator": "legacy", "tone": "formal"},
            "line-001": {"translator": "qwen", "notes": "first line"},
        }

        state = TranslationReviewState.for_page(page)

        self.assertEqual(state.state_key_for(first), "line-001")
        self.assertEqual(state.state_key_for(second), "line-002")
        self.assertEqual(state.translation_for(first, default=""), "first translation")
        self.assertEqual(state.translation_for(second, default=""), "legacy translation")
        self.assertEqual(state.context_for(first)["translator"], "qwen")
        self.assertEqual(state.context_for(second)["translator"], "legacy")

        context = state.update_line(
            second,
            translated_text="second translation",
            context_updates={"reviewed": True},
        )

        self.assertEqual(state.translation_for(first, default=""), "first translation")
        self.assertEqual(state.translation_for(second, default=""), "second translation")
        self.assertEqual(page.translations["source"], "legacy translation")
        self.assertEqual(page.translation_contexts["source"], {"translator": "legacy", "tone": "formal"})
        self.assertEqual(context["translator"], "legacy")
        self.assertTrue(context["reviewed"])
        self.assertEqual(context["line_id"], "line-002")
        self.assertEqual(state.source_compat_translations(page.render_blocks), {"source": "second translation"})

    def test_translation_review_state_applies_fallback_as_one_operation(self) -> None:
        block = TextBlock("æ¯", (1, 2, 30, 40), 91.0, "ctd", metadata={"line_id": "line-001"})
        page = _page_with_block(block)
        page.translations[block.text] = "legacy"
        page.translation_contexts[block.text] = {"translator": "legacy"}

        translated = TranslationReviewState.for_page(page).apply_fallback(
            block,
            reason="empty_translation",
            source_translation="",
        )

        self.assertEqual(translated, "...")
        self.assertEqual(lookup_translation(page.translations, block), "...")
        self.assertEqual(page.translations[block.text], "legacy")
        context = lookup_context(page.translation_contexts, block)
        self.assertEqual(context["translator"], "legacy")
        self.assertEqual(context["fallback_used"], True)
        self.assertEqual(context["fallback_reason"], "empty_translation")
        self.assertEqual(context["fallback_source_translation"], "")
        self.assertEqual(context["fallback_translation"], "...")
        self.assertEqual(page.translation_fallback_blocks[0]["line_id"], "line-001")
        self.assertEqual(page.translation_fallback_blocks[0]["translated_text"], "...")

    def test_translated_line_state_update_uses_line_identity_for_translation_and_context(self) -> None:
        block = TextBlock("source", (1, 2, 30, 40), 91.0, "ctd", metadata={"line_id": "line-001"})
        translations = {block.text: "legacy"}
        contexts = {block.text: {"translator": "legacy"}}

        context = update_translated_line_state(
            translations,
            contexts,
            block,
            translated_text="Mom",
            context_updates={"vision_accepted": True},
        )

        self.assertEqual(lookup_translation(translations, block), "Mom")
        self.assertEqual(translations[block.text], "legacy")
        self.assertEqual(lookup_context(contexts, block)["translator"], "legacy")
        self.assertTrue(lookup_context(contexts, block)["vision_accepted"])
        self.assertEqual(context["line_id"], "line-001")

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
