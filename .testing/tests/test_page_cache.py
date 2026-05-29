from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from manga_local_translator.detect_types import TextBlock
from manga_local_translator.page_cache import (
    load_prepared_page_cache,
    load_translation_cache,
    save_prepared_page_cache,
    save_translation_cache,
)
from manga_local_translator.page_types import PreparedPage
from manga_local_translator.translated_state import TranslationReviewState

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


@unittest.skipIf(cv2 is None, "opencv-python is not installed")
class PageCacheTests(unittest.TestCase):
    def test_load_prepared_page_cache_can_override_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "page.png"
            original_output = root / "old.png"
            new_output = root / "new.png"
            cache_path = root / "page.prepared.json"
            cv2.imwrite(str(image_path), np.full((12, 18, 3), 255, dtype=np.uint8))
            page = PreparedPage(
                image_path=image_path,
                output_path=original_output,
                image_bgr=np.full((12, 18, 3), 255, dtype=np.uint8),
                width=18,
                height=12,
                raw_blocks=[TextBlock("母", (1, 2, 3, 4), 90.0, "ctd")],
                render_blocks=[TextBlock("母", (1, 2, 3, 4), 90.0, "ctd")],
                skipped_blocks=[],
                grouping_report=[],
                page_order_report=[],
            )
            save_prepared_page_cache(page, cache_path)

            loaded = load_prepared_page_cache(cache_path, output_path=new_output)

        self.assertEqual(loaded.output_path, new_output)
        self.assertEqual(loaded.image_path, image_path)
        self.assertEqual(loaded.render_blocks[0].text, "母")


class TranslationCacheTests(unittest.TestCase):
    def test_translation_cache_preserves_line_id_state_and_source_compatibility(self) -> None:
        first = TextBlock("母", (1, 2, 30, 40), 90.0, "ctd", metadata={"line_id": "page_001"})
        second = TextBlock("母", (40, 2, 70, 40), 91.0, "ctd", metadata={"line_id": "page_002"})
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
            page_order_report=[],
            translations={"page_001": "Mother.", "page_002": "Mom."},
            translation_contexts={
                "page_001": {"translator": "qwen", "line_id": "page_001"},
                "page_002": {"translator": "qwen", "line_id": "page_002"},
            },
        )

        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "page.translation.json"
            save_translation_cache(page, cache_path)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            fresh = PreparedPage(
                image_path=page.image_path,
                output_path=page.output_path,
                image_bgr=None,
                width=100,
                height=100,
                raw_blocks=[],
                render_blocks=[first, second],
                skipped_blocks=[],
                grouping_report=[],
                page_order_report=[],
            )
            load_translation_cache(fresh, cache_path)

        self.assertEqual(payload["translations_by_id"], {"page_001": "Mother.", "page_002": "Mom."})
        self.assertEqual(payload["translation_contexts_by_id"]["page_001"]["translator"], "qwen")
        self.assertEqual(payload["translation_contexts_by_id"]["page_002"]["translator"], "qwen")
        self.assertEqual(payload["translations"], {"母": "Mom."})
        self.assertIn("translation_contexts", payload)
        state = TranslationReviewState.for_page(fresh)
        self.assertEqual(state.translation_for(first), "Mother.")
        self.assertEqual(state.translation_for(second), "Mom.")
        self.assertEqual(state.context_for(first)["translator"], "qwen")
        self.assertEqual(state.context_for(second)["translator"], "qwen")


if __name__ == "__main__":
    unittest.main()
