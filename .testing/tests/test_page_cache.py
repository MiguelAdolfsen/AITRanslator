from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from manga_local_translator.detect_types import TextBlock
from manga_local_translator.page_cache import load_prepared_page_cache, save_prepared_page_cache
from manga_local_translator.page_types import PreparedPage

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


if __name__ == "__main__":
    unittest.main()
