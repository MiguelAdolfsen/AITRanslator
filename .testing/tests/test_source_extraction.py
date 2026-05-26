import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from manga_local_translator.config import PipelineConfig
from manga_local_translator.detect_types import TextBlock
from manga_local_translator.source_extraction import extract_source_page


class SourceExtractionTests(unittest.TestCase):
    def test_extract_source_page_runs_ocr_filter_group_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            output_path = Path(tmp) / "page.out.png"
            Image.new("RGB", (200, 200), "white").save(image_path)
            ocr_block = TextBlock(
                text="こんにちは",
                box=(40, 30, 80, 100),
                confidence=92.0,
                detector="test",
                metadata={"vertical": True},
            )

            with mock.patch("manga_local_translator.detect_ocr.run_ocr", return_value=[ocr_block]) as run_ocr:
                result = extract_source_page(image_path, output_path, PipelineConfig())

        run_ocr.assert_called_once()
        self.assertEqual(result.image_path, image_path)
        self.assertEqual(result.output_path, output_path)
        self.assertEqual(result.width, 200)
        self.assertEqual(result.height, 200)
        self.assertEqual(len(result.raw_blocks), 1)
        self.assertEqual(len(result.render_blocks), 1)
        self.assertEqual(result.render_blocks[0].text, "こんにちは")
        self.assertIn("block_id", result.raw_blocks[0].metadata)
        self.assertIn("line_id", result.render_blocks[0].metadata)
        self.assertEqual(result.page_order_report[0]["source_text"], "こんにちは")


if __name__ == "__main__":
    unittest.main()
