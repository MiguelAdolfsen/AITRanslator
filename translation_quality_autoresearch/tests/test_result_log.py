from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translation_quality_autoresearch.common.result_log import append_result_row


class ResultLogTests(unittest.TestCase):
    def test_appends_rows_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            row = {"timestamp": "t", "run_id": "r", "translation_quality_score": 1.0}
            append_result_row(path, row)
            append_result_row(path, {**row, "run_id": "r2"})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn("run_id", lines[0])
            self.assertIn("benchmark_id", lines[0])
            self.assertIn("reference_policy", lines[0])
            self.assertIn("r2", lines[2])


if __name__ == "__main__":
    unittest.main()
