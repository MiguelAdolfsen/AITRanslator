from __future__ import annotations

import unittest
from pathlib import Path

from translation_quality_autoresearch.common.hashing import sha256_text
from translation_quality_autoresearch.common.io_utils import load_jsonl
from translation_quality_autoresearch.common.schemas import TranslationCase
from translation_quality_autoresearch.scripts.validate_benchmark import validate_benchmark


class SchemaTests(unittest.TestCase):
    def test_loads_seed_cases(self) -> None:
        rows = load_jsonl(Path("translation_quality_autoresearch/benchmark/cases.jsonl"))
        self.assertGreaterEqual(len(rows), 12)
        cases = [TranslationCase.from_dict(row) for row in rows]
        self.assertTrue(all(case.case_id for case in cases))

    def test_hash_generation_is_stable(self) -> None:
        self.assertEqual(
            sha256_text("\u304a\u9858\u3044\u3057\u307e\u3059"),
            "sha256:c5c3df248eaaa492712e0f65960e8e3f3bae2552ee807a4132574534c88f64c7",
        )

    def test_hash_mismatch_is_caught(self) -> None:
        row = load_jsonl(Path("translation_quality_autoresearch/benchmark/cases.jsonl"))[0]
        row = dict(row)
        row["source_hash"] = "sha256:bad"
        case = TranslationCase.from_dict(row)
        self.assertTrue(any("source_hash mismatch" in error for error in case.validate()))

    def test_v1_benchmark_validates_with_version_metadata(self) -> None:
        self.assertEqual(validate_benchmark(Path("translation_quality_autoresearch/benchmark_v1")), [])


if __name__ == "__main__":
    unittest.main()
