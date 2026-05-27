from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from translation_quality_autoresearch.common.hashing import sha256_text
from translation_quality_autoresearch.common.io_utils import load_json
from translation_quality_autoresearch.scripts.eval_translation_quality import main as eval_main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def case_row(source: str = "お願いします") -> dict:
    return {
        "case_id": "case_001",
        "source_text": source,
        "normalized_source": source,
        "source_hash": sha256_text(source),
        "source_type": "dialogue",
        "ocr_risk": "clean",
        "context_before": [],
        "context_after": [],
        "forbidden_patterns": ["Translation:"],
        "must_preserve": [],
        "tags": ["test"],
    }


def eval_silent(argv: list[str]) -> int:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return eval_main(argv)


class EvalHandoffTests(unittest.TestCase):
    def test_replay_eval_rejects_candidate_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = case_row()
            write_jsonl(root / "cases.jsonl", [case])
            write_jsonl(root / "references.jsonl", [{"case_id": "case_001", "reference_translations": ["Please."]}])
            write_jsonl(
                root / "frozen.jsonl",
                [
                    {
                        "case_id": "case_001",
                        "source_hash": "sha256:bad",
                        "candidates": [{"candidate_id": "c1", "agent": "baseline", "text": "Please."}],
                    }
                ],
            )
            result = eval_silent(
                [
                    "--mode",
                    "replay",
                    "--cases",
                    str(root / "cases.jsonl"),
                    "--references",
                    str(root / "references.jsonl"),
                    "--frozen-outputs",
                    str(root / "frozen.jsonl"),
                    "--output",
                    str(root / "run"),
                    "--results",
                    str(root / "results.tsv"),
                    "--run-id",
                    "mismatch",
                    "--no-append-results",
                ]
            )
            self.assertEqual(result, 1)

    def test_eval_loads_human_gold_from_selected_benchmark_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = case_row()
            write_jsonl(root / "cases.jsonl", [case])
            write_jsonl(root / "references.jsonl", [{"case_id": "case_001", "reference_translations": ["Good."]}])
            write_jsonl(
                root / "frozen.jsonl",
                [
                    {
                        "case_id": "case_001",
                        "source_hash": case["source_hash"],
                        "candidates": [{"candidate_id": "c1", "agent": "baseline", "text": "Bad."}],
                    }
                ],
            )
            write_jsonl(
                root / "human_gold.jsonl",
                [
                    {
                        "case_id": "case_001",
                        "gold_rating": {"must_select_one_of": ["Good."]},
                        "severity_if_failed": "critical",
                    }
                ],
            )
            result = eval_silent(
                [
                    "--mode",
                    "replay",
                    "--cases",
                    str(root / "cases.jsonl"),
                    "--references",
                    str(root / "references.jsonl"),
                    "--frozen-outputs",
                    str(root / "frozen.jsonl"),
                    "--output",
                    str(root / "run"),
                    "--results",
                    str(root / "results.tsv"),
                    "--run-id",
                    "gold",
                    "--no-append-results",
                ]
            )
            self.assertEqual(result, 0)
            summary = load_json(root / "run" / "summary.json")
            self.assertTrue(summary["hard_reject"])
            self.assertEqual(summary["hard_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
