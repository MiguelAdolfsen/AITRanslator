from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cat_response_autoresearch.scripts import eval_cat_responses  # noqa: E402
from cat_response_autoresearch.scripts.eval_cat_responses import maybe_update_best  # noqa: E402
from cat_response_autoresearch.scripts.serve_dashboard import dashboard_state  # noqa: E402
from cat_response_autoresearch.scripts.io_adapters import build_prompt, load_profile, profile_hash  # noqa: E402
from cat_response_autoresearch.scripts.score import score_case, summarize_metrics  # noqa: E402
from cat_response_autoresearch.scripts.validate_fixtures import validate_benchmark  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class CatResponseAutoresearchTests(unittest.TestCase):
    def test_validate_fixtures_rejects_missing_reference_and_duplicate_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = Path(temp_dir)
            write_jsonl(
                benchmark / "cases.jsonl",
                [
                    {
                        "schema_version": 1,
                        "case_id": "dup",
                        "source_text": "母",
                        "source_type": "short_fragment",
                        "risk_labels": [],
                    },
                    {
                        "schema_version": 1,
                        "case_id": "dup",
                        "source_text": "父",
                        "source_type": "short_fragment",
                        "risk_labels": [],
                    },
                ],
            )
            write_jsonl(
                benchmark / "references.jsonl",
                [
                    {
                        "schema_version": 1,
                        "case_id": "other",
                        "expected_decision": "accept",
                        "forbidden_patterns": [],
                        "allowed_japanese_output": False,
                        "max_chars": 40,
                        "max_words": 8,
                    }
                ],
            )

            errors = validate_benchmark(benchmark)

        joined = "\n".join(errors)
        self.assertIn("duplicate case_id", joined)
        self.assertIn("case_id mismatch", joined)

    def test_scoring_catches_chatter_japanese_prompt_and_false_reject(self) -> None:
        case = {
            "case_id": "case-1",
            "source_text": "何あれ",
            "source_type": "short_fragment",
        }
        reference = {
            "case_id": "case-1",
            "expected_decision": "accept",
            "forbidden_patterns": ["please provide"],
            "allowed_japanese_output": False,
            "max_chars": 40,
            "max_words": 8,
        }

        chatter = score_case(
            case,
            {
                "raw_output": "Could you please provide the Japanese passage?",
                "final_output": "Could you please provide the Japanese passage?",
                "reject_reason": "",
            },
            reference,
        )
        japanese = score_case(case, {"raw_output": "母です", "final_output": "母です", "reject_reason": ""}, reference)
        prompt = score_case(
            case,
            {
                "raw_output": "Translate the following Japanese text into English.",
                "final_output": "Translate the following Japanese text into English.",
                "reject_reason": "",
            },
            reference,
        )
        rejected = score_case(case, {"raw_output": "", "final_output": "", "reject_reason": "cat_empty"}, reference)

        self.assertIn("accepted_prompt_chatter", chatter["violations"])
        self.assertIn("accepted_japanese_leakage", japanese["violations"])
        self.assertIn("accepted_prompt_fragment", prompt["violations"])
        self.assertIn("false_reject", rejected["violations"])
        self.assertEqual(chatter["outcome_class"], "unsafe_accept")
        self.assertEqual(rejected["outcome_class"], "false_reject")
        self.assertTrue(summarize_metrics([chatter])["hard_failure"])

    def test_category_metrics_make_low_category_a_hard_failure(self) -> None:
        rows = [
            {
                "case_id": "a",
                "source_type": "short_fragment",
                "approved": False,
                "outcome_class": "false_reject",
                "latency_ms": 0.0,
                "false_reject_count": 1,
            },
            {
                "case_id": "b",
                "source_type": "short_fragment",
                "approved": True,
                "outcome_class": "clean_accept",
                "latency_ms": 0.0,
                "false_reject_count": 0,
            },
        ]

        metrics = summarize_metrics(rows)

        self.assertEqual(metrics["category_metrics"]["short_fragment"]["approval_rate"], 0.5)
        self.assertEqual(metrics["low_categories"], "short_fragment")
        self.assertTrue(metrics["hard_failure"])

    def test_profile_hash_changes_when_settings_change(self) -> None:
        official = load_profile("official")
        changed = {**official, "num_predict": int(official["num_predict"]) + 1}

        self.assertNotEqual(profile_hash(official), profile_hash(changed))

    def test_best_update_ignores_latency_only_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            best_path = Path(temp_dir) / "best.json"
            best_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "best_run_id": "old",
                        "best_quality_score": 100.0,
                        "best_score": 100.0,
                        "cat_approval_rate": 0.9,
                        "benchmark_fingerprint": "same",
                    }
                ),
                encoding="utf-8",
            )
            latency_only = {
                "hard_failure": False,
                "cat_quality_score": 100.0,
                "cat_response_score": 90.0,
                "cat_latency_score": -10.0,
                "cat_approval_rate": 0.9,
            }
            marginal = {
                "hard_failure": False,
                "cat_quality_score": 99.5,
                "cat_response_score": 99.5,
                "cat_latency_score": 0.0,
                "cat_approval_rate": 0.9,
            }
            real_improvement = {
                "hard_failure": False,
                "cat_quality_score": 98.9,
                "cat_response_score": 98.9,
                "cat_latency_score": 0.0,
                "cat_approval_rate": 0.9,
            }

            self.assertFalse(
                maybe_update_best(best_path, "latency", "commit", latency_only, benchmark_hash="same", disabled=False, min_improvement=0.01)
            )
            self.assertFalse(
                maybe_update_best(best_path, "marginal", "commit", marginal, benchmark_hash="same", disabled=False, min_improvement=0.01)
            )
            self.assertTrue(
                maybe_update_best(best_path, "better", "commit", real_improvement, benchmark_hash="same", disabled=False, min_improvement=0.01)
            )

            best = json.loads(best_path.read_text(encoding="utf-8"))
        self.assertEqual(best["best_run_id"], "better")
        self.assertEqual(best["keep_reason"], "quality_margin_improved")

    def test_prompt_construction_uses_official_template(self) -> None:
        profile = load_profile("official")

        self.assertEqual(
            build_prompt("母", profile),
            "Translate the following Japanese text into English.\n\n母",
        )

    def test_eval_fake_outputs_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark = root / "benchmark"
            output = root / "run"
            results = root / "results.tsv"
            write_jsonl(
                benchmark / "cases.jsonl",
                [
                    {
                        "schema_version": 1,
                        "case_id": "accept-1",
                        "source_text": "何あれ",
                        "source_type": "short_fragment",
                        "risk_labels": ["short_fragment"],
                    },
                    {
                        "schema_version": 1,
                        "case_id": "reject-1",
                        "source_text": "▽日１回健人『アベツオナ",
                        "source_type": "metadata_or_noise",
                        "risk_labels": ["metadata_like"],
                    },
                ],
            )
            write_jsonl(
                benchmark / "references.jsonl",
                [
                    {
                        "schema_version": 1,
                        "case_id": "accept-1",
                        "expected_decision": "accept",
                        "forbidden_patterns": ["please provide"],
                        "allowed_japanese_output": False,
                        "max_chars": 40,
                        "max_words": 8,
                    },
                    {
                        "schema_version": 1,
                        "case_id": "reject-1",
                        "expected_decision": "reject",
                        "forbidden_patterns": ["daily"],
                        "allowed_japanese_output": False,
                        "max_chars": 0,
                        "max_words": 0,
                    },
                ],
            )
            fake = benchmark / "fake_outputs.jsonl"
            write_jsonl(
                fake,
                [
                    {"case_id": "accept-1", "raw_output": "What's that?"},
                    {"case_id": "reject-1", "raw_output": "Could you please provide the Japanese passage?"},
                ],
            )

            with redirect_stdout(StringIO()):
                code = eval_cat_responses.main(
                    [
                        "--benchmark",
                        str(benchmark),
                        "--output",
                        str(output),
                        "--results",
                        str(results),
                        "--run-id",
                        "fake",
                        "--fake-outputs",
                        str(fake),
                        "--repeats",
                        "2",
                        "--no-update-best",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue((output / "raw_outputs.jsonl").exists())
            self.assertTrue((output / "per_case_metrics.jsonl").exists())
            self.assertTrue((output / "human_review.md").exists())
            self.assertTrue((output / "comparison.md").exists())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"]["cat_approval_rate"], 1.0)
            self.assertEqual(summary["metrics"]["repeat_count"], 2)
            self.assertEqual(summary["metrics"]["cases_total"], 2)
            self.assertEqual(summary["metrics"]["evaluations_total"], 4)
            self.assertEqual(len(summary["repeat_metrics"]), 2)
            raw_rows = [
                json.loads(line)
                for line in (output / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(raw_rows), 4)

    def test_dashboard_state_reads_results_best_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results = root / "results.tsv"
            best = root / "best.json"
            runs = root / "runs"
            run_dir = runs / "run-a"
            run_dir.mkdir(parents=True)
            results.write_text(
                "\t".join(["run_id", "cat_response_score", "cat_approval_rate", "hard_failure", "kept", "repeat_count", "low_categories"])
                + "\n"
                + "\t".join(["run-a", "12.5", "0.9", "False", "True", "4", ""])
                + "\n",
                encoding="utf-8",
            )
            best.write_text(
                json.dumps({"best_run_id": "run-a", "best_score": 12.5, "cat_approval_rate": 0.9}),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps({"profile": {"name": "official"}, "metrics": {"category_metrics": {"short_fragment": {"approval_rate": 0.9, "evaluations": 4}}}}),
                encoding="utf-8",
            )
            (run_dir / "progress.json").write_text(
                json.dumps({"run_id": "run-a", "status": "running", "completed_evaluations": 2, "evaluations_total": 8}),
                encoding="utf-8",
            )

            state = dashboard_state(results, best, runs)

        self.assertEqual(state["latest_result"]["run_id"], "run-a")
        self.assertEqual(state["best"]["best_run_id"], "run-a")
        self.assertEqual(state["active_progress"]["status"], "running")
        self.assertEqual(state["best_summary"]["profile"]["name"], "official")


if __name__ == "__main__":
    unittest.main()
