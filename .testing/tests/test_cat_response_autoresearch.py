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
from cat_response_autoresearch.scripts.io_adapters import build_prompt, find_harness_cat_gguf, load_profile, profile_hash  # noqa: E402
from cat_response_autoresearch.scripts.score import score_case, summarize_metrics  # noqa: E402
from cat_response_autoresearch.scripts.validate_fixtures import validate_benchmark  # noqa: E402
from manga_local_translator.hf_translators import build_cat_prompt, cat_num_predict  # noqa: E402


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
        explanatory = score_case(
            case,
            {
                "raw_output": 'The Japanese phrase "何あれ" translates to "What is that?" depending on context.',
                "final_output": "What is that?",
                "reject_reason": "",
            },
            reference,
        )
        dangling = score_case(case, {"raw_output": "such,", "final_output": "such,", "reject_reason": ""}, reference)
        unbacked_apology = score_case(
            {"case_id": "case-2", "source_text": "残念な", "source_type": "short_fragment"},
            {"raw_output": "I'm sorry about that.", "final_output": "I'm sorry about that.", "reject_reason": ""},
            reference,
        )
        sfx_romaji = score_case(
            {"case_id": "case-3", "source_text": "ドキドキ", "source_type": "sfx"},
            {"raw_output": "Dokidoki", "final_output": "Dokidoki", "reject_reason": ""},
            reference,
        )
        real_apology = score_case(
            {"case_id": "case-4", "source_text": "ごめん", "source_type": "short_fragment"},
            {"raw_output": "Sorry.", "final_output": "Sorry.", "reject_reason": ""},
            reference,
        )

        self.assertIn("accepted_prompt_chatter", chatter["violations"])
        self.assertIn("accepted_japanese_leakage", japanese["violations"])
        self.assertIn("accepted_prompt_fragment", prompt["violations"])
        self.assertIn("false_reject", rejected["violations"])
        self.assertIn("accepted_explanatory_output", explanatory["violations"])
        self.assertIn("accepted_fragment_shape_warning", dangling["violations"])
        self.assertIn("accepted_fragment_shape_warning", unbacked_apology["violations"])
        self.assertIn("accepted_fragment_shape_warning", sfx_romaji["violations"])
        self.assertNotIn("accepted_fragment_shape_warning", real_apology["violations"])
        self.assertEqual(chatter["outcome_class"], "unsafe_accept")
        self.assertEqual(rejected["outcome_class"], "false_reject")
        self.assertEqual(explanatory["outcome_class"], "weak_accept")
        self.assertTrue(summarize_metrics([chatter])["hard_failure"])

    def test_scoring_catches_semantic_required_and_forbidden_terms(self) -> None:
        case = {
            "case_id": "semantic-1",
            "source_text": "あっちはうんこだ。",
            "source_type": "semantic_trap",
        }
        reference = {
            "case_id": "semantic-1",
            "expected_decision": "accept",
            "forbidden_patterns": [],
            "forbidden_meaning_terms": ["dog"],
            "required_meaning_terms": ["poop|shit|crap"],
            "allowed_japanese_output": False,
            "max_chars": 80,
            "max_words": 12,
        }

        wrong = score_case(
            case,
            {"raw_output": "The other side is a dog.", "final_output": "The other side is a dog.", "reject_reason": ""},
            reference,
        )
        right = score_case(
            case,
            {"raw_output": "That side is crap.", "final_output": "That side is crap.", "reject_reason": ""},
            reference,
        )

        self.assertIn("accepted_forbidden_meaning_term", wrong["violations"])
        self.assertIn("accepted_required_meaning_missing", wrong["violations"])
        self.assertEqual(wrong["outcome_class"], "unsafe_accept")
        self.assertEqual(right["violations"], [])
        self.assertTrue(right["approved"])

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

    def test_cat_gguf_lookup_uses_repo_root_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / ".models" / "CAT-Translate"
            model_dir.mkdir(parents=True)
            old_model = model_dir / "CAT-Translate-7b.Q4_K_M.gguf"
            q8_model = model_dir / "CAT-Translate-7b.Q8_0.gguf"
            old_model.write_text("q4", encoding="utf-8")
            q8_model.write_text("q8", encoding="utf-8")

            found = find_harness_cat_gguf(root)

        self.assertEqual(found, q8_model.resolve())

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

    def test_best_update_can_keep_retry_reduction_when_quality_ties(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            best_path = Path(temp_dir) / "best.json"
            best_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "best_run_id": "old",
                        "best_quality_score": 0.0,
                        "best_score": 0.0,
                        "cat_approval_rate": 1.0,
                        "retry_rate": 0.5,
                        "benchmark_fingerprint": "same",
                    }
                ),
                encoding="utf-8",
            )
            retry_reduction = {
                "hard_failure": False,
                "cat_quality_score": 0.0,
                "cat_response_score": 800.0,
                "cat_latency_score": 800.0,
                "cat_approval_rate": 1.0,
                "retry_rate": 0.25,
                "retried_count": 1,
                "primary_rejected_rate": 0.25,
            }

            self.assertTrue(
                maybe_update_best(
                    best_path,
                    "less-retry",
                    "commit",
                    retry_reduction,
                    benchmark_hash="same",
                    disabled=False,
                    min_improvement=0.002,
                )
            )

            best = json.loads(best_path.read_text(encoding="utf-8"))
        self.assertEqual(best["best_run_id"], "less-retry")
        self.assertEqual(best["keep_reason"], "retry_dependence_improved")
        self.assertEqual(best["retry_rate"], 0.25)

    def test_best_update_can_keep_response_tiebreak_when_quality_approval_and_retry_tie(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            best_path = Path(temp_dir) / "best.json"
            best_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "best_run_id": "old",
                        "best_quality_score": 0.0,
                        "best_response_score": 1000.0,
                        "cat_approval_rate": 1.0,
                        "retry_rate": 0.0,
                        "benchmark_fingerprint": "same",
                    }
                ),
                encoding="utf-8",
            )
            response_improvement = {
                "hard_failure": False,
                "cat_quality_score": 0.0,
                "cat_response_score": 900.0,
                "cat_latency_score": 900.0,
                "cat_approval_rate": 1.0,
                "retry_rate": 0.0,
                "retried_count": 0,
                "primary_rejected_rate": 0.0,
            }

            self.assertTrue(
                maybe_update_best(
                    best_path,
                    "faster-tie",
                    "commit",
                    response_improvement,
                    benchmark_hash="same",
                    disabled=False,
                    min_improvement=0.002,
                )
            )

            best = json.loads(best_path.read_text(encoding="utf-8"))
        self.assertEqual(best["best_run_id"], "faster-tie")
        self.assertEqual(best["keep_reason"], "response_score_tiebreak_improved")

    def test_best_update_falls_back_to_last_kept_result_for_same_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            best_path = root / "best.json"
            results_path = root / "results.tsv"
            eval_cat_responses.append_result(
                results_path,
                {
                    "run_id": "previous-real-mined",
                    "commit": "old",
                    "benchmark_set": "real_mined",
                    "cat_quality_score": 100.0,
                    "cat_response_score": 120.0,
                    "cat_latency_score": 20.0,
                    "cat_approval_rate": 0.9,
                    "retry_rate": 0.5,
                    "retried_count": 5,
                    "primary_rejected_rate": 0.5,
                    "hard_failure": False,
                    "kept": True,
                },
            )
            worse = {
                "hard_failure": False,
                "cat_quality_score": 101.0,
                "cat_response_score": 121.0,
                "cat_latency_score": 20.0,
                "cat_approval_rate": 0.9,
                "retry_rate": 0.5,
            }
            better = {
                "hard_failure": False,
                "cat_quality_score": 90.0,
                "cat_response_score": 110.0,
                "cat_latency_score": 20.0,
                "cat_approval_rate": 0.9,
                "retry_rate": 0.5,
                "retried_count": 4,
                "primary_rejected_rate": 0.4,
            }

            self.assertFalse(
                maybe_update_best(
                    best_path,
                    "worse",
                    "commit",
                    worse,
                    benchmark_hash="new-real-mined-hash",
                    benchmark_set="real_mined",
                    disabled=False,
                    min_improvement=0.01,
                )
            )
            self.assertTrue(
                maybe_update_best(
                    best_path,
                    "better",
                    "commit",
                    better,
                    benchmark_hash="new-real-mined-hash",
                    benchmark_set="real_mined",
                    disabled=False,
                    min_improvement=0.01,
                )
            )

            best = json.loads(best_path.read_text(encoding="utf-8"))
        self.assertEqual(best["best_run_id"], "better")
        self.assertEqual(best["benchmark_set"], "real_mined")
        self.assertEqual(best["keep_reason"], "quality_margin_improved")

    def test_prompt_construction_uses_official_template(self) -> None:
        profile = load_profile("official")

        self.assertEqual(
            build_prompt("母", profile),
            "Translate the following Japanese text into English.\n\n母",
        )

    def test_production_profile_uses_production_cat_prompt_and_token_cap(self) -> None:
        profile = load_profile("production")

        self.assertEqual(build_prompt("母", profile), build_cat_prompt("母"))
        self.assertEqual(build_prompt("母", profile, retry=True), build_cat_prompt("母", retry=True))
        self.assertEqual(profile["num_predict"], cat_num_predict())

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
            self.assertIn("retry_rate", summary["metrics"])
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

    def test_real_mined_fixtures_validate_and_fake_smoke_runs(self) -> None:
        for benchmark_name in ("real_mined", "real_mined_holdout"):
            benchmark = REPO_ROOT / "cat_response_autoresearch" / "benchmarks" / benchmark_name
            with self.subTest(benchmark=benchmark_name):
                self.assertEqual(validate_benchmark(benchmark), [])
                with tempfile.TemporaryDirectory() as temp_dir:
                    output = Path(temp_dir) / "run"
                    results = Path(temp_dir) / "results.tsv"
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
                                f"{benchmark_name}-fake",
                                "--fake-outputs",
                                str(benchmark / "fake_outputs.jsonl"),
                                "--repeats",
                                "1",
                                "--no-update-best",
                            ]
                        )
                    self.assertEqual(code, 0)
                    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["metrics"]["cat_approval_rate"], 1.0)
                    self.assertIn("production_cat", summary)

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
