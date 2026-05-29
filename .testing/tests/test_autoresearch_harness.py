import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "source_extraction_autoresearch" / "scripts"
if not SCRIPTS_DIR.exists():
    raise unittest.SkipTest("source_extraction_autoresearch is a local ignored harness")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_source_extraction import best_sort_key_from_metrics, decision_preflight_errors, forbidden_dirty_files  # noqa: E402
from run_benchmark_matrix import DEFAULT_MATRIX  # noqa: E402


class AutoresearchHarnessTests(unittest.TestCase):
    def test_forbidden_path_matcher_allows_evaluator_owned_outputs(self):
        dirty = [
            "source_extraction_autoresearch/runs/current/summary.json",
            "source_extraction_autoresearch/results/best.synthetic.json",
        ]

        self.assertEqual(forbidden_dirty_files(dirty, allow_dirty_results=False), [])

    def test_forbidden_path_matcher_rejects_frozen_harness_and_unrelated_subsystems(self):
        dirty = [
            "source_extraction_autoresearch/benchmarks/synthetic/labels/page_001.labels.json",
            "source_extraction_autoresearch/scripts/score.py",
            "translation_routing_autoresearch/README.md",
            "manga_local_translator/render.py",
            ".testing/tests/test_grouping.py",
        ]

        self.assertEqual(forbidden_dirty_files(dirty, allow_dirty_results=False), sorted(dirty))

    def test_results_tsv_requires_explicit_allowance_when_dirty_before_run(self):
        dirty = ["source_extraction_autoresearch/results/results.tsv"]

        self.assertEqual(forbidden_dirty_files(dirty, allow_dirty_results=False), dirty)
        self.assertEqual(forbidden_dirty_files(dirty, allow_dirty_results=True), [])

    def test_matrix_default_and_historical_entries_are_named_for_stable_run_ids(self):
        names = [item[0] for item in DEFAULT_MATRIX]

        self.assertEqual(names, ["synthetic", "v2", "spy_v1", "hard_v1"])
        self.assertTrue(DEFAULT_MATRIX[0][3])
        self.assertFalse(DEFAULT_MATRIX[1][3])

    def test_decision_preflight_rejects_smoke_draft_skip_validation_dirty_and_missing_tests(self):
        errors = decision_preflight_errors(
            decision_benchmark=True,
            adapter_smoke=True,
            draft_benchmark=True,
            skip_validation=True,
            dirty_forbidden=["source_extraction_autoresearch/scripts/score.py"],
            allow_dirty_forbidden=False,
            tests_available=False,
            tests_ok=False,
            allow_missing_tests=False,
        )

        joined = "\n".join(errors)
        self.assertIn("--adapter-smoke", joined)
        self.assertIn("draft", joined)
        self.assertIn("--skip-validation", joined)
        self.assertIn("score.py", joined)
        self.assertIn("--run-tests", joined)

    def test_decision_preflight_accepts_explicit_overrides(self):
        errors = decision_preflight_errors(
            decision_benchmark=True,
            adapter_smoke=False,
            draft_benchmark=False,
            skip_validation=False,
            dirty_forbidden=["source_extraction_autoresearch/scripts/score.py"],
            allow_dirty_forbidden=True,
            tests_available=False,
            tests_ok=False,
            allow_missing_tests=True,
        )

        self.assertEqual(errors, [])

    def test_best_tie_breaker_uses_timing_after_quality_and_counts(self):
        base = {
            "source_extraction_score": 10.0,
            "missed_dialogue_region_count": 0,
            "destructive_false_positive_count": 0,
            "mean_ocr_cer": 0.0,
            "severe_ocr_error_count": 0,
            "overmerge_count": 0,
            "undermerge_count": 0,
            "reading_order_error_count": 0,
            "orientation_error_count": 0,
        }
        fast = {**base, "p95_extraction_ms_per_page": 100.0}
        slow = {**base, "p95_extraction_ms_per_page": 200.0}
        lower_quality = {**base, "source_extraction_score": 9.0, "p95_extraction_ms_per_page": 1000.0}

        self.assertLess(best_sort_key_from_metrics(fast, changed_count=2), best_sort_key_from_metrics(slow, changed_count=2))
        self.assertLess(best_sort_key_from_metrics(lower_quality, changed_count=2), best_sort_key_from_metrics(fast, changed_count=2))


if __name__ == "__main__":
    unittest.main()
