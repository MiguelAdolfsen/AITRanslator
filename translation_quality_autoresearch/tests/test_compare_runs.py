from __future__ import annotations

import unittest

from translation_quality_autoresearch.scripts.compare_runs import compare_summaries


class CompareRunsTests(unittest.TestCase):
    def test_keep_requires_score_improvement_without_guardrail_regression(self) -> None:
        baseline = {
            "translation_quality_score": 10,
            "hard_reject": False,
            "critical_mqm_errors": 0,
            "japanese_leakage_count": 1,
            "assistant_chatter_count": 1,
            "glossary_violation_count": 1,
            "hard_failure_count": 0,
        }
        candidate = {
            **baseline,
            "translation_quality_score": 9,
        }
        self.assertEqual(compare_summaries(baseline, candidate)["recommendation"], "KEEP")

    def test_revert_on_japanese_leakage_regression_even_when_score_improves(self) -> None:
        baseline = {
            "translation_quality_score": 10,
            "hard_reject": False,
            "critical_mqm_errors": 0,
            "japanese_leakage_count": 0,
            "assistant_chatter_count": 0,
            "glossary_violation_count": 0,
            "hard_failure_count": 0,
        }
        candidate = {
            **baseline,
            "translation_quality_score": 9,
            "japanese_leakage_count": 1,
        }
        comparison = compare_summaries(baseline, candidate)
        self.assertEqual(comparison["recommendation"], "REVERT")
        self.assertEqual(comparison["metric_regressions"]["japanese_leakage_count"], 1)

    def test_rejects_cross_benchmark_comparison(self) -> None:
        baseline = {"benchmark_id": "benchmark_a", "translation_quality_score": 10}
        candidate = {"benchmark_id": "benchmark_b", "translation_quality_score": 9}
        with self.assertRaises(ValueError):
            compare_summaries(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
