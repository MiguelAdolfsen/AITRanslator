from __future__ import annotations

import unittest

from translation_quality_autoresearch.scorers.score_formula import candidate_score, run_score


def row(hard: bool = False, critical: int = 0, overlap: float = 1.0) -> dict:
    return {
        "deterministic": {
            "hard_fail": hard,
            "assistant_chatter": hard,
            "japanese_leakage": False,
            "empty_output": False,
            "source_copied": False,
            "forbidden_pattern": False,
            "glossary_violation": False,
            "line_mapping_error": False,
            "warnings": [],
        },
        "mqm": {"critical_errors": critical, "major_errors": 0, "minor_errors": 0},
        "mt_metrics": {"reference_token_f1": overlap},
        "style": {"length_ratio": 1.0, "punctuation_ok": True},
        "cost_proxy": 0,
        "latency_ms": 0,
    }


class ScoreFormulaTests(unittest.TestCase):
    def test_hard_rejected_candidate_loses(self) -> None:
        self.assertGreater(candidate_score(row(hard=True)), candidate_score(row(hard=False, overlap=0.0)))

    def test_critical_errors_dominate_style(self) -> None:
        self.assertGreater(candidate_score(row(critical=1, overlap=1.0)), candidate_score(row(critical=0, overlap=0.0)))

    def test_bonuses_cannot_rescue_hard_failures(self) -> None:
        self.assertGreater(candidate_score(row(hard=True, overlap=1.0)), candidate_score(row(hard=False, overlap=0.0)))

    def test_run_score_lower_for_fewer_failures(self) -> None:
        good = [{"case_score": 10, "hard_failures": []}]
        bad = [{"case_score": 10, "hard_failures": ["x"]}]
        self.assertLess(run_score(good, []), run_score(bad, []))


if __name__ == "__main__":
    unittest.main()
