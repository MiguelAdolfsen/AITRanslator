from __future__ import annotations

import unittest

from translation_quality_autoresearch.common.io_utils import load_json, load_jsonl
from translation_quality_autoresearch.common.metrics import rank_candidates
from translation_quality_autoresearch.common.schemas import Candidate, ReferenceRecord, TranslationCase
from translation_quality_autoresearch.scripts.eval_translation_quality import score_one_candidate
from translation_quality_autoresearch.scorers.score_formula import load_weights


def score(candidate_id: str, value: float, *, hard: bool = False, agent: str = "a", cost: float = 1.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "agent": agent,
        "text": candidate_id,
        "candidate_quality_score": value,
        "hard_reject": hard,
        "deterministic": {"hard_fail": hard, "warnings": [], "glossary_violation": False, "japanese_leakage": hard, "assistant_chatter": False},
        "mqm": {"critical_errors": 0, "major_errors": 0},
        "cost_proxy": cost,
    }


class RerankerTests(unittest.TestCase):
    def test_selects_good_candidate(self) -> None:
        ranked = rank_candidates([score("bad", 20), score("good", 1)])
        self.assertEqual(ranked[0]["candidate_id"], "good")

    def test_selects_good_candidate_from_seed_frozen_outputs(self) -> None:
        cases = {row["case_id"]: TranslationCase.from_dict(row) for row in load_jsonl("translation_quality_autoresearch/benchmark/cases.jsonl")}
        refs = {row["case_id"]: ReferenceRecord.from_dict(row) for row in load_jsonl("translation_quality_autoresearch/benchmark/references.jsonl")}
        frozen = {row["case_id"]: row for row in load_jsonl("translation_quality_autoresearch/benchmark/frozen_agent_outputs.jsonl")}
        glossary = load_json("translation_quality_autoresearch/benchmark/glossary.json", default={})
        case = cases["seed_glossary_006"]
        rows = [
            score_one_candidate(case, Candidate.from_dict(candidate), refs[case.case_id], glossary, {}, load_weights())
            for candidate in frozen[case.case_id]["candidates"]
        ]
        ranked = rank_candidates(rows)
        self.assertEqual(ranked[0]["candidate_id"], "seed_glossary_006__good")

    def test_rejects_assistant_chatter(self) -> None:
        ranked = rank_candidates([score("chatter", 0, hard=True), score("good", 50)])
        self.assertEqual(ranked[0]["candidate_id"], "good")

    def test_rejects_japanese_leakage(self) -> None:
        ranked = rank_candidates([score("leak", 0, hard=True), score("good", 50)])
        self.assertEqual(ranked[0]["candidate_id"], "good")

    def test_tiebreaks_by_cost_after_quality(self) -> None:
        ranked = rank_candidates([score("expensive", 1, cost=10), score("cheap", 1, cost=1)])
        self.assertEqual(ranked[0]["candidate_id"], "cheap")


if __name__ == "__main__":
    unittest.main()
