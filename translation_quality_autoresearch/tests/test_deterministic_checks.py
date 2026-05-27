from __future__ import annotations

import unittest

from translation_quality_autoresearch.common.schemas import Candidate, TranslationCase
from translation_quality_autoresearch.scorers.deterministic_checks import check_candidate


def case(**updates):
    payload = {
        "case_id": "c1",
        "source_text": "\u65e9\u304f\u884c\u3053\u3046",
        "source_hash": "sha256:aad6c2e003b74124d4ee0c78a981f134d8ca20b9555a1575acbe624f75a791b1",
        "source_type": "dialogue",
        "ocr_risk": "clean",
        "context_before": [],
        "context_after": [],
        "forbidden_patterns": ["Translation:"],
        "must_preserve": [],
        "tags": ["test"],
        "glossary_terms": [],
    }
    payload.update(updates)
    return TranslationCase.from_dict(payload)


def candidate(text: str) -> Candidate:
    return Candidate(candidate_id="x", agent="a", text=text)


class DeterministicCheckTests(unittest.TestCase):
    def test_assistant_chatter_hard_fails(self) -> None:
        self.assertTrue(check_candidate(case(), candidate("As an AI, I cannot translate this."))["hard_fail"])

    def test_japanese_leakage_hard_fails(self) -> None:
        self.assertTrue(check_candidate(case(), candidate("\u65e9\u304f\u884c\u3053\u3046 - let's go"))["japanese_leakage"])

    def test_empty_output_hard_fails(self) -> None:
        self.assertTrue(check_candidate(case(), candidate(""))["empty_output"])

    def test_source_copied_hard_fails(self) -> None:
        self.assertTrue(check_candidate(case(), candidate("\u65e9\u304f\u884c\u3053\u3046"))["source_copied"])

    def test_forbidden_pattern_hard_fails(self) -> None:
        self.assertTrue(check_candidate(case(), candidate("Translation: Let's go"))["forbidden_pattern"])

    def test_glossary_required_term_missing_hard_fails(self) -> None:
        c = case(source_text="\u661f\u706f\u306e\u5951\u7d04", source_hash="sha256:7368e6873f1ffd30656661434db0f6b4464a587cd812d17cfd3dcba5989c2062", glossary_terms=["starlight_pact"])
        glossary = {"terms": [{"id": "starlight_pact", "source": "\u661f\u706f\u306e\u5951\u7d04", "target": "Starlight Pact"}]}
        self.assertTrue(check_candidate(c, candidate("the star lamp contract"), glossary=glossary)["glossary_violation"])

    def test_oververbose_warns(self) -> None:
        result = check_candidate(case(), candidate("This is an extremely long explanation that keeps going far beyond a manga bubble and explains the full situation in unnecessary detail."))
        self.assertTrue(result["oververbose"])
        self.assertIn("oververbose", result["warnings"])

    def test_normal_candidate_passes(self) -> None:
        result = check_candidate(case(), candidate("Let's hurry."))
        self.assertFalse(result["hard_fail"])


if __name__ == "__main__":
    unittest.main()
