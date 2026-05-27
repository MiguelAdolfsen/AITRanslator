from __future__ import annotations

import unittest

from translation_quality_autoresearch.common.io_utils import load_jsonl
from translation_quality_autoresearch.scripts.report_benchmark_coverage import coverage_report, render_report


class BenchmarkCoverageTests(unittest.TestCase):
    def test_reports_gaps_for_seed_sized_benchmark(self) -> None:
        cases = [
            {
                "case_id": "one",
                "source_text": "助けて",
                "source_type": "dialogue",
                "ocr_risk": "clean",
                "tags": ["clean_dialogue"],
                "context_before": [],
                "context_after": [],
            }
        ]
        report = coverage_report(
            cases,
            references=[{"case_id": "one", "reference_translations": ["Help!"]}],
            human_gold=[],
            min_total_cases=2,
        )
        self.assertEqual(report["readiness"], "NEEDS_REVIEWED_BENCHMARK_EXPANSION")
        self.assertIn("total_cases", report["gaps"])

    def test_render_report_includes_readiness_summary(self) -> None:
        cases = []
        references = []
        human_gold = []
        for index in range(2):
            case_id = f"case_{index}"
            cases.append(
                {
                    "case_id": case_id,
                    "source_text": "ドン！",
                    "source_type": "sfx",
                    "ocr_risk": "clean",
                    "tags": [
                        "short_fragment",
                        "reaction",
                        "ambiguous_subject",
                        "ambiguous_speaker",
                        "names_and_honorifics",
                        "glossary_sensitive",
                        "ocr_noisy_but_salvageable",
                        "hallucination_trap",
                        "page_context_required",
                        "line_mapping",
                        "oververbose_guard",
                    ],
                    "context_before": ["前"],
                    "context_after": [],
                    "glossary_terms": ["term"],
                    "must_preserve": ["Name"],
                }
            )
            references.append({"case_id": case_id, "reference_translations": ["Boom!"]})
            human_gold.append({"case_id": case_id, "gold_rating": {}})
        report = coverage_report(cases, references=references, human_gold=human_gold, min_total_cases=1)
        report["targets"] = {name: 1 for name in report["targets"]}
        report = coverage_report(cases, references=references, human_gold=human_gold, min_total_cases=1)
        # This helper uses full production targets for tag categories, so a tiny benchmark stays not ready.
        self.assertEqual(report["readiness"], "NEEDS_REVIEWED_BENCHMARK_EXPANSION")
        self.assertIn("# Benchmark Coverage Report", render_report(report))

    def test_machine_assisted_profile_is_ready_for_v1(self) -> None:
        cases = load_jsonl("translation_quality_autoresearch/benchmark_v1/cases.jsonl")
        references = load_jsonl("translation_quality_autoresearch/benchmark_v1/references.jsonl")
        report = coverage_report(
            cases,
            references=references,
            human_gold=[],
            min_total_cases=60,
            readiness_profile="machine_assisted",
        )
        self.assertEqual(report["readiness"], "READY_FOR_MACHINE_ASSISTED_AGENT_HANDOFF")
        self.assertFalse(report["gaps"])


if __name__ == "__main__":
    unittest.main()
