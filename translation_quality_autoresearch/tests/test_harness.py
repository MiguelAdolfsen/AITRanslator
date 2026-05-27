from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from manga_local_translator.vision_types import VisionFactsRequest
from manga_local_translator.vision_validation import parse_vision_facts_response, vision_facts_acceptance_reason

from translation_quality_autoresearch.scripts import io_adapters, profiles, scoring
from translation_quality_autoresearch.scripts.validate_benchmark import validate_benchmark


BENCHMARK = Path("translation_quality_autoresearch/benchmarks/frieren_ch26_pages_002_010")


class BenchmarkValidationTests(unittest.TestCase):
    def test_frieren_benchmark_validates(self) -> None:
        self.assertEqual(validate_benchmark(BENCHMARK), [])

    def test_validator_rejects_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark = root / "bench"
            labels = benchmark / "labels"
            labels.mkdir(parents=True)
            image = root / "page.jpg"
            image.write_bytes(b"fake")
            cache = root / "prepared.json"
            row = {
                "page_id": "page_001",
                "line_id": "001_001",
                "page_order": 1,
                "source_text": "source",
                "source_box": [1, 2, 3, 4],
                "block_id": "ocr_0001",
                "source_hash": "label_hash",
                "reference_text": "Reference.",
                "alignment_confidence": "high",
                "reference_role": "speech",
                "skip_reference": False,
                "skip_reason": "",
                "notes": "",
            }
            cache.write_text(
                json.dumps(
                    {
                        "page_order_report": [
                            {
                                "line_id": "001_001",
                                "source_text": "source",
                                "box": [1, 2, 3, 4],
                                "source_hash": "cache_hash",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            page = {
                "schema_version": 1,
                "dataset": "bench",
                "page_id": "page_001",
                "page_number": 1,
                "japanese_image": str(image),
                "english_image": str(image),
                "prepared_cache": str(cache),
                "image_size": [10, 10],
                "labels": [row],
            }
            (labels / "page_001.references.json").write_text(json.dumps(page), encoding="utf-8")
            (benchmark / "references.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (benchmark / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "bench",
                        "pages": ["page_001"],
                        "splits": {"main": ["page_001"], "holdout": ["page_001_holdout"]},
                    }
                ),
                encoding="utf-8",
            )
            errors = validate_benchmark(benchmark)
            self.assertTrue(any("source_hash mismatch" in error for error in errors), errors)


class VisionFactsParserTests(unittest.TestCase):
    def test_parser_accepts_optional_fields(self) -> None:
        rows = parse_vision_facts_response(
            json.dumps(
                {
                    "lines": [
                        {
                            "line_id": "001_001",
                            "number": 1,
                            "source_text": "source",
                            "bubble_type": "speech",
                            "line_role": "speech",
                            "speaker_position": "left",
                            "speaker_anchor": "character left of bubble",
                            "visible_emotion": "angry",
                            "tone_hint": "angry",
                            "observable_action": "pointing",
                            "mapping_confidence": "high",
                            "facts": ["speech bubble near character"],
                            "context_hints": ["angry tone"],
                            "needs_review": False,
                            "risk_flags": [],
                            "warnings": [],
                        }
                    ]
                }
            )
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].line_role, "speech")
        self.assertEqual(rows[0].speaker_anchor, "character left of bubble")
        self.assertEqual(rows[0].context_hints, ("angry tone",))

    def test_acceptance_rejects_named_speaker_anchor(self) -> None:
        rows = parse_vision_facts_response(
            json.dumps(
                {
                    "lines": [
                        {
                            "line_id": "001_001",
                            "number": 1,
                            "source_text": "source",
                            "bubble_type": "speech",
                            "line_role": "speech",
                            "speaker_position": "left side",
                            "speaker_anchor": "Frieren",
                            "visible_emotion": "unknown",
                            "tone_hint": "unknown",
                            "observable_action": "unknown",
                            "mapping_confidence": "high",
                            "facts": [],
                            "context_hints": [],
                            "needs_review": False,
                            "risk_flags": [],
                            "warnings": [],
                        }
                    ]
                }
            )
        )
        request = VisionFactsRequest(line_id="001_001", number=1, source_text="source", box=(0, 0, 10, 10))
        self.assertEqual(vision_facts_acceptance_reason(request, rows[0]), "ungrounded_speaker_anchor")


class ProfileControlTests(unittest.TestCase):
    def test_profile_rejects_uncontrolled_prompt_keys(self) -> None:
        profile = {
            "schema_version": 1,
            "name": "bad",
            "enabled_fields": ["facts"],
            "prompt": {"task": "describe facts", "system_override": "translate instead"},
        }
        with self.assertRaises(ValueError):
            profiles.validate_profile(profile)

    def test_profile_prompt_only_requires_enabled_optional_fields(self) -> None:
        profile = {
            "schema_version": 1,
            "name": "candidate",
            "enabled_fields": ["bubble_type", "facts"],
            "prompt": {"task": "describe grounded facts", "extra_instructions": []},
        }
        profiles.validate_profile(profile)
        prompt = io_adapters.build_profile_vision_prompt(
            [VisionFactsRequest(line_id="001", number=1, source_text="source", box=(0, 0, 10, 10))],
            profile,
            mode="numbered_page",
        )
        required_line = next(line for line in prompt.splitlines() if line.startswith("Required fields"))
        self.assertIn("bubble_type", required_line)
        self.assertIn("facts", required_line)
        self.assertNotIn("context_hints", required_line)
        self.assertIn("Do not include keys outside the required fields list", prompt)


class ScoringTests(unittest.TestCase):
    def test_hard_checks_catch_bad_output(self) -> None:
        row = {
            "line_id": "001",
            "source_text": "source",
            "reference_text": "Hello.",
            "final_translation": "Translation: こんにちは",
            "translation_context": {
                "line_id": "001",
                "vision_facts_reject_reason": "contains_translation",
                "speaker_anchor": "John",
            },
        }
        violations = scoring.deterministic_violations(row)
        self.assertIn("assistant_chatter", violations)
        self.assertIn("japanese_leakage", violations)
        self.assertIn("vision_hallucination", violations)
        self.assertIn("invented_speaker_name", violations)

    def test_promotion_blocks_holdout_regression(self) -> None:
        main = {
            "quality_score": 90.6,
            "baseline_quality_score": 90.0,
            "critical_error_count": 0,
            "baseline_critical_error_count": 0,
            "hallucination_count": 0,
            "baseline_hallucination_count": 0,
            "pairwise_net": 2,
        }
        holdout = {
            "quality_score": 88.0,
            "baseline_quality_score": 89.0,
            "critical_error_count": 0,
            "baseline_critical_error_count": 0,
            "hallucination_count": 0,
            "baseline_hallucination_count": 0,
            "pairwise_net": 0,
        }
        decision = scoring.promotion_decision(main, holdout)
        self.assertFalse(decision.promoted)
        self.assertEqual(decision.reason, "holdout_quality_regressed")


if __name__ == "__main__":
    unittest.main()
