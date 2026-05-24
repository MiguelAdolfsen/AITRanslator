from __future__ import annotations

import unittest

from manga_local_translator.qwen_prompts import build_qwen_translation_prompt, build_qwen_verification_prompt
from manga_local_translator.qwen_types import QWEN_REPAIR_SETTINGS, QWEN_TRANSLATION_SETTINGS
from manga_local_translator.qwen_validation import (
    accept_qwen_translation,
    parse_qwen_page_translations,
    parse_qwen_translation,
    parse_qwen_verification,
    verifier_reason_says_reject,
)


class QwenValidationTests(unittest.TestCase):
    def test_qwen_prompts_include_anti_hallucination_contract(self) -> None:
        translation_prompt = build_qwen_translation_prompt(
            "\u306f\u3044",
            before=None,
            after=None,
            baseline="Yes.",
        )
        verification_prompt = build_qwen_verification_prompt(
            "\u306f\u3044",
            before=None,
            after=None,
            baseline="Yes.",
            candidate="Yes, father.",
        )

        for prompt in (translation_prompt, verification_prompt):
            with self.subTest(prompt=prompt.splitlines()[1]):
                self.assertIn("Anti-hallucination rules", prompt)
                self.assertIn("Do not add dialogue", prompt)
                self.assertIn("If uncertain, become less specific", prompt)

    def test_qwen_prompt_includes_visual_facts_as_context_only(self) -> None:
        prompt = build_qwen_translation_prompt(
            "\u306f\u3044",
            before=None,
            after=None,
            baseline="Yes.",
            visual_facts=("Speaker position: girl on right", "Visible emotion: surprised"),
        )

        self.assertIn("Visual facts from the page image", prompt)
        self.assertIn("context only", prompt)
        self.assertIn("target/source text remains authoritative", prompt)
        self.assertIn("Speaker position: girl on right", prompt)

    def test_qwen_default_settings_are_conservative(self) -> None:
        self.assertLessEqual(QWEN_TRANSLATION_SETTINGS.temperature, 0.2)
        self.assertLessEqual(QWEN_REPAIR_SETTINGS.temperature, 0.4)
        self.assertLessEqual(QWEN_REPAIR_SETTINGS.top_p, 0.9)

    def test_parse_translation_from_json_and_wrappers(self) -> None:
        raw = '<think>draft</think>\n```json\n{"translation":"Hello there."}\n```'

        self.assertEqual(parse_qwen_translation(raw), "Hello there.")

    def test_parse_verification_prefers_correction_payload(self) -> None:
        raw = (
            '{"choice":"baseline","ok":false,"reason":"candidate adds detail"}\n'
            '{"choice":"correction","ok":true,"translation":"Look out, senpai!",'
            '"unsupported_terms":[],"reason":"target text decides"}'
        )

        decision = parse_qwen_verification(raw)

        self.assertTrue(decision.ok)
        self.assertEqual(decision.choice, "correction")
        self.assertEqual(decision.translation, "Look out, senpai!")

    def test_parse_page_translations_from_json(self) -> None:
        raw = (
            '```json\n{"translations":['
            '{"id":1,"translation":"Hello there.","confidence":0.8,"reason":"target"},'
            '{"id":2,"translation":"Wait!","confidence":"0.6","reason":"context"}'
            ']}\n```'
        )

        parsed = parse_qwen_page_translations(raw)

        self.assertEqual([item.page_id for item in parsed], [1, 2])
        self.assertEqual(parsed[0].translation, "Hello there.")
        self.assertEqual(parsed[1].confidence, 0.6)

    def test_parse_page_translations_ignores_malformed_items(self) -> None:
        raw = '{"translations":[{"id":"bad","translation":"Nope"},{"id":3,"translation":"Safe."}]}'

        parsed = parse_qwen_page_translations(raw)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].page_id, 3)

    def test_parse_page_translations_recovers_bare_array_items(self) -> None:
        raw = (
            '[{"id":1,"translation":"One.","confidence":0.8,"reason":"target"},'
            '{"id":2,"translation":"Two.","confidence":0.7,"reason":"target"},'
            '{"id":3,"translation":"'
        )

        parsed = parse_qwen_page_translations(raw)

        self.assertEqual([item.page_id for item in parsed], [1, 2])

    def test_parse_page_translations_chooses_best_recovered_sequence(self) -> None:
        raw = (
            '[{"id":1,"translation":"Bad first."},{"id":3,"translation":"Bad third."}'
            'JSON:[{"id":1,"translation":"Good first."},{"id":3,"translation":"Good third."},'
            '{"id":4,"translation":"Good fourth."}]'
        )

        parsed = parse_qwen_page_translations(raw)

        self.assertEqual([item.page_id for item in parsed], [1, 3, 4])
        self.assertEqual(parsed[0].translation, "Good first.")

    def test_candidate_rejects_unsupported_or_drifting_output(self) -> None:
        cases = [
            ("こんにちは", "Hello.", "こんにちは", "contains_japanese"),
            ("くっ", "Ngh.", "This means he is probably angry.", "hedging_or_explanation"),
            ("罠だ", "It's a trap.", "the second-way candy store, san", "known_hallucination_artifact"),
            ("\u304a\u307e\u3048\u3089", "You guys.", "Come on, boys!", "unnecessary_gendering"),
            ("\u305d\u306e\u5f8c\u304a\u83d3\u5b50\u3092\u8cb7\u3063\u3066\u3082\u3089\u3063\u305f", "Afterward, she bought sweets.", "(Laughter)", "stage_direction_only"),
            ("あ", "Ah.", "one-two-three-four-five-six", "malformed_hyphen_chain"),
        ]

        for source, baseline, candidate, reason in cases:
            with self.subTest(candidate=candidate):
                self.assertEqual(
                    accept_qwen_translation(source_text=source, baseline=baseline, candidate=candidate),
                    (False, reason),
                )

    def test_verifier_reject_reason_filter_ignores_safe_phrasing(self) -> None:
        self.assertFalse(verifier_reason_says_reject("candidate is direct and no unsupported details"))
        self.assertTrue(verifier_reason_says_reject("adds unsupported location not in Japanese"))


if __name__ == "__main__":
    unittest.main()
