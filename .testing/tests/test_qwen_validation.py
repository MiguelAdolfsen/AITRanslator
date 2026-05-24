from __future__ import annotations

import unittest

from manga_local_translator.qwen_prompts import build_qwen_critic_prompt, build_qwen_translation_prompt, build_qwen_verification_prompt
from manga_local_translator.qwen_types import QWEN_REPAIR_SETTINGS, QWEN_TRANSLATION_SETTINGS
from manga_local_translator.qwen_validation import (
    accept_qwen_translation,
    parse_qwen_page_translations,
    parse_qwen_critic,
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

    def test_qwen_critic_prompt_forbids_rewriting(self) -> None:
        prompt = build_qwen_critic_prompt(
            "\u7236\u3082\u6bcd\u3082\u304d\u3089\u3044",
            before=None,
            after=None,
            baseline="I hate Father and Mother.",
            current_translation="Father and Mother are hated!!",
            trigger_reasons=("awkward_english_pattern",),
        )

        self.assertIn("Do not translate or rewrite", prompt)
        self.assertIn("Return issue labels only", prompt)
        self.assertIn("awkward_literal", prompt)
        self.assertIn("Father and Mother are hated!!", prompt)

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

    def test_parse_qwen_critic_accepts_clean_json(self) -> None:
        raw = '{"ok":false,"severity":"medium","issues":["awkward_literal"],"reason":"passive wording is unnatural"}'

        decision = parse_qwen_critic(raw)

        self.assertFalse(decision.ok)
        self.assertEqual(decision.severity, "medium")
        self.assertEqual(decision.issues, ("awkward_literal",))
        self.assertEqual(decision.reason, "passive wording is unnatural")

    def test_parse_qwen_critic_allows_low_severity_style_issue(self) -> None:
        raw = '{"ok":true,"severity":"low","issues":["awkward_literal"],"reason":"slightly literal"}'

        decision = parse_qwen_critic(raw)

        self.assertTrue(decision.ok)
        self.assertEqual(decision.severity, "low")
        self.assertEqual(decision.issues, ("awkward_literal",))

    def test_parse_qwen_critic_rejects_schema_talk(self) -> None:
        raw = '{"ok":true,"severity":"none","issues":[],"reason":"JSON schema is valid"}'

        decision = parse_qwen_critic(raw)

        self.assertFalse(decision.ok)
        self.assertIn("critic_schema_talk", decision.issues)

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

    def test_candidate_rejects_common_manga_drift_patterns(self) -> None:
        cases = [
            ("\u7d44\u7e54", "my organization", "watch my organization for what it is", "organization_membership_drift"),
            ("\u7acb\u6d3e\u306a\u5b50\u5206", "fine subordinates", "a good son of a bitch", "kobun_offensive_mistranslation"),
            ("\u30a2\u30fc\u30cb\u30e3\u3055\u3093", "Anya san", "Anja san", "anya_name_drift"),
            ("\u7236\u3082\u6bcd\u3082\u304d\u3089\u3044", "I hate my father and mother.", "I hate my mother!", "dropped_father_or_mother"),
            ("\u3072\u307f\u3064\u305d\u3057\u304d\u3008\u3074\u30fc\u3064\u30fc\u3009\u306e\u30dc\u30b9", "boss of P2", "I'm the boss of the secret organization.", "dropped_bracket_term"),
        ]

        for source, baseline, candidate, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(
                    accept_qwen_translation(source_text=source, baseline=baseline, candidate=candidate),
                    (False, reason),
                )

    def test_candidate_allows_unknown_or_noisy_bracket_terms_without_hard_reject(self) -> None:
        cases = [
            ("\u3008\u30a2\u30fc\u30cb\u3055\u3009\u306e\u90e8\u5c4b", "It's Anisama's room."),
            ("\u3072\u307f\u3064\u305d\u3057\u304d\u3008unknown\u3009\u306e\u30dc\u30b9", "I'm the boss of the secret organization."),
            ("\u3053\u3044\u3064\u306f\u30a8\u30fc\u30b8\u30a7\u30f3\u30c8\u3008\u306f\u306f\u3009", "This agent is strong."),
        ]

        for source, candidate in cases:
            with self.subTest(source=source):
                accepted, reason = accept_qwen_translation(source_text=source, baseline=candidate, candidate=candidate)
                self.assertTrue(accepted, reason)


if __name__ == "__main__":
    unittest.main()
