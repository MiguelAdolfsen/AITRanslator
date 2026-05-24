import unittest

from manga_local_translator.translation_evidence import (
    analyze_translation_evidence,
    build_consistency_memory,
    build_evidence_context,
    extract_source_features,
)


class TranslationEvidenceTests(unittest.TestCase):
    def test_source_features_extract_numbers_terms_honorifics_and_punctuation(self) -> None:
        features = extract_source_features("つばめ先輩は第２番街の〈ぴーつー〉へ行く？")

        self.assertIn("2", features["numbers"])
        self.assertIn("ぴーつー", features["bracket_terms"])
        self.assertIn("つばめ先輩", features["honorific_names"])
        self.assertIn("question", features["punctuation_intent"])

    def test_source_features_flag_repeated_and_credit_noise(self) -> None:
        repeated = extract_source_features("それから僕たちはそれから僕たちは")
        credit = extract_source_features("第６５話・帰還漫画・奥橋陸原作・柳野かなたキャラクター")

        self.assertIn("repeated_ocr", repeated["risk_flags"])
        self.assertIn("ocr_noise", credit["risk_flags"])

    def test_translation_evidence_detects_dropped_bracket_terms_and_broken_english(self) -> None:
        evidence = analyze_translation_evidence(
            "〈PII2〉へ行く？",
            "Are you going there?",
        )
        awkward = analyze_translation_evidence(
            "おまえはお風呂に付き合うはずだっただけじゃない。",
            "You weren't just supposed to catch a bath.",
        )

        self.assertIn("broken_english", awkward["preservation_failures"])
        self.assertIn("translation_evidence_failure", awkward["risk_flags"])
        self.assertIn("dropped_bracket_term", evidence["preservation_failures"])

    def test_translation_evidence_detects_noisy_source_invented_english_name(self) -> None:
        evidence = analyze_translation_evidence(
            "第６５話・帰還漫画・奥橋陸原作・柳野かなたキャラクター",
            ">> Vic Gundotra:",
        )

        self.assertIn("invented_english_name_on_noisy_source", evidence["preservation_failures"])

    def test_time_words_can_preserve_twelve(self) -> None:
        evidence = analyze_translation_evidence("あっもう１２時だ", "It's already midnight.")

        self.assertNotIn("dropped_number", evidence["preservation_failures"])

    def test_number_words_preserve_small_numbers(self) -> None:
        evidence = analyze_translation_evidence("３人同時に入れるの？", "Can three people enter at once?")

        self.assertNotIn("dropped_number", evidence["preservation_failures"])

    def test_consistency_memory_records_recurring_terms_and_reports_conflict(self) -> None:
        memory = build_consistency_memory(
            [
                ("マキは言った", "Maki said it."),
                ("マキを励まして", "Encourage Maki."),
                ("マキはもう相談しない", "Macha won't talk anymore."),
            ]
        )
        context = build_evidence_context("マキはもう相談しない", "Macha won't talk anymore.", memory)

        self.assertTrue(context["consistency_memory_entries"])
        self.assertIn("consistency_conflict", context["translation_evidence"]["preservation_failures"])


if __name__ == "__main__":
    unittest.main()
