from __future__ import annotations

import unittest

from manga_local_translator.vision_prompt import (
    build_vision_facts_prompt,
    build_vision_json_repair_prompt,
    build_vision_repair_prompt,
)
from manga_local_translator.vision_types import VisionFactsRequest, VisionRepairRequest
from manga_local_translator.vision_validation import (
    parse_vision_facts_response,
    parse_vision_repair_response,
    validate_vision_facts_results,
    validate_vision_results,
    vision_facts_acceptance_reason,
    vision_acceptance_reason,
)


class VisionPromptValidationTests(unittest.TestCase):
    def test_facts_prompt_forbids_translation(self) -> None:
        request = VisionFactsRequest("p1_001", 1, "\u306f\u3044", (1, 2, 3, 4))

        prompt = build_vision_facts_prompt([request], mode="numbered_page")

        self.assertTrue(prompt.startswith("/no_think"))
        self.assertIn("Do not translate", prompt)
        self.assertIn("Do not OCR Japanese from the image", prompt)
        self.assertIn("bubble_type", prompt)
        self.assertIn("speaker_position", prompt)
        self.assertIn("visible_emotion", prompt)
        self.assertIn("observable_action", prompt)
        self.assertIn('"line_id":"p1_001"', prompt)
        self.assertNotIn("current_translation", prompt)

    def test_facts_parser_accepts_valid_visual_facts(self) -> None:
        request = VisionFactsRequest("p1_001", 1, "\u306f\u3044", (0, 0, 10, 10))
        raw = (
            '{"lines":[{"line_id":"p1_001","number":1,"source_text":"\\u306f\\u3044",'
            '"bubble_type":"speech","speaker_position":"girl on right",'
            '"visible_emotion":"surprised","observable_action":"looking left",'
            '"mapping_confidence":"high","facts":["speech bubble near girl on right"],'
            '"needs_review":false,"risk_flags":[],"warnings":[]}]}'
        )

        result = parse_vision_facts_response(raw)[0]
        accepted, errors = validate_vision_facts_results([request], [result])

        self.assertFalse(errors)
        self.assertIn("p1_001", accepted)
        self.assertIsNone(vision_facts_acceptance_reason(request, result))

    def test_facts_validation_rejects_bad_mapping(self) -> None:
        requests = [
            VisionFactsRequest("p1_001", 1, "\u306f\u3044", (0, 0, 10, 10)),
            VisionFactsRequest("p1_002", 2, "\u3044\u3044\u3048", (0, 0, 10, 10)),
        ]
        raw = (
            '{"lines":['
            '{"line_id":"p1_001","number":1,"source_text":"\\u306f\\u3044","mapping_confidence":"high","facts":[]},'
            '{"line_id":"p1_001","number":1,"source_text":"\\u306f\\u3044","mapping_confidence":"high","facts":[]},'
            '{"line_id":"p1_002","number":9,"source_text":"\\u3044\\u3044\\u3048","mapping_confidence":"high","facts":[]},'
            '{"line_id":"p1_999","number":99,"source_text":"x","mapping_confidence":"high","facts":[]}'
            ']}'
        )

        accepted, errors = validate_vision_facts_results(requests, parse_vision_facts_response(raw))

        self.assertIn("p1_001", accepted)
        self.assertEqual(errors["p1_001"], "duplicate_line_id")
        self.assertEqual(errors["p1_002"], "wrong_number")
        self.assertEqual(errors["unknown_line_id:p1_999"], "unknown_line_id")

    def test_facts_acceptance_rejects_translation_and_invented_speaker(self) -> None:
        request = VisionFactsRequest("p1_001", 1, "\u306f\u3044", (0, 0, 10, 10))
        translated = parse_vision_facts_response(
            '{"lines":[{"line_id":"p1_001","number":1,"source_text":"\\u306f\\u3044",'
            '"translation":"Yes.","mapping_confidence":"high","facts":["Yes, I understand."]}]}'
        )[0]
        invented_speaker = parse_vision_facts_response(
            '{"lines":[{"line_id":"p1_001","number":1,"source_text":"\\u306f\\u3044",'
            '"speaker_position":"Anya","mapping_confidence":"high","facts":["small bubble near the character"]}]}'
        )[0]

        self.assertEqual(vision_facts_acceptance_reason(request, translated), "contains_translation_field")
        self.assertEqual(vision_facts_acceptance_reason(request, invented_speaker), "ungrounded_speaker")

    def test_facts_acceptance_allows_position_grounded_speaker_descriptions(self) -> None:
        request = VisionFactsRequest("p1_001", 1, "\u306f\u3044", (0, 0, 10, 10))
        positioned_speaker = parse_vision_facts_response(
            '{"lines":[{"line_id":"p1_001","number":1,"source_text":"\\u306f\\u3044",'
            '"speaker_position":"girl on right","mapping_confidence":"high",'
            '"facts":["small bubble near the character"]}]}'
        )[0]

        self.assertIsNone(vision_facts_acceptance_reason(request, positioned_speaker))

    def test_facts_acceptance_rejects_text_content_descriptions(self) -> None:
        request = VisionFactsRequest("p1_001", 1, "\u4f55\u3042\u308c", (0, 0, 10, 10))
        result = parse_vision_facts_response(
            '{"lines":[{"line_id":"p1_001","number":1,"source_text":"\\u4f55\\u3042\\u308c",'
            '"mapping_confidence":"high","facts":["speech bubble with text what is that present"]}]}'
        )[0]

        self.assertEqual(vision_facts_acceptance_reason(request, result), "invalid_visual_fact")

    def test_prompt_includes_flagged_line_fields(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "母", "Mom", "qwen_rejected", (1, 2, 3, 4))

        prompt = build_vision_repair_prompt([request], mode="numbered_page")

        self.assertTrue(prompt.startswith("/no_think"))
        self.assertIn("Do not OCR Japanese from the image", prompt)
        self.assertIn("Anti-hallucination rules", prompt)
        self.assertIn("Do not add dialogue", prompt)
        self.assertIn("source_text", prompt)
        self.assertIn("speaker_confidence", prompt)
        self.assertIn("visual_evidence_type", prompt)
        self.assertIn("needs_review", prompt)
        self.assertIn("risk_flags", prompt)
        self.assertIn('Output wrapper must be exactly: {"lines":[...]}', prompt)
        self.assertNotIn('"type":"object"', prompt)
        self.assertIn('"line_id":"p1_001"', prompt)
        self.assertIn('"current_translation":"Mom"', prompt)
        self.assertIn('"issue":"qwen_rejected"', prompt)

    def test_json_repair_prompt_preserves_mapping_and_forbids_reinterpretation(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "æ¯", "Mom", "qwen_rejected", (1, 2, 3, 4))

        prompt = build_vision_json_repair_prompt([request], '{"broken":')

        self.assertTrue(prompt.startswith("/no_think"))
        self.assertIn("JSON repair task only", prompt)
        self.assertIn("do not invent translations", prompt)
        self.assertIn("Copy source_text exactly", prompt)
        self.assertIn('Output wrapper must be exactly: {"lines":[...]}', prompt)
        self.assertNotIn('"type":"object"', prompt)
        self.assertIn('"line_id":"p1_001"', prompt)
        self.assertIn('"number":1', prompt)
        self.assertIn('"current_translation":"Mom"', prompt)
        self.assertIn('{"broken":', prompt)

    def test_parser_reads_grounding_fields(self) -> None:
        raw = (
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace",'
            '"translation":"Mom","source_text":"\\u6bcd","speaker":"narrator",'
            '"speaker_confidence":"high","situation":"caption",'
            '"visual_evidence_type":"direct","visual_evidence":"caption is at the panel edge",'
            '"confidence":"high","needs_review":false,"risk_flags":[],"warnings":[]}]}'
        )

        result = parse_vision_repair_response(raw)[0]

        self.assertEqual(result.source_text, "\u6bcd")
        self.assertEqual(result.speaker_confidence, "high")
        self.assertEqual(result.visual_evidence_type, "direct")
        self.assertFalse(result.needs_review)
        self.assertEqual(result.risk_flags, ())

    def test_parser_and_validator_reject_missing_duplicate_and_wrong_number(self) -> None:
        requests = [
            VisionRepairRequest("p1_001", 1, "母", "Mom", "issue", (0, 0, 10, 10)),
            VisionRepairRequest("p1_002", 2, "父", "Dad", "issue", (0, 0, 10, 10)),
        ]
        raw = (
            '{"lines":['
            '{"line_id":"p1_001","number":1,"action":"replace","translation":"Mother",'
            '"speaker":null,"situation":null,"visual_evidence":"adult woman shown","confidence":"high","warnings":[]},'
            '{"line_id":"p1_001","number":1,"action":"replace","translation":"Mom",'
            '"speaker":null,"situation":null,"visual_evidence":"duplicate","confidence":"high","warnings":[]},'
            '{"line_id":"p1_002","number":9,"action":"replace","translation":"Father",'
            '"speaker":null,"situation":null,"visual_evidence":"adult man shown","confidence":"high","warnings":[]}'
            ']}'
        )

        results = parse_vision_repair_response(raw)
        accepted, errors = validate_vision_results(requests, results)

        self.assertEqual(errors["p1_001"], "duplicate_line_id")
        self.assertEqual(errors["p1_002"], "wrong_number")
        self.assertIn("p1_001", accepted)

    def test_validator_reports_unknown_extra_line_id(self) -> None:
        requests = [VisionRepairRequest("p1_001", 1, "\u6bcd", "Mom", "issue", (0, 0, 10, 10))]
        raw = (
            '{"lines":['
            '{"line_id":"p1_001","number":1,"action":"replace","translation":"Mother",'
            '"source_text":"\\u6bcd","speaker":null,"situation":null,'
            '"visual_evidence":"adult woman shown","confidence":"high","warnings":[]},'
            '{"line_id":"p1_999","number":99,"action":"replace","translation":"Extra line",'
            '"source_text":"\\u306a\\u3057","speaker":null,"situation":null,'
            '"visual_evidence":"invented extra output","confidence":"high","warnings":[]}'
            ']}'
        )

        results = parse_vision_repair_response(raw)
        _, errors = validate_vision_results(requests, results)

        self.assertEqual(errors["unknown_line_id:p1_999"], "unknown_line_id")

    def test_parser_recovers_direct_line_object_from_malformed_outer_json(self) -> None:
        raw = (
            '<think></think>{"lines":[BROKEN {"line_id":"p1_001","number":1,"action":"replace",'
            '"translation":"Mom","speaker":null,"situation":null,"visual_evidence":"mother shown",'
            '"confidence":"high","warnings":[]}]}'
        )

        results = parse_vision_repair_response(raw)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].line_id, "p1_001")
        self.assertEqual(results[0].translation, "Mom")

    def test_acceptance_keeps_text_when_vision_is_invalid(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "母", "Mom", "issue", (0, 0, 10, 10))
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace","translation":"母",'
            '"speaker":null,"situation":null,"visual_evidence":"adult woman shown","confidence":"high","warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, result), "contains_japanese")

    def test_acceptance_allows_longer_repair_for_failed_qwen_block(self) -> None:
        request = VisionRepairRequest(
            "p1_001",
            1,
            "その後お菓子を買ってもらったアーニャは機嫌を直し",
            "(Laughter)",
            "qwen_rejected,qwen_repair_failed",
            (0, 0, 10, 10),
        )
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace",'
            '"translation":"After that, Anya bought some sweets and cheered up.",'
            '"speaker":"narrator","situation":"Anya is smiling",'
            '"visual_evidence":"Anya appears cheerful after getting sweets.",'
            '"confidence":"high","warnings":[]}]}'
        )[0]

        self.assertIsNone(vision_acceptance_reason(request, result))

    def test_acceptance_rejects_vague_visual_evidence(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "æ¯", "Mom", "qwen_rejected", (0, 0, 10, 10))
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace","translation":"Mother",'
            '"speaker":null,"situation":null,"visual_evidence":"the image supports this",'
            '"confidence":"high","warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, result), "invalid_visual_evidence")

    def test_acceptance_rejects_source_text_mismatch(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "\u6bcd", "Mom", "qwen_rejected", (0, 0, 10, 10))
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace","source_text":"\\u7236",'
            '"translation":"Mother","speaker":null,"situation":null,'
            '"visual_evidence_type":"direct","visual_evidence":"adult woman is next to the label",'
            '"confidence":"high","warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, result), "source_text_mismatch")

    def test_acceptance_rejects_needs_review_and_risk_flags(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "\u6bcd", "Mom", "qwen_rejected", (0, 0, 10, 10))
        needs_review = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace","source_text":"\\u6bcd",'
            '"translation":"Mother","speaker":null,"situation":null,'
            '"visual_evidence_type":"direct","visual_evidence":"adult woman is next to the label",'
            '"confidence":"high","needs_review":true,"risk_flags":[],"warnings":[]}]}'
        )[0]
        risk_flagged = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace","source_text":"\\u6bcd",'
            '"translation":"Mother","speaker":null,"situation":null,'
            '"visual_evidence_type":"direct","visual_evidence":"adult woman is next to the label",'
            '"confidence":"high","needs_review":false,"risk_flags":["uncertain_mapping"],"warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, needs_review), "needs_review")
        self.assertEqual(vision_acceptance_reason(request, risk_flagged), "risk_flags:uncertain_mapping")

    def test_acceptance_rejects_evidence_type_mismatch(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "\u6bcd", "Mom", "qwen_rejected", (0, 0, 10, 10))
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace","source_text":"\\u6bcd",'
            '"translation":"Mother","speaker":null,"situation":null,'
            '"visual_evidence_type":"none","visual_evidence":"adult woman is next to the label",'
            '"confidence":"high","warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, result), "visual_evidence_type_mismatch")

    def test_acceptance_rejects_low_confidence_specific_speaker(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "\u306f\u3044", "Yes.", "qwen_rejected", (0, 0, 10, 10))
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace","source_text":"\\u306f\\u3044",'
            '"translation":"Yes.","speaker":"Anya","speaker_confidence":"low",'
            '"situation":"answering","visual_evidence_type":"direct",'
            '"visual_evidence":"one character is near the bubble",'
            '"confidence":"high","warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, result), "ungrounded_speaker")

    def test_acceptance_rejects_unsupported_added_detail(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "ãã®ã‚ã¨ã©ã†ã—ãŸã®", "What happened after that?", "qwen_rejected", (0, 0, 10, 10))
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace",'
            '"translation":"Yeah, I found the stolen treasure map inside the castle.",'
            '"speaker":"girl","situation":"girl answers",'
            '"visual_evidence":"girl is answering nearby","confidence":"high","warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, result), "unsupported_detail")


    def test_acceptance_rejects_invented_proper_noun_from_plain_hiragana(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "\u3088\u3057\u3064\u304e\u3078\u3044\u304f\u305e\uff01", "Alright, let's go!", "layout:excessive_line_count", (0, 0, 10, 10))
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace",'
            '"translation":"Alright, let\\u0027s go to Yoshitsugi!",'
            '"speaker":"leader","situation":"moving on",'
            '"visual_evidence":"Yoshitsugi is likely a proper noun or place name.",'
            '"confidence":"high","warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, result), "speculative_visual_evidence")

    def test_acceptance_rejects_romanized_short_sound_effect_drift(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "\u304f\u3063", "Ngh.", "layout:excessive_line_count", (0, 0, 10, 10))
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace",'
            '"translation":"Kuchuuu!",'
            '"speaker":"girl","situation":"small sound",'
            '"visual_evidence":"The bubble is small and near the character.",'
            '"confidence":"high","warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, result), "romanized_short_source")

    def test_acceptance_rejects_layout_only_semantic_drift(self) -> None:
        request = VisionRepairRequest("p1_001", 1, "\u306f\u3093\u3076\u3093\u3053\u3057\u305f\u3074\u30fc\u306a\u3064\u3092\u304a\u305f\u304c\u3044\u305f\u3079\u308b", "If you do that, you will be accepted into my organization.", "layout:excessive_line_count", (0, 0, 10, 10))
        result = parse_vision_repair_response(
            '{"lines":[{"line_id":"p1_001","number":1,"action":"replace",'
            '"translation":"If you do that, you can see my organization for what it is.",'
            '"speaker":"girl","situation":"explaining a ritual",'
            '"visual_evidence":"A girl is explaining something to another character.",'
            '"confidence":"high","warnings":[]}]}'
        )[0]

        self.assertEqual(vision_acceptance_reason(request, result), "layout_semantic_drift")


if __name__ == "__main__":
    unittest.main()
