from __future__ import annotations

import json

from .vision_types import VisionFactsRequest, VisionRepairRequest


VISION_ANTI_HALLUCINATION_RULES = [
    "Anti-hallucination rules:",
    "- Translate only the provided source_text fields.",
    "- Do not add dialogue, narration, sound effects, thoughts, names, facts, locations, relationships, motives, or backstory.",
    "- Do not invent a speaker name, gender, age, or relationship.",
    "- Do not use the image to correct Japanese OCR.",
    "- The image is visual context only; never treat it as the source of text.",
    "- If uncertain, become less specific, not more creative.",
    "- visual_evidence must mention only observable facts from the image.",
    "- If no observable visual fact helped, use visual_evidence_type=\"none\" and visual_evidence=\"not visually informed\".",
    "- Set needs_review=true and add risk_flags when mapping, OCR, speaker, or visual evidence is uncertain.",
]


VISION_REQUIRED_FIELDS = (
    "line_id",
    "number",
    "action",
    "translation",
    "source_text",
    "speaker",
    "speaker_confidence",
    "situation",
    "visual_evidence_type",
    "visual_evidence",
    "confidence",
    "needs_review",
    "risk_flags",
    "warnings",
)

VISION_FACTS_REQUIRED_FIELDS = (
    "line_id",
    "number",
    "source_text",
    "bubble_type",
    "line_role",
    "speaker_position",
    "speaker_anchor",
    "visible_emotion",
    "tone_hint",
    "observable_action",
    "mapping_confidence",
    "facts",
    "context_hints",
    "needs_review",
    "risk_flags",
    "warnings",
)


def build_vision_facts_prompt(requests: list[VisionFactsRequest], *, mode: str) -> str:
    payload = [
        {
            "line_id": request.line_id,
            "number": request.number,
            "source_text": request.source_text,
        }
        for request in requests
    ]
    lines = [
        "/no_think",
        "You are a manga visual-context extractor.",
        "Task: describe only grounded visual facts for the numbered OCR lines listed below.",
        "Do not translate. Do not rewrite source_text. Do not OCR Japanese from the image.",
        "The provided source_text is authoritative and is included only for ID matching.",
        "Use the image only for bubble type, line role, speaker position/anchor, visible emotion, observable action, and immediate scene context.",
        "Do not describe, paraphrase, quote, or translate the text printed in the bubble.",
        "Do not invent names, relationships, motives, backstory, locations, dialogue, or narration.",
        "If a fact is uncertain, use unknown, set needs_review=true, and add a risk flag.",
        "Return JSON only. Do not use markdown fences. Do not include thinking text.",
        "For each input line, return exactly one object with the same line_id, number, and source_text copied exactly.",
        f"Required fields in every output line object: {', '.join(VISION_FACTS_REQUIRED_FIELDS)}.",
        "Allowed bubble_type values: speech, thought, narration, sound_effect, sign, unknown.",
        "Allowed mapping_confidence values: low, medium, high.",
        "Allowed line_role values: speech, thought, narration, sound_effect, sign, metadata, unknown.",
        "speaker_anchor must be a visual anchor only, such as character left of bubble, narrator, or unknown. Do not invent names.",
        "tone_hint must be visible tone only, such as angry, surprised, calm, crying, shouting, or unknown.",
        "The facts array must contain short visual facts only, not English translations or descriptions of bubble text.",
        "The context_hints array may contain short conservative hints for pronouns, tone, or line role only.",
        "Output wrapper must be exactly: {\"lines\":[...]}",
        "Example output shape: {\"lines\":[{\"line_id\":\"example_001\",\"number\":1,\"source_text\":\"日本語\",\"bubble_type\":\"speech\",\"line_role\":\"speech\",\"speaker_position\":\"right side\",\"speaker_anchor\":\"character right of bubble\",\"visible_emotion\":\"surprised\",\"tone_hint\":\"surprised\",\"observable_action\":\"looking left\",\"mapping_confidence\":\"medium\",\"facts\":[\"speech bubble near character on right\"],\"context_hints\":[\"surprised tone\"],\"needs_review\":false,\"risk_flags\":[],\"warnings\":[]}]}",
        f"Vision mode: {mode}",
        "Lines to describe:",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "JSON:",
    ]
    return "\n".join(lines)


def build_vision_facts_json_repair_prompt(requests: list[VisionFactsRequest], invalid_output: str) -> str:
    expected = [
        {
            "line_id": request.line_id,
            "number": request.number,
            "source_text": request.source_text,
        }
        for request in requests
    ]
    lines = [
        "/no_think",
        "Repair the following model output so it exactly matches the required JSON shape for visual facts.",
        "This is a JSON repair task only. Do not translate, reinterpret the manga, or add new facts.",
        "Return exactly one object per expected line_id and number. Copy source_text exactly. Do not add new lines. Do not remove expected lines.",
        "Return JSON only. Do not include markdown fences, explanations, or thinking text.",
        f"Required fields in every output line object: {', '.join(VISION_FACTS_REQUIRED_FIELDS)}.",
        "Output wrapper must be exactly: {\"lines\":[...]}",
        "Expected line IDs and numbers:",
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")),
        "Invalid output:",
        invalid_output,
        "JSON:",
    ]
    return "\n".join(lines)


def build_vision_repair_prompt(requests: list[VisionRepairRequest], *, mode: str) -> str:
    payload = [
        {
            "line_id": request.line_id,
            "number": request.number,
            "source_text": request.source_text,
            "current_translation": request.current_translation,
            "issue": request.issue,
        }
        for request in requests
    ]
    lines = [
        "/no_think",
        "You are a professional manga translation QA assistant.",
        "Task: review only the numbered lines listed below.",
        *VISION_ANTI_HALLUCINATION_RULES,
        "The provided Japanese OCR/source text is authoritative. Do not OCR Japanese from the image.",
        "Use the page image only for speaker, emotion, scene, objects, bubble location, and whether the current English is visually plausible.",
        "Return JSON only. Do not use markdown fences. Do not include thinking text.",
        "For each input line, return exactly one object with the same line_id, number, and source_text copied exactly.",
        "Use action=\"replace\" only when the image clearly supports a better concise English translation.",
        "Use action=\"keep\" when the current translation is acceptable or the image is uncertain.",
        "Keep translations short enough for manga bubbles.",
        "Preserve Japanese honorifics as romanized suffixes like san, sama, chan, kun, senpai, sensei.",
        f"Required fields in every output line object: {', '.join(VISION_REQUIRED_FIELDS)}.",
        "Allowed action values: replace, keep.",
        "Allowed speaker_confidence values: low, medium, high.",
        "Allowed visual_evidence_type values: none, weak, direct.",
        "Output wrapper must be exactly: {\"lines\":[...]}",
        "Example output shape: {\"lines\":[{\"line_id\":\"example_001\",\"number\":1,\"action\":\"keep\",\"translation\":\"Current English.\",\"source_text\":\"日本語\",\"speaker\":\"unknown\",\"speaker_confidence\":\"low\",\"situation\":null,\"visual_evidence_type\":\"none\",\"visual_evidence\":\"not visually informed\",\"confidence\":\"medium\",\"needs_review\":false,\"risk_flags\":[],\"warnings\":[]}]}",
        f"Vision mode: {mode}",
        "Lines to review:",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "JSON:",
    ]
    return "\n".join(lines)


def build_vision_json_repair_prompt(requests: list[VisionRepairRequest], invalid_output: str) -> str:
    expected = [
        {
            "line_id": request.line_id,
            "number": request.number,
            "source_text": request.source_text,
            "current_translation": request.current_translation,
            "issue": request.issue,
        }
        for request in requests
    ]
    lines = [
        "/no_think",
        "Repair the following model output so it exactly matches the required JSON shape for a manga translation pipeline.",
        "This is a JSON repair task only. Do not reinterpret the manga, do not add visual details, and do not invent translations.",
        "Preserve the original translation wording whenever possible. Change wording only when required to restore valid JSON escaping.",
        "Return exactly one object per expected line_id and number. Copy source_text exactly. Do not add new lines. Do not remove expected lines.",
        "Return JSON only. Do not include markdown fences, explanations, or thinking text.",
        f"Required fields in every output line object: {', '.join(VISION_REQUIRED_FIELDS)}.",
        "Output wrapper must be exactly: {\"lines\":[...]}",
        "Example output shape: {\"lines\":[{\"line_id\":\"example_001\",\"number\":1,\"action\":\"keep\",\"translation\":\"Current English.\",\"source_text\":\"日本語\",\"speaker\":\"unknown\",\"speaker_confidence\":\"low\",\"situation\":null,\"visual_evidence_type\":\"none\",\"visual_evidence\":\"not visually informed\",\"confidence\":\"medium\",\"needs_review\":false,\"risk_flags\":[],\"warnings\":[]}]}",
        "Expected line IDs and numbers:",
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")),
        "Invalid output:",
        invalid_output,
        "JSON:",
    ]
    return "\n".join(lines)
