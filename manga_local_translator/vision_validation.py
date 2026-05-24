from __future__ import annotations

import json
import re

from .qwen_validation import accept_qwen_translation, clean_qwen_output, extract_qwen_json_payloads
from .render import render_text_length
from .text_filter import suspected_bad_translation
from .vision_types import VisionFactsRequest, VisionFactsResult, VisionRepairRequest, VisionRepairResult


def parse_vision_repair_response(text: str) -> list[VisionRepairResult]:
    best: list[VisionRepairResult] = []
    for payload in extract_qwen_json_payloads(text):
        lines = payload.get("lines")
        if not isinstance(lines, list):
            if "line_id" in payload and "number" in payload:
                lines = [payload]
            else:
                continue
        parsed = [result for result in (vision_result_from_payload(item) for item in lines if isinstance(item, dict)) if result]
        if len(parsed) > len(best):
            best = parsed
    return best


def parse_vision_facts_response(text: str) -> list[VisionFactsResult]:
    best: list[VisionFactsResult] = []
    for payload in extract_qwen_json_payloads(text):
        lines = payload.get("lines")
        if not isinstance(lines, list):
            if "line_id" in payload and "number" in payload:
                lines = [payload]
            else:
                continue
        parsed = [result for result in (vision_facts_from_payload(item) for item in lines if isinstance(item, dict)) if result]
        if len(parsed) > len(best):
            best = parsed
    return best


def vision_facts_from_payload(payload: dict[str, object]) -> VisionFactsResult | None:
    try:
        number = int(payload.get("number", 0))
    except (TypeError, ValueError):
        return None
    line_id = str(payload.get("line_id", "") or "").strip()
    if not line_id or not number:
        return None
    facts = payload.get("facts", [])
    if isinstance(facts, str):
        facts = [facts]
    if not isinstance(facts, list):
        facts = []
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    risk_flags = payload.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    return VisionFactsResult(
        line_id=line_id,
        number=number,
        source_text=clean_qwen_output(str(payload.get("source_text", "") or "")),
        bubble_type=enum_string(payload.get("bubble_type"), {"speech", "thought", "narration", "sound_effect", "sign", "unknown"}, "unknown"),
        speaker_position=clean_qwen_output(str(payload.get("speaker_position", "") or "unknown")),
        visible_emotion=clean_qwen_output(str(payload.get("visible_emotion", "") or "unknown")),
        observable_action=clean_qwen_output(str(payload.get("observable_action", "") or "unknown")),
        mapping_confidence=enum_string(payload.get("mapping_confidence"), {"low", "medium", "high"}, "medium"),
        facts=tuple(clean_qwen_output(str(value)) for value in facts if str(value).strip()),
        needs_review=bool(payload.get("needs_review", False)),
        risk_flags=tuple(clean_qwen_output(str(value)) for value in risk_flags if str(value).strip()),
        warnings=tuple(clean_qwen_output(str(value)) for value in warnings if str(value).strip()),
        raw=payload,
    )


def vision_result_from_payload(payload: dict[str, object]) -> VisionRepairResult | None:
    try:
        number = int(payload.get("number", 0))
    except (TypeError, ValueError):
        return None
    line_id = str(payload.get("line_id", "") or "").strip()
    action = str(payload.get("action", "") or "").strip().lower()
    if action not in {"replace", "keep"}:
        action = "replace" if clean_qwen_output(str(payload.get("translation", "") or "")) else "keep"
    if not line_id or not number:
        return None
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    return VisionRepairResult(
        line_id=line_id,
        number=number,
        action=action,
        translation=clean_qwen_output(str(payload.get("translation", "") or "")),
        source_text=clean_qwen_output(str(payload.get("source_text", "") or "")),
        speaker=optional_clean_string(payload.get("speaker")),
        speaker_confidence=enum_string(payload.get("speaker_confidence"), {"low", "medium", "high"}, "medium"),
        situation=optional_clean_string(payload.get("situation")),
        visual_evidence_type=enum_string(payload.get("visual_evidence_type"), {"none", "weak", "direct"}, "weak"),
        visual_evidence=clean_qwen_output(str(payload.get("visual_evidence", "") or "")),
        confidence=str(payload.get("confidence", "") or "medium").strip().lower(),
        needs_review=bool(payload.get("needs_review", False)),
        risk_flags=tuple(clean_qwen_output(str(value)) for value in payload.get("risk_flags", []) if str(value).strip())
        if isinstance(payload.get("risk_flags", []), list)
        else (),
        warnings=tuple(clean_qwen_output(str(value)) for value in warnings if str(value).strip()),
        raw=payload,
    )


def optional_clean_string(value: object) -> str | None:
    if value is None:
        return None
    text = clean_qwen_output(str(value))
    if not text or text.lower() in {"null", "none", "unknown", "n/a"}:
        return None
    return text


def enum_string(value: object, allowed: set[str], default: str) -> str:
    text = clean_qwen_output(str(value or "")).lower()
    return text if text in allowed else default


def validate_vision_results(
    requests: list[VisionRepairRequest],
    results: list[VisionRepairResult],
) -> tuple[dict[str, VisionRepairResult], dict[str, str]]:
    expected = {request.line_id: request for request in requests}
    seen: set[str] = set()
    accepted: dict[str, VisionRepairResult] = {}
    errors: dict[str, str] = {}
    for result in results:
        request = expected.get(result.line_id)
        if request is None:
            errors[f"unknown_line_id:{result.line_id}"] = "unknown_line_id"
            continue
        if result.line_id in seen:
            errors[result.line_id] = "duplicate_line_id"
            continue
        seen.add(result.line_id)
        if result.number != request.number:
            errors[result.line_id] = "wrong_number"
            continue
        accepted[result.line_id] = result
    for line_id in expected:
        if line_id not in seen:
            errors[line_id] = "missing_line_id"
    return accepted, errors


def validate_vision_facts_results(
    requests: list[VisionFactsRequest],
    results: list[VisionFactsResult],
) -> tuple[dict[str, VisionFactsResult], dict[str, str]]:
    expected = {request.line_id: request for request in requests}
    seen: set[str] = set()
    accepted: dict[str, VisionFactsResult] = {}
    errors: dict[str, str] = {}
    for result in results:
        request = expected.get(result.line_id)
        if request is None:
            errors[f"unknown_line_id:{result.line_id}"] = "unknown_line_id"
            continue
        if result.line_id in seen:
            errors[result.line_id] = "duplicate_line_id"
            continue
        seen.add(result.line_id)
        if result.number != request.number:
            errors[result.line_id] = "wrong_number"
            continue
        accepted[result.line_id] = result
    for line_id in expected:
        if line_id not in seen:
            errors[line_id] = "missing_line_id"
    return accepted, errors


def vision_facts_acceptance_reason(
    request: VisionFactsRequest,
    result: VisionFactsResult | None,
    *,
    malformed_reason: str | None = None,
) -> str | None:
    if malformed_reason:
        return malformed_reason
    if result is None:
        return "missing_result"
    if result.source_text != request.source_text:
        return "source_text_mismatch"
    if result.mapping_confidence == "low":
        return "low_mapping_confidence"
    if result.needs_review:
        return "needs_review"
    if result.risk_flags:
        return f"risk_flags:{','.join(result.risk_flags[:3])}"
    if vision_facts_has_translation_field(result.raw):
        return "contains_translation_field"
    fact_values = [
        result.speaker_position,
        result.visible_emotion,
        result.observable_action,
        *result.facts,
    ]
    if any(visual_fact_looks_invalid(value) for value in fact_values):
        return "invalid_visual_fact"
    if any(visual_fact_looks_like_translation(value) for value in fact_values):
        return "contains_translation"
    if specific_speaker_is_ungrounded(request.source_text, result.speaker_position, result.mapping_confidence):
        return "ungrounded_speaker"
    return None


def vision_facts_has_translation_field(payload: dict[str, object]) -> bool:
    blocked = {"translation", "current_translation", "english", "translated_text", "action"}
    return any(str(key).strip().lower() in blocked for key in payload)


def visual_fact_looks_invalid(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    if normalized.startswith(("{", "[")):
        return True
    if re.search(r"\b(json|schema|prompt|ocr text says|source_text|line_id|translation)\b", normalized):
        return True
    if re.search(r"\b(with text|bubble text|text reads|text says|reads|says)\b", normalized):
        return True
    if re.search(r"\bcontains (?:command|warning|text|words?|line|dialogue|speech)\b", normalized):
        return True
    if re.search(r"\b(thinks|knows|wants|intends|decides|remembers|because)\b", normalized):
        return True
    return False


def visual_fact_looks_like_translation(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized or normalized in {"unknown", "none", "n/a"}:
        return False
    if re.search(r"^[\"'“”].+[\"'“”]$", text.strip()):
        return True
    if re.search(r"\b(says|reads|means|translates to|english is)\b", normalized):
        return True
    if re.search(r"[.!?]$", text.strip()) and len(content_words(text)) >= 3:
        return True
    return False


def vision_acceptance_reason(
    request: VisionRepairRequest,
    result: VisionRepairResult | None,
    *,
    malformed_reason: str | None = None,
) -> str | None:
    if malformed_reason:
        return malformed_reason
    if result is None:
        return "missing_result"
    if result.action == "keep":
        return "vision_keep"
    if result.source_text and result.source_text != request.source_text:
        return "source_text_mismatch"
    if result.confidence == "low":
        return "low_confidence"
    if result.needs_review:
        return "needs_review"
    if result.risk_flags:
        return f"risk_flags:{','.join(result.risk_flags[:3])}"
    if not result.translation:
        return "empty_translation"
    accepted, reason = accept_qwen_translation(
        source_text=request.source_text,
        baseline=request.current_translation,
        candidate=result.translation,
    )
    if not accepted:
        return reason or "invalid_translation"
    if is_render_worse(request.current_translation, result.translation, issue=request.issue):
        return "render_worse"
    if visual_evidence_looks_invalid(result.visual_evidence):
        return "invalid_visual_evidence"
    if visual_evidence_type_mismatch(result.visual_evidence_type, result.visual_evidence):
        return "visual_evidence_type_mismatch"
    if specific_speaker_is_ungrounded(request.source_text, result.speaker, result.speaker_confidence):
        return "ungrounded_speaker"
    if vision_evidence_is_speculative_for_translation_change(request.current_translation, result.translation, result.visual_evidence):
        return "speculative_visual_evidence"
    if translation_romanizes_short_source(request.source_text, request.current_translation, result.translation):
        return "romanized_short_source"
    if translation_invents_proper_noun(request.source_text, request.current_translation, result.translation):
        return "invented_proper_noun"
    if layout_only_change_drifts_semantics(request.issue, request.current_translation, result.translation, result.visual_evidence):
        return "layout_semantic_drift"
    if translation_adds_unsupported_detail(request.current_translation, result.translation, result.visual_evidence):
        return "unsupported_detail"
    return None


def is_render_worse(current: str, candidate: str, *, issue: str) -> bool:
    if suspected_bad_translation(current):
        return False
    if any(marker in issue for marker in ("qwen_rejected", "qwen_repair_failed", "qwen_fallback_rejected")):
        return render_text_length(candidate) > 120
    current_length = render_text_length(current)
    candidate_length = render_text_length(candidate)
    return candidate_length > max(28, int(current_length * 1.35) + 8)


def visual_evidence_looks_invalid(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    if normalized in {"not visually informed", "none", "n/a", "unknown", "unclear"}:
        return True
    if text.startswith(("{", "[")):
        return True
    if re.search(r"\b(json|schema|ocr text says)\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(image supports|visually plausible|looks correct|seems correct|the image confirms)\b", normalized):
        return True
    return False


def visual_evidence_type_mismatch(evidence_type: str, evidence: str) -> bool:
    normalized = evidence.strip().lower()
    if evidence_type == "none":
        return normalized not in {"not visually informed", "none", "n/a", ""}
    if evidence_type == "direct" and normalized in {"not visually informed", "none", "n/a", ""}:
        return True
    return False


def specific_speaker_is_ungrounded(source_text: str, speaker: str | None, confidence: str) -> bool:
    if not speaker:
        return False
    normalized = speaker.strip().lower()
    if normalized in {"unknown", "narrator", "speaker", "character", "male character", "female character"}:
        return False
    if confidence == "low":
        return True
    speaker_names = proper_noun_tokens(speaker)
    if speaker_names and not source_has_name_signal(source_text):
        return True
    if source_has_name_signal(source_text):
        return False
    return False


def vision_evidence_is_speculative_for_translation_change(current: str, candidate: str, visual_evidence: str) -> bool:
    if suspected_bad_translation(current):
        return False
    if equivalent_content(current, candidate):
        return False
    return bool(re.search(r"\b(likely|possibly|probably|might be|seems to be|appears to be|makes the most sense)\b", visual_evidence, flags=re.IGNORECASE))


def translation_invents_proper_noun(source_text: str, current: str, candidate: str) -> bool:
    new_names = proper_noun_tokens(candidate) - proper_noun_tokens(current)
    if not new_names:
        return False
    if source_has_name_signal(source_text):
        return False
    allowed = {"i"}
    return any(name.lower() not in allowed for name in new_names)


def source_has_name_signal(text: str) -> bool:
    return bool(re.search(r"[\u30a0-\u30ff\u4e00-\u9fff\u3008\u3009\u300a\u300b\u30fb]", text))


def proper_noun_tokens(text: str) -> set[str]:
    tokens = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    common_sentence_starters = {
        "After",
        "Agent",
        "Alright",
        "Dad",
        "Father",
        "Here",
        "Mom",
        "Mother",
        "Nice",
        "Okay",
        "The",
        "This",
        "Watch",
        "What",
        "Yeah",
        "Yes",
    }
    return {token for token in tokens if token not in common_sentence_starters}


def translation_romanizes_short_source(source_text: str, current: str, candidate: str) -> bool:
    if len(source_text.strip()) > 3:
        return False
    if content_words(current) & content_words(candidate):
        return False
    return bool(re.search(r"\b[A-Z]?[a-z]*(chu+|kuchu+|sasuga|desu|nya+|tata)\b", candidate, flags=re.IGNORECASE))


def layout_only_change_drifts_semantics(issue: str, current: str, candidate: str, visual_evidence: str) -> bool:
    issue_parts = [part for part in issue.split(",") if part]
    if not issue_parts or any(not part.startswith("layout:") for part in issue_parts):
        return False
    if suspected_bad_translation(current) or equivalent_content(current, candidate):
        return False
    current_words = content_words(current)
    candidate_words = content_words(candidate)
    if not current_words or not candidate_words:
        return False
    overlap = len(current_words & candidate_words) / max(1, min(len(current_words), len(candidate_words)))
    if overlap >= 0.45:
        return False
    evidence_words = content_words(visual_evidence)
    new_words = candidate_words - current_words
    return bool(new_words - evidence_words)


def equivalent_content(current: str, candidate: str) -> bool:
    current_words = content_words(current)
    candidate_words = content_words(candidate)
    if current_words == candidate_words:
        return True
    if not current_words or not candidate_words:
        return current.strip().lower() == candidate.strip().lower()
    overlap = len(current_words & candidate_words) / max(1, max(len(current_words), len(candidate_words)))
    return overlap >= 0.78


def translation_adds_unsupported_detail(current: str, candidate: str, visual_evidence: str) -> bool:
    if suspected_bad_translation(current):
        return False
    current_words = content_words(current)
    candidate_words = content_words(candidate)
    evidence_words = content_words(visual_evidence)
    added = candidate_words - current_words
    unsupported = [word for word in added if word not in evidence_words]
    return len(unsupported) >= 4 and len(candidate_words) >= max(7, len(current_words) + 4)


def content_words(text: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "we",
        "you",
        "your",
    }
    return {word for word in re.findall(r"[a-zA-Z][a-zA-Z']{2,}", text.lower()) if word not in stop_words}
