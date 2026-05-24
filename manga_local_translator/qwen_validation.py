from __future__ import annotations

import json
import re

from .qwen_types import QwenPageTranslation, QwenVerificationDecision


def clean_qwen_output(text: str) -> str:
    cleaned = strip_qwen_wrappers(text)
    payload = extract_qwen_json_payload(cleaned)
    if payload and "translation" in payload:
        translated = str(payload.get("translation", "") or "").strip()
        if translated:
            return clean_qwen_output(translated)
    cleaned = cleaned.strip().strip("\"'")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[0]
    return cleaned.strip()


def strip_qwen_wrappers(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^English(?: translation)?\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def strip_qwen_wrappers_for_json(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^English(?: translation)?\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_qwen_json_payload(text: str) -> dict[str, object] | None:
    payloads = extract_qwen_json_payloads(text)
    return payloads[0] if payloads else None


def extract_qwen_json_payloads(text: str) -> list[dict[str, object]]:
    cleaned = strip_qwen_wrappers_for_json(text)
    decoder = json.JSONDecoder()
    payloads: list[dict[str, object]] = []
    index = 0
    while index < len(cleaned):
        start = cleaned.find("{", index)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
        index = start + max(1, end)
    return payloads


def parse_qwen_translation(text: str) -> str:
    payload = extract_qwen_json_payload(text)
    if payload is not None:
        translation = str(payload.get("translation", "") or "").strip()
        if translation:
            return clean_qwen_output(translation)
    translation = extract_qwen_string_field(text, "translation")
    if translation:
        return clean_qwen_output(translation)
    cleaned = clean_qwen_output(text)
    return cleaned


def parse_qwen_page_translations(text: str) -> list[QwenPageTranslation]:
    best: list[QwenPageTranslation] = []
    for items in extract_qwen_page_translation_payloads(text):
        parsed = [item for item in (qwen_page_translation_from_payload(payload) for payload in items) if item is not None]
        if len(parsed) > len(best):
            best = parsed
    return best


def extract_qwen_page_translation_payloads(text: str) -> list[list[dict[str, object]]]:
    cleaned = strip_qwen_wrappers_for_json(text)
    decoder = json.JSONDecoder()
    groups: list[list[dict[str, object]]] = []
    index = 0
    while index < len(cleaned):
        start_candidates = [value for value in (cleaned.find("{", index), cleaned.find("[", index)) if value >= 0]
        if not start_candidates:
            break
        start = min(start_candidates)
        try:
            payload, end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict):
            translations = payload.get("translations")
            if isinstance(translations, list):
                groups.append([item for item in translations if isinstance(item, dict)])
            elif "id" in payload and "translation" in payload:
                groups.append([payload])
        elif isinstance(payload, list):
            groups.append([item for item in payload if isinstance(item, dict)])
        index = start + max(1, end)

    groups.extend(recover_qwen_page_translation_object_groups(cleaned))
    return groups


def recover_qwen_page_translation_object_groups(text: str) -> list[list[dict[str, object]]]:
    recovered: list[dict[str, object]] = []
    for match in re.finditer(r'\{[^{}]*"id"\s*:\s*[^{}]*"translation"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*"[^{}]*\}', text, flags=re.DOTALL):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            recovered.append(payload)
    if not recovered:
        return []

    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    seen: set[int] = set()
    previous_id = 0
    for payload in recovered:
        try:
            page_id = int(payload.get("id", 0))
        except (TypeError, ValueError):
            page_id = 0
        if current and (page_id in seen or (previous_id and page_id < previous_id)):
            groups.append(current)
            current = []
            seen = set()
        current.append(payload)
        if page_id:
            seen.add(page_id)
            previous_id = page_id
    if current:
        groups.append(current)
    return groups


def qwen_page_translation_from_payload(payload: dict[str, object]) -> QwenPageTranslation | None:
    try:
        page_id = int(payload.get("id", 0))
    except (TypeError, ValueError):
        return None
    translation = clean_qwen_output(str(payload.get("translation", "") or ""))
    if not page_id or not translation:
        return None
    confidence = parse_optional_float(payload.get("confidence"))
    reason = clean_qwen_output(str(payload.get("reason", "") or ""))
    return QwenPageTranslation(
        page_id=page_id,
        translation=translation,
        confidence=confidence,
        reason=reason,
    )


def parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_qwen_verification(text: str) -> QwenVerificationDecision:
    payloads = extract_qwen_json_payloads(text)
    if not payloads:
        loose_payload = extract_qwen_loose_payload(text)
        payloads = [loose_payload] if loose_payload is not None else []
    if not payloads:
        return QwenVerificationDecision(False, "invalid", "invalid_verification_json", raw_response=text)
    decisions = [qwen_verification_decision_from_payload(payload, raw_response=text) for payload in payloads]
    for decision in decisions:
        if decision.ok and decision.choice == "correction" and decision.translation:
            return decision
    for decision in decisions:
        if decision.ok and decision.choice == "candidate":
            return decision
    for decision in decisions:
        if decision.ok:
            return decision
    return decisions[-1]


def qwen_verification_decision_from_payload(
    payload: dict[str, object],
    *,
    raw_response: str,
) -> QwenVerificationDecision:
    choice = str(payload.get("choice", "") or "").strip().lower()
    if choice not in {"candidate", "baseline", "correction"}:
        choice = "candidate" if bool(payload.get("ok")) else "baseline"
    ok = bool(payload.get("ok")) and choice in {"candidate", "correction"}
    if choice == "baseline":
        ok = False
    reason = str(payload.get("reason", "") or ("verified" if ok else "rejected")).strip()
    translation = str(payload.get("translation", "") or "").strip()
    translation = clean_qwen_output(translation) if translation else None
    if translation and is_placeholder_qwen_value(translation):
        translation = None
    if is_placeholder_qwen_value(reason):
        reason = "verified" if ok else "rejected"
    unsupported_terms = payload.get("unsupported_terms", [])
    if not isinstance(unsupported_terms, list):
        unsupported_terms = []
    clean_terms = tuple(
        clean_qwen_output(str(term))
        for term in unsupported_terms
        if str(term).strip()
    )
    if clean_terms and choice == "candidate":
        ok = False
        if reason:
            reason = f"{reason}; unsupported_terms={', '.join(clean_terms)}"
        else:
            reason = f"unsupported_terms={', '.join(clean_terms)}"
    if choice == "candidate" and not translation:
        translation = None
    if choice == "correction" and not translation:
        ok = False
        reason = f"{reason}; missing_correction" if reason else "missing_correction"
    if choice == "baseline":
        translation = None
    return QwenVerificationDecision(ok, choice, reason, translation, clean_terms, raw_response=raw_response)


def is_placeholder_qwen_value(text: str) -> bool:
    stripped = text.strip()
    return stripped in {"", "...", "…", "N/A", "none", "None"}


def extract_qwen_loose_payload(text: str) -> dict[str, object] | None:
    choice = extract_qwen_string_field(text, "choice")
    translation = extract_qwen_string_field(text, "translation")
    reason = extract_qwen_string_field(text, "reason")
    ok = extract_qwen_bool_field(text, "ok")
    if choice is None and translation is None and reason is None and ok is None:
        return None
    payload: dict[str, object] = {}
    if choice is not None:
        payload["choice"] = choice
    if translation is not None:
        payload["translation"] = translation
    if reason is not None:
        payload["reason"] = reason
    if ok is not None:
        payload["ok"] = ok
    unsupported_terms = extract_qwen_array_strings(text, "unsupported_terms")
    if unsupported_terms is not None:
        payload["unsupported_terms"] = unsupported_terms
    return payload


def extract_qwen_string_field(text: str, field: str) -> str | None:
    cleaned = strip_qwen_wrappers_for_json(text)
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return str(json.loads(f'"{match.group(1)}"')).strip()
    except json.JSONDecodeError:
        return match.group(1).strip()


def extract_qwen_bool_field(text: str, field: str) -> bool | None:
    cleaned = strip_qwen_wrappers_for_json(text)
    match = re.search(rf'"{re.escape(field)}"\s*:\s*(true|false)', cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"


def extract_qwen_array_strings(text: str, field: str) -> list[str] | None:
    cleaned = strip_qwen_wrappers_for_json(text)
    match = re.search(rf'"{re.escape(field)}"\s*:\s*\[(.*?)\]', cleaned, flags=re.DOTALL)
    if not match:
        return None
    return [
        value
        for value in (extract_qwen_string_literal(item) for item in re.findall(r'"((?:\\.|[^"\\])*)"', match.group(1)))
        if value
    ]


def extract_qwen_string_literal(value: str) -> str | None:
    try:
        return str(json.loads(f'"{value}"')).strip()
    except json.JSONDecodeError:
        return value.strip() or None


def accept_qwen_translation(
    *,
    source_text: str,
    baseline: str,
    candidate: str,
) -> tuple[bool, str | None]:
    candidate = candidate.strip()
    baseline = baseline.strip()
    if not candidate:
        return False, "empty"
    if candidate.startswith(("{", "[")):
        return False, "invalid_json_fragment"
    if count_japanese_chars(candidate) > 0:
        return False, "contains_japanese"
    if "<|" in candidate or re.search(r"</\w+>|<think\b[^>]*>|<\w+\s+[^>]*>", candidate, flags=re.IGNORECASE):
        return False, "markup_or_thinking_tag"
    if re.search(r"\b(translation|japanese|baseline|context|ocr|json)\b", candidate, flags=re.IGNORECASE):
        return False, "explanatory_output"
    if candidate.lower() in {"unclear", "[unclear]", "unknown"}:
        return False, "unclear"
    if not any(char.isascii() and char.isalnum() for char in candidate):
        return False, "no_english_words"
    if len(candidate) > 180:
        return False, "too_long"
    if len(source_text) > 12 and re.fullmatch(r"\([A-Za-z][A-Za-z\s.'!?-]{2,}\)", candidate):
        return False, "stage_direction_only"
    if re.search(r"\b[A-Za-z]+(?:-[A-Za-z]+){4,}\b", candidate):
        return False, "malformed_hyphen_chain"
    if re.search(r"\b(second-way|two-and-a-half|freaks?|candy store,\s*san)\b", candidate, flags=re.IGNORECASE):
        return False, "known_hallucination_artifact"
    if source_uses_gender_neutral_group_address(source_text) and re.search(
        r"\b(boys|girls|men|women|ladies|gentlemen)\b",
        candidate,
        flags=re.IGNORECASE,
    ):
        return False, "unnecessary_gendering"
    if (
        "\u30aa\u30da\u30ec\u30fc\u30b7\u30e7\u30f3" in source_text
        and "\u74e6\u89e3" in source_text
        and re.search(r"\b(what are you doing|don't know what to do)\b", candidate, flags=re.IGNORECASE)
    ):
        return False, "operation_collapse_mismatch"
    candidate_words = candidate.split()
    baseline_words = baseline.split()
    if baseline_words and len(candidate_words) > max(18, len(baseline_words) * 2 + 6):
        return False, "much_longer_than_baseline"
    if len(source_text) <= 8 and len(candidate_words) > 8:
        return False, "short_source_long_translation"
    if re.search(r"\b(i think|maybe|probably|it means|this means)\b", candidate, flags=re.IGNORECASE):
        return False, "hedging_or_explanation"
    if has_repeated_ngram(candidate_words):
        return False, "repeated_phrase_loop"
    return True, None


def should_try_qwen_repair(reason: str | None) -> bool:
    return reason in {
        "empty",
        "invalid_json_fragment",
        "no_english_words",
        "markup_or_thinking_tag",
        "explanatory_output",
        "too_long",
        "stage_direction_only",
        "malformed_hyphen_chain",
        "known_hallucination_artifact",
        "unnecessary_gendering",
        "operation_collapse_mismatch",
        "much_longer_than_baseline",
        "short_source_long_translation",
        "hedging_or_explanation",
        "repeated_phrase_loop",
    }


def source_uses_gender_neutral_group_address(source_text: str) -> bool:
    compact = re.sub(r"\s+", "", source_text)
    return any(marker in compact for marker in ("\u304a\u307e\u3048\u3089", "\u304a\u524d\u3089", "\u30aa\u30de\u30a8\u3089"))


def verifier_reason_says_reject(reason: str) -> bool:
    normalized = reason.lower()
    if not normalized:
        return False
    safe_phrases = (
        "candidate matches",
        "candidate is direct",
        "candidate is accurate",
        "no added",
        "does not add",
        "without adding",
        "no unsupported",
        "not unsupported",
        "no drift",
        "does not drift",
    )
    if any(phrase in normalized for phrase in safe_phrases):
        return False
    reject_markers = (
        "invent",
        "unsupported",
        "adds ",
        "added ",
        "not in japanese",
        "not in the japanese",
        "not implied",
        "drift",
        "mistranslat",
        "ignores",
    )
    return any(marker in normalized for marker in reject_markers)


def has_repeated_ngram(words: list[str], *, n: int = 3, max_count: int = 2) -> bool:
    normalized = [
        re.sub(r"[^a-z0-9']+", "", word.lower())
        for word in words
    ]
    normalized = [word for word in normalized if word]
    if len(normalized) < n * (max_count + 1):
        return False
    counts: dict[tuple[str, ...], int] = {}
    for index in range(0, len(normalized) - n + 1):
        ngram = tuple(normalized[index:index + n])
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] > max_count:
            return True
    return False


def count_japanese_chars(text: str) -> int:
    count = 0
    for char in text:
        code = ord(char)
        if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0x4E00 <= code <= 0x9FFF:
            count += 1
    return count
