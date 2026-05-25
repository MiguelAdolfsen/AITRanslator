from __future__ import annotations

import json
import re

from .qwen_types import QwenCriticDecision, QwenPageTranslation, QwenVerificationDecision

FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
ENGLISH_NUMBER_WORDS = {
    "0": ("zero",),
    "1": ("one", "single"),
    "2": ("two", "second", "both"),
    "3": ("three", "third"),
    "4": ("four", "fourth"),
    "5": ("five", "fifth"),
    "6": ("six", "sixth"),
    "7": ("seven", "seventh"),
    "8": ("eight", "eighth"),
    "9": ("nine", "ninth"),
    "10": ("ten", "tenth"),
    "11": ("eleven", "eleventh"),
    "12": ("twelve", "twelfth", "midnight", "noon"),
}
HONORIFIC_MARKERS = (
    (("さん", "サン"), "san"),
    (("さま", "サマ", "様"), "sama"),
    (("ちゃん", "チャン"), "chan"),
    (("くん", "クン", "君"), "kun"),
    (("先輩", "せんぱい", "センパイ"), "senpai"),
    (("先生", "せんせい", "センセイ"), "sensei"),
    (("殿", "どの", "ドノ"), "dono"),
    (("氏",), "shi"),
)
HONORIFIC_FAMILY_BASES = (
    "父",
    "母",
    "兄",
    "姉",
    "弟",
    "妹",
    "お父",
    "お母",
    "お兄",
    "お姉",
)
KANA_ROMAJI = {
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",
    "か": "ka",
    "き": "ki",
    "く": "ku",
    "け": "ke",
    "こ": "ko",
    "さ": "sa",
    "し": "shi",
    "す": "su",
    "せ": "se",
    "そ": "so",
    "た": "ta",
    "ち": "chi",
    "つ": "tsu",
    "て": "te",
    "と": "to",
    "な": "na",
    "に": "ni",
    "ぬ": "nu",
    "ね": "ne",
    "の": "no",
    "は": "ha",
    "ひ": "hi",
    "ふ": "fu",
    "へ": "he",
    "ほ": "ho",
    "ま": "ma",
    "み": "mi",
    "む": "mu",
    "め": "me",
    "も": "mo",
    "や": "ya",
    "ゆ": "yu",
    "よ": "yo",
    "ら": "ra",
    "り": "ri",
    "る": "ru",
    "れ": "re",
    "ろ": "ro",
    "わ": "wa",
    "を": "o",
    "ん": "n",
    "が": "ga",
    "ぎ": "gi",
    "ぐ": "gu",
    "げ": "ge",
    "ご": "go",
    "ざ": "za",
    "じ": "ji",
    "ず": "zu",
    "ぜ": "ze",
    "ぞ": "zo",
    "だ": "da",
    "ぢ": "ji",
    "づ": "zu",
    "で": "de",
    "ど": "do",
    "ば": "ba",
    "び": "bi",
    "ぶ": "bu",
    "べ": "be",
    "ぼ": "bo",
    "ぱ": "pa",
    "ぴ": "pi",
    "ぷ": "pu",
    "ぺ": "pe",
    "ぽ": "po",
    "きゃ": "kya",
    "きゅ": "kyu",
    "きょ": "kyo",
    "しゃ": "sha",
    "しゅ": "shu",
    "しょ": "sho",
    "ちゃ": "cha",
    "ちゅ": "chu",
    "ちょ": "cho",
    "にゃ": "nya",
    "にゅ": "nyu",
    "にょ": "nyo",
    "ひゃ": "hya",
    "ひゅ": "hyu",
    "ひょ": "hyo",
    "みゃ": "mya",
    "みゅ": "myu",
    "みょ": "myo",
    "りゃ": "rya",
    "りゅ": "ryu",
    "りょ": "ryo",
    "ぎゃ": "gya",
    "ぎゅ": "gyu",
    "ぎょ": "gyo",
    "じゃ": "ja",
    "じゅ": "ju",
    "じょ": "jo",
    "びゃ": "bya",
    "びゅ": "byu",
    "びょ": "byo",
    "ぴゃ": "pya",
    "ぴゅ": "pyu",
    "ぴょ": "pyo",
}


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


def parse_qwen_critic(text: str) -> QwenCriticDecision:
    payloads = extract_qwen_json_payloads(text)
    if not payloads:
        loose_payload = extract_qwen_critic_loose_payload(text)
        payloads = [loose_payload] if loose_payload is not None else []
    if not payloads:
        return QwenCriticDecision(
            ok=False,
            severity="high",
            issues=("invalid_json",),
            reason="invalid_critic_json",
            raw_response=text,
        )
    decisions = [qwen_critic_decision_from_payload(payload, raw_response=text) for payload in payloads]
    for decision in decisions:
        if decision.ok:
            return decision
    return decisions[-1]


def qwen_critic_decision_from_payload(
    payload: dict[str, object],
    *,
    raw_response: str,
) -> QwenCriticDecision:
    severity = clean_qwen_output(str(payload.get("severity", "") or "")).lower()
    if severity not in {"none", "low", "medium", "high"}:
        severity = "medium" if bool(payload.get("ok")) is False else "none"
    raw_issues = payload.get("issues", [])
    if not isinstance(raw_issues, list):
        raw_issues = []
    issues = tuple(
        normalize_critic_issue(str(issue))
        for issue in raw_issues
        if normalize_critic_issue(str(issue))
    )
    reason = clean_qwen_output(str(payload.get("reason", "") or "")).strip()
    if not reason or is_placeholder_qwen_value(reason):
        reason = "accepted" if severity in {"none", "low"} else "flagged"
    if severity == "none" and issues:
        severity = "medium"
    ok = bool(payload.get("ok", severity in {"none", "low"}))
    if severity in {"medium", "high"}:
        ok = False
    if re.search(r"\b(json|schema|prompt)\b", reason, flags=re.IGNORECASE):
        ok = False
        if "critic_schema_talk" not in issues:
            issues = (*issues, "critic_schema_talk")
        severity = "medium"
    return QwenCriticDecision(
        ok=ok,
        severity=severity,
        issues=issues,
        reason=reason,
        raw_response=raw_response,
        source_evidence=tuple(clean_qwen_output(str(value)) for value in list_string_payload(payload.get("source_evidence"))),
        translation_evidence=tuple(clean_qwen_output(str(value)) for value in list_string_payload(payload.get("translation_evidence"))),
        repair_recommended=payload.get("repair_recommended") if isinstance(payload.get("repair_recommended"), bool) else None,
    )


def extract_qwen_critic_loose_payload(text: str) -> dict[str, object] | None:
    severity = extract_qwen_string_field(text, "severity")
    reason = extract_qwen_string_field(text, "reason")
    ok = extract_qwen_bool_field(text, "ok")
    issues = extract_qwen_array_strings(text, "issues")
    if severity is None and reason is None and ok is None and issues is None:
        return None
    payload: dict[str, object] = {}
    if severity is not None:
        payload["severity"] = severity
    if reason is not None:
        payload["reason"] = reason
    if ok is not None:
        payload["ok"] = ok
    if issues is not None:
        payload["issues"] = issues
    return payload


def list_string_payload(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_critic_issue(issue: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", issue.strip().lower()).strip("_")
    allowed = {
        "awkward_literal",
        "context_mismatch",
        "omitted_term",
        "invented_detail",
        "name_drift",
        "tone_mismatch",
        "untranslated_text",
        "glossary_conflict",
        "grammar_problem",
        "invalid_json",
        "critic_schema_talk",
    }
    return normalized if normalized in allowed else ""


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
    if re.match(r"^\s*english\s*:", candidate, flags=re.IGNORECASE):
        return False, "explanatory_output"
    if candidate.lower() in {"unclear", "[unclear]", "unknown"}:
        return False, "unclear"
    if not any(char.isascii() and char.isalnum() for char in candidate):
        return False, "no_english_words"
    if len(candidate) > 180:
        return False, "too_long"
    if count_japanese_chars(source_text) > 0 and re.fullmatch(r"\([A-Za-z][A-Za-z\s.'!?-]{2,}\)", candidate):
        return False, "stage_direction_only"
    if re.search(r"\b[A-Za-z]+(?:-[A-Za-z]+){4,}\b", candidate):
        return False, "malformed_hyphen_chain"
    if has_malformed_english_pattern(candidate):
        return False, "grammar_problem"
    if short_kana_fragment_gets_unsupported_first_person_detail(source_text, candidate):
        return False, "unsupported_first_person_salvage"
    if identity_fragment_invents_detail(source_text, candidate):
        return False, "invented_identity_detail"
    if ruby_annotation_literal_gloss(source_text, candidate):
        return False, "ruby_literal_gloss"
    if kana_only_fragment_gets_fluent_sentence(source_text, candidate):
        return False, "ocr_risk_fluent_sentence"
    if mixed_symbol_noise_gets_translation(source_text, candidate):
        return False, "ocr_symbol_noise_translation"
    if source_has_bracket_term(source_text) and not translation_preserves_bracket_term(source_text, candidate):
        return False, "dropped_bracket_term"
    if source_has_latin_code(source_text) and not translation_preserves_latin_codes(source_text, candidate):
        return False, "dropped_latin_code"
    if source_has_number(source_text) and not translation_preserves_numbers(source_text, candidate):
        return False, "dropped_number"
    if source_has_honorific(source_text) and not translation_preserves_honorific(source_text, candidate):
        return False, "dropped_honorific"
    if source_uses_gender_neutral_group_address(source_text) and re.search(
        r"\b(boys|girls|men|women|ladies|gentlemen)\b",
        candidate,
        flags=re.IGNORECASE,
    ):
        return False, "unnecessary_gendering"
    if source_has_confirmation_ending(source_text) and not translation_preserves_confirmation_ending(candidate):
        return False, "dropped_confirmation_tone"
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
        "grammar_problem",
        "invented_identity_detail",
        "ruby_literal_gloss",
        "ocr_risk_fluent_sentence",
        "ocr_symbol_noise_translation",
        "unsupported_first_person_salvage",
        "dropped_bracket_term",
        "dropped_latin_code",
        "dropped_number",
        "dropped_honorific",
        "unnecessary_gendering",
        "dropped_confirmation_tone",
        "much_longer_than_baseline",
        "short_source_long_translation",
        "hedging_or_explanation",
        "repeated_phrase_loop",
    }


def source_uses_gender_neutral_group_address(source_text: str) -> bool:
    compact = re.sub(r"\s+", "", source_text)
    return any(
        marker in compact
        for marker in (
            "\u304a\u307e\u3048\u3089",
            "\u304a\u524d\u3089",
            "\u30aa\u30de\u30a8\u3089",
            "\u307f\u3093\u306a",
            "\u307f\u306a",
            "\u7686",
        )
    )


def source_has_confirmation_ending(source_text: str) -> bool:
    compact = re.sub(r"\s+", "", source_text)
    return compact.endswith(("\u3060\u308d", "\u3060\u308d\u3046", "\u3067\u3057\u3087", "\u3067\u3057\u3087\u3046"))


def translation_preserves_confirmation_ending(candidate: str) -> bool:
    lower = candidate.lower()
    return bool(
        "?" in candidate
        or re.search(r"\b(?:right|isn't it|aren't you|aren't we|don't you|don't we|didn't you|didn't we|didn't i|wouldn't you|wouldn't we)\b", lower)
    )


def source_has_bracket_term(source_text: str) -> bool:
    return enforced_bracket_term_kind(source_text) is not None


def has_malformed_english_pattern(candidate: str) -> bool:
    lower = " ".join(candidate.lower().split())
    return bool(
        re.search(r"\bbeing\s+(?:the\s+)?[a-z]{3,}\s+of\b", lower)
        or re.search(r"\b(?:is|are|was|were)\s+the\s+[a-z]{3,}\s+of\s+[a-z]{3,}\b", lower)
        or re.search(r"\b(?:feeling|thought|voice)\s+of\s+(?:chest|body|heart|mind)\b", lower)
    )


def identity_fragment_invents_detail(source_text: str, candidate: str) -> bool:
    source = re.sub(r"\s+", "", str(source_text))
    words = re.findall(r"[A-Za-z']+", candidate)
    lower = " ".join(candidate.lower().split())
    if is_short_katakana_name_source(source) and len(words) > 2:
        return True
    if "あの人" in source and has_unsupported_english_name(candidate):
        return True
    if "あの子" in source and re.search(
        r"\b(?:my|his|her|our|their)\s+(?:little\s+|big\s+|older\s+|younger\s+)?"
        r"(?:sister|brother|mother|father|mom|dad|daughter|son|wife|husband)\b",
        lower,
    ):
        return True
    return False


def is_short_katakana_name_source(source_text: str) -> bool:
    return bool(re.fullmatch(r"[\u30a1-\u30fa\u30fc]{2,8}", source_text))


def has_unsupported_english_name(candidate: str) -> bool:
    blocked = {"I", "Is", "Are", "Am", "Who", "What", "Where", "When", "Why", "How", "That", "This", "The"}
    names = re.findall(r"\b[A-Z][a-z]{2,}\b", candidate)
    return any(name not in blocked for name in names)


def ruby_annotation_literal_gloss(source_text: str, candidate: str) -> bool:
    source = re.sub(r"\s+", "", str(source_text))
    match = re.search(r"([\u3040-\u30ff\u30fc]{2,})[（(][\u3400-\u9fff]{1,4}[）)]", source)
    if not match:
        return False
    words = re.findall(r"[A-Za-z]+", candidate)
    if len(words) != 1:
        return False
    reading_romaji = kana_to_rough_romaji(match.group(1))
    candidate_word = words[0].lower()
    return bool(reading_romaji and candidate_word != reading_romaji)


def kana_to_rough_romaji(text: str) -> str:
    kana = "".join(katakana_to_hiragana(char) for char in text)
    result: list[str] = []
    index = 0
    double_next = False
    while index < len(kana):
        char = kana[index]
        if char == "っ":
            double_next = True
            index += 1
            continue
        if char == "ー":
            index += 1
            continue
        piece = ""
        if index + 1 < len(kana):
            pair = kana[index : index + 2]
            piece = KANA_ROMAJI.get(pair, "")
            if piece:
                index += 2
            else:
                piece = KANA_ROMAJI.get(char, "")
                index += 1
        else:
            piece = KANA_ROMAJI.get(char, "")
            index += 1
        if not piece:
            continue
        if double_next:
            piece = piece[0] + piece
            double_next = False
        result.append(piece)
    return "".join(result)


def katakana_to_hiragana(char: str) -> str:
    code = ord(char)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    return char


def kana_only_fragment_gets_fluent_sentence(source_text: str, candidate: str) -> bool:
    source = re.sub(r"\s+", "", str(source_text))
    if not re.fullmatch(r"[\u3040-\u30ff\u30fc]{7,12}", source):
        return False
    if any(char in source for char in ("\u3002", "\uff1f", "\uff01", "\u2026", "?", "!")):
        return False
    words = re.findall(r"[A-Za-z']+", candidate)
    return len(words) >= 5


def mixed_symbol_noise_gets_translation(source_text: str, candidate: str) -> bool:
    source = re.sub(r"\s+", "", str(source_text))
    japanese_count = count_japanese_chars(source)
    if not (1 <= japanese_count <= 3):
        return False
    if not (re.search(r"[☆★♡♥]", source) or re.search(r"[!?！？]{2,}", source)):
        return False
    return bool(re.search(r"[A-Za-z]", candidate))


def short_kana_fragment_gets_unsupported_first_person_detail(source_text: str, candidate: str) -> bool:
    source = re.sub(r"\s+", "", str(source_text))
    if count_japanese_chars(source) > 6:
        return False
    if not re.search(r"[\u30a1-\u30fa\u30fc]{2,}", source):
        return False
    if any(marker in source for marker in ("\u79c1", "\u50d5", "\u4ffa", "\u3042\u305f\u3057")):
        return False
    return bool(
        re.search(
            r"\b(?:i|we)\s+(?:have|had|am|was|were|will|want|need|love|hate|own|remember)\b",
            candidate,
            flags=re.IGNORECASE,
        )
        or re.search(r"\bmy\s+[a-z]{3,}\b", candidate, flags=re.IGNORECASE)
    )


def source_has_number(source_text: str) -> bool:
    return bool(source_numbers(source_text))


def translation_preserves_numbers(source_text: str, candidate: str) -> bool:
    normalized = candidate.translate(FULLWIDTH_DIGITS)
    lower = normalized.lower()
    for number in source_numbers(source_text):
        if number in normalized:
            continue
        words = ENGLISH_NUMBER_WORDS.get(number, ())
        if words and any(re.search(rf"\b{re.escape(word)}\b", lower) for word in words):
            continue
        return False
    return True


def source_numbers(source_text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\d+", str(source_text).translate(FULLWIDTH_DIGITS))))


def source_has_latin_code(source_text: str) -> bool:
    return bool(source_latin_codes(source_text))


def translation_preserves_latin_codes(source_text: str, candidate: str) -> bool:
    normalized = re.sub(r"[^A-Za-z0-9]+", "", candidate).lower()
    return all(re.sub(r"[^A-Za-z0-9]+", "", code).lower() in normalized for code in source_latin_codes(source_text))


def source_latin_codes(source_text: str) -> list[str]:
    codes = re.findall(r"\b[A-Z][A-Z0-9_-]*\d[A-Z0-9_-]*\b", str(source_text).translate(FULLWIDTH_DIGITS))
    return list(dict.fromkeys(code for code in codes if len(code) >= 2))


def source_has_honorific(source_text: str) -> bool:
    return any(source_honorifics(source_text))


def translation_preserves_honorific(source_text: str, candidate: str) -> bool:
    lower = candidate.lower()
    return all(re.search(rf"\b{re.escape(honorific)}\b", lower) for honorific in source_honorifics(source_text))


def source_honorifics(source_text: str) -> list[str]:
    text = str(source_text)
    expected: list[str] = []
    for markers, rendered in HONORIFIC_MARKERS:
        if any(source_has_non_family_honorific(text, marker) for marker in markers):
            expected.append(rendered)
    return list(dict.fromkeys(expected))


def source_has_non_family_honorific(source_text: str, marker: str) -> bool:
    start = 0
    while True:
        index = source_text.find(marker, start)
        if index < 0:
            return False
        prefix = source_text[:index]
        if not any(prefix.endswith(base) for base in HONORIFIC_FAMILY_BASES):
            return True
        start = index + len(marker)


def translation_preserves_bracket_term(source_text: str, candidate: str) -> bool:
    normalized = candidate.lower()
    kind = enforced_bracket_term_kind(source_text)
    if kind is None:
        return True
    if kind == "code":
        return any(term.lower() in normalized for term in bracket_terms(source_text) if is_code_like_bracket_term(term))
    return True


def enforced_bracket_term_kind(source_text: str) -> str | None:
    terms = bracket_terms(source_text)
    if not terms:
        return None
    for term in terms:
        if is_code_like_bracket_term(term):
            return "code"
    return None


def is_code_like_bracket_term(term: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", term) and (re.search(r"\d", term) or term.upper() == term))


def bracket_terms(source_text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"\u3008([^>\u3009]{1,20})\u3009", source_text)
        if match.group(1).strip()
    ]


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
