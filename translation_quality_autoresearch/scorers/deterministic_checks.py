from __future__ import annotations

import re
from typing import Any

from translation_quality_autoresearch.common.text_normalize import compact_for_copy_check, normalize_candidate_text, words
from translation_quality_autoresearch.scorers.glossary_checks import glossary_violations
from translation_quality_autoresearch.scorers.japanese_leakage import count_japanese_chars, japanese_leakage_ratio

ASSISTANT_CHATTER_PATTERNS = (
    r"\bas an ai\b",
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bplease provide\b",
    r"\bprovide more context\b",
    r"\bhere is (?:a|the) translation\b",
    r"\bhere's (?:a|the) translation\b",
    r"\btranslation\s*:",
    r"\bjapanese\s*:",
    r"\benglish\s*:",
    r"\bjson\b",
)


def check_candidate(case, candidate, *, glossary: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    if isinstance(candidate, dict):
        candidate_text = candidate.get("text", "")
    else:
        candidate_text = getattr(candidate, "text", "")
    text = normalize_candidate_text(candidate_text)
    source_text = str(getattr(case, "source_text", ""))
    forbidden_patterns = list(getattr(case, "forbidden_patterns", []) or [])
    warnings: list[str] = []

    empty_output = not text
    assistant_chatter = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in ASSISTANT_CHATTER_PATTERNS)
    ratio = japanese_leakage_ratio(text)
    japanese_leakage = count_japanese_chars(text) > 0 and ratio >= float(config.get("hard_fail_japanese_leakage_ratio", 0.02))
    source_copied = bool(source_text.strip()) and compact_for_copy_check(source_text) == compact_for_copy_check(text)
    forbidden_pattern = any(pattern and re.search(re.escape(str(pattern)), text, flags=re.IGNORECASE) for pattern in forbidden_patterns)
    violations = glossary_violations(case, text, glossary)
    glossary_violation = bool(violations)
    line_mapping_error = has_line_mapping_error(source_text, text)
    oververbose = is_oververbose(case, text, config=config)
    if oververbose:
        warnings.append("oververbose")
    for addition in (getattr(case, "metadata", {}) or {}).get("forbidden_additions", []) or []:
        if str(addition).lower() in text.lower():
            forbidden_pattern = True
            warnings.append(f"forbidden_addition:{addition}")

    hard_fail = any(
        [
            empty_output,
            assistant_chatter,
            japanese_leakage,
            source_copied,
            forbidden_pattern,
            glossary_violation,
            line_mapping_error,
        ]
    )
    if violations:
        warnings.extend(f"glossary_violation:{violation}" for violation in violations)
    return {
        "hard_fail": hard_fail,
        "assistant_chatter": assistant_chatter,
        "japanese_leakage": japanese_leakage,
        "empty_output": empty_output,
        "source_copied": source_copied,
        "forbidden_pattern": forbidden_pattern,
        "glossary_violation": glossary_violation,
        "line_mapping_error": line_mapping_error,
        "oververbose": oververbose,
        "warnings": list(dict.fromkeys(warnings)),
        "japanese_char_count": count_japanese_chars(text),
        "japanese_leakage_ratio": ratio,
    }


def has_line_mapping_error(source_text: str, candidate_text: str) -> bool:
    source_lines = [line for line in str(source_text).splitlines() if line.strip()]
    candidate_lines = [line for line in str(candidate_text).splitlines() if line.strip()]
    return len(source_lines) > 1 and len(candidate_lines) not in {1, len(source_lines)}


def is_oververbose(case, candidate_text: str, *, config: dict[str, Any]) -> bool:
    word_count = len(words(candidate_text))
    max_default = int(config.get("max_target_words_default", 28))
    max_words = int((getattr(case, "metadata", {}) or {}).get("max_target_words", max_default))
    source_word_proxy = max(1.0, len(str(getattr(case, "source_text", ""))) / 3.0)
    return word_count > max_words or word_count / source_word_proxy > float(config.get("oververbose_word_ratio", 3.2))
