from __future__ import annotations

import math
import re
from typing import Any


WEIGHTS = {
    "accepted_prompt_chatter": 12000,
    "accepted_japanese_leakage": 11000,
    "accepted_prompt_fragment": 10000,
    "accepted_schema_fragment": 10000,
    "source_text_mutation": 9000,
    "accepted_forbidden_pattern": 7000,
    "accepted_overlong_fragment": 4500,
    "accepted_repetitive": 3500,
    "false_reject": 3000,
    "empty_output": 2000,
    "verbose_output": 1000,
    "model_error": 12000,
}

CATEGORY_MIN_APPROVAL_RATE = 0.80
CATEGORY_MIN_EVALUATIONS = 2
HARD_VIOLATIONS = {
    "accepted_prompt_chatter",
    "accepted_japanese_leakage",
    "accepted_prompt_fragment",
    "accepted_schema_fragment",
    "source_text_mutation",
    "accepted_forbidden_pattern",
    "model_error",
}


def score_case(case: dict[str, Any], result: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    source = str(case.get("source_text", "") or "")
    raw = str(result.get("raw_output", "") or "")
    final = str(result.get("final_output", "") or "")
    reject_reason = str(result.get("reject_reason", "") or "")
    accepted = bool(final.strip()) and not reject_reason
    expected_accept = str(reference.get("expected_decision")) == "accept"
    violations: list[str] = []

    if result.get("model_error"):
        violations.append("model_error")
    if not raw.strip() and expected_accept:
        violations.append("empty_output")
    if accepted and source.strip() and final.strip() == source.strip():
        violations.append("source_text_mutation")
    if accepted and contains_prompt_chatter(f"{raw}\n{final}"):
        violations.append("accepted_prompt_chatter")
    if accepted and contains_japanese(final) and not bool(reference.get("allowed_japanese_output")):
        violations.append("accepted_japanese_leakage")
    if accepted and contains_prompt_fragment(f"{raw}\n{final}"):
        violations.append("accepted_prompt_fragment")
    if accepted and contains_schema_fragment(f"{raw}\n{final}"):
        violations.append("accepted_schema_fragment")
    if accepted and forbidden_pattern_present(final, reference.get("forbidden_patterns", [])):
        violations.append("accepted_forbidden_pattern")
    if accepted and output_too_long_for_case(case, final, reference):
        violations.append("accepted_overlong_fragment")
    if accepted and looks_repetitive(final):
        violations.append("accepted_repetitive")
    if accepted and looks_verbose(case, final, reference):
        violations.append("verbose_output")
    if expected_accept and not accepted:
        violations.append("false_reject")
    if not expected_accept and accepted:
        violations.append("accepted_forbidden_pattern")

    approved = not violations and ((expected_accept and accepted) or (not expected_accept and not accepted))
    unique = list(dict.fromkeys(violations))
    outcome_class = classify_outcome(expected_accept=expected_accept, accepted=accepted, approved=approved, violations=unique)
    counts = {f"{name}_count": 0 for name in WEIGHTS}
    for violation in unique:
        key = f"{violation}_count"
        if key in counts:
            counts[key] = 1
    latency_ms = float(result.get("latency_ms", 0.0) or 0.0)
    case_quality_score = sum(WEIGHTS[name] for name in WEIGHTS if counts.get(f"{name}_count", 0))
    case_score = case_quality_score + 10 * latency_ms
    return {
        "case_id": case.get("case_id"),
        "source_type": case.get("source_type"),
        "accepted": accepted,
        "expected_accept": expected_accept,
        "approved": approved,
        "outcome_class": outcome_class,
        "raw_output": raw,
        "final_output": final,
        "reject_reason": reject_reason,
        "violations": unique,
        "latency_ms": round(latency_ms, 3),
        "case_quality_score": round(case_quality_score, 6),
        "case_score": round(case_score, 6),
        **counts,
    }


def summarize_metrics(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(per_case)
    metrics: dict[str, Any] = {"cases_total": total}
    for name in WEIGHTS:
        metrics[f"{name}_count"] = sum(int(row.get(f"{name}_count", 0)) for row in per_case)
    metrics["cat_approved_count"] = sum(1 for row in per_case if row.get("approved"))
    metrics["expected_accept_count"] = sum(1 for row in per_case if row.get("expected_accept"))
    metrics["expected_reject_count"] = total - metrics["expected_accept_count"]
    metrics["accepted_count"] = sum(1 for row in per_case if row.get("accepted"))
    metrics["rejected_count"] = total - metrics["accepted_count"]
    metrics["cat_approval_rate"] = round(metrics["cat_approved_count"] / max(1, total), 6)
    for outcome in ("clean_accept", "safe_reject", "false_reject", "unsafe_accept", "weak_accept", "unclear_fail"):
        metrics[f"{outcome}_count"] = sum(1 for row in per_case if row.get("outcome_class") == outcome)
    metrics["mean_latency_ms"] = round(
        sum(float(row.get("latency_ms", 0.0) or 0.0) for row in per_case) / max(1, total),
        6,
    )
    quality_score = sum(WEIGHTS[name] * (metrics[f"{name}_count"] / max(1, total)) for name in WEIGHTS)
    latency_score = 10 * metrics["mean_latency_ms"]
    metrics["cat_quality_score"] = round(quality_score, 6)
    metrics["cat_latency_score"] = round(latency_score, 6)
    metrics["cat_response_score"] = round(quality_score + latency_score, 6)
    category_metrics = summarize_category_metrics(per_case)
    low_categories = [
        name
        for name, data in sorted(category_metrics.items())
        if int(data.get("evaluations", 0)) >= CATEGORY_MIN_EVALUATIONS
        and float(data.get("approval_rate", 0.0)) < CATEGORY_MIN_APPROVAL_RATE
    ]
    metrics["category_metrics"] = category_metrics
    metrics["category_min_approval_rate"] = round(
        min((float(data.get("approval_rate", 0.0)) for data in category_metrics.values()), default=1.0),
        6,
    )
    metrics["low_category_count"] = len(low_categories)
    metrics["low_categories"] = ",".join(low_categories)
    metrics["hard_failure"] = hard_failure(metrics)
    return metrics


def classify_outcome(*, expected_accept: bool, accepted: bool, approved: bool, violations: list[str]) -> str:
    if approved and expected_accept and accepted:
        return "clean_accept"
    if approved and not expected_accept and not accepted:
        return "safe_reject"
    if expected_accept and not accepted:
        return "false_reject"
    if not expected_accept and accepted:
        return "unsafe_accept"
    if accepted and violations and not any(violation in HARD_VIOLATIONS for violation in violations):
        return "weak_accept"
    if accepted and violations:
        return "unsafe_accept"
    return "unclear_fail"


def summarize_category_metrics(per_case: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_case:
        grouped.setdefault(str(row.get("source_type") or "unknown"), []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for source_type, rows in sorted(grouped.items()):
        evaluations = len(rows)
        approved = sum(1 for row in rows if row.get("approved"))
        summary[source_type] = {
            "evaluations": evaluations,
            "approved": approved,
            "approval_rate": round(approved / max(1, evaluations), 6),
            "false_reject_count": sum(1 for row in rows if row.get("outcome_class") == "false_reject"),
            "unsafe_accept_count": sum(1 for row in rows if row.get("outcome_class") == "unsafe_accept"),
            "weak_accept_count": sum(1 for row in rows if row.get("outcome_class") == "weak_accept"),
        }
    return summary


def hard_failure(metrics: dict[str, Any]) -> bool:
    try:
        value = float(metrics.get("cat_response_score"))
        if math.isnan(value):
            return True
    except (TypeError, ValueError):
        return True
    return any(
        int(metrics.get(key, 0) or 0) > 0
        for key in (
            "accepted_prompt_chatter_count",
            "accepted_japanese_leakage_count",
            "accepted_prompt_fragment_count",
            "accepted_schema_fragment_count",
            "source_text_mutation_count",
            "model_error_count",
            "low_category_count",
        )
    )


def contains_japanese(text: str) -> bool:
    return any(
        0x3040 <= ord(char) <= 0x30FF
        or 0x31F0 <= ord(char) <= 0x31FF
        or 0x3400 <= ord(char) <= 0x4DBF
        or 0x4E00 <= ord(char) <= 0x9FFF
        for char in str(text)
    )


def contains_prompt_chatter(text: str) -> bool:
    return bool(
        re.search(
            r"\b(please provide|could you please|i don't see|i do not see|i don't understand|"
            r"i do not understand|as an ai|i cannot|i can't|feel free to ask|once i have it|"
            r"ready to translate|provide more context)\b",
            str(text),
            flags=re.IGNORECASE,
        )
    )


def contains_prompt_fragment(text: str) -> bool:
    lower = " ".join(str(text).lower().split())
    return any(
        fragment in lower
        for fragment in (
            "translate the following japanese text",
            "return only the translation",
            "return english translation",
            "japanese:",
            "english:",
            "<|im_start|>",
            "<|im_end|>",
        )
    )


def contains_schema_fragment(text: str) -> bool:
    lower = str(text).lower()
    return "json" in lower or "schema" in lower or bool(re.search(r"\{\s*\"(?:translation|source|text)\"", text))


def forbidden_pattern_present(output: str, patterns: Any) -> bool:
    if not isinstance(patterns, list):
        return False
    return any(str(pattern) and re.search(re.escape(str(pattern)), output, flags=re.IGNORECASE) for pattern in patterns)


def output_too_long_for_case(case: dict[str, Any], output: str, reference: dict[str, Any]) -> bool:
    max_chars = int(reference.get("max_chars", 0) or 0)
    max_words = int(reference.get("max_words", 0) or 0)
    if max_chars and len(output) > max_chars:
        return True
    if max_words and len(re.findall(r"[A-Za-z']+", output)) > max_words:
        return True
    source_type = str(case.get("source_type", ""))
    if source_type in {"ellipsis_fragment", "short_fragment", "punctuation_fragment", "interrupted_speech", "sfx"}:
        return len(output) > 60 or len(re.findall(r"[A-Za-z']+", output)) > 10
    return False


def looks_repetitive(output: str) -> bool:
    words = re.findall(r"[A-Za-z']+", output.lower())
    if len(words) < 8:
        return False
    most_common = max((words.count(word) for word in set(words)), default=0)
    return most_common >= 5 or most_common / max(1, len(words)) >= 0.55


def looks_verbose(case: dict[str, Any], output: str, reference: dict[str, Any]) -> bool:
    source = str(case.get("source_text", ""))
    if len(source) <= 8 and len(output) > 80:
        return True
    return len(output) > max(180, int(reference.get("max_chars", 0) or 0) * 2)
