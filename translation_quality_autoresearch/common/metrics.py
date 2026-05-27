from __future__ import annotations

import re
from typing import Any

from .text_normalize import words


def token_f1(candidate: str, references: list[str]) -> float:
    cand = words(candidate)
    if not cand or not references:
        return 0.0
    best = 0.0
    cand_counts = counts(cand)
    for ref in references:
        ref_words = words(ref)
        if not ref_words:
            continue
        ref_counts = counts(ref_words)
        overlap = sum(min(cand_counts.get(token, 0), ref_counts.get(token, 0)) for token in cand_counts)
        precision = overlap / max(1, len(cand))
        recall = overlap / max(1, len(ref_words))
        if precision + recall:
            best = max(best, 2 * precision * recall / (precision + recall))
    return round(best, 6)


def counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def rank_candidates(scores: list[dict[str, Any]], *, baseline_agent: str = "baseline", tolerance: float = 0.001) -> list[dict[str, Any]]:
    if not scores:
        return []
    usable = [row for row in scores if not row.get("hard_reject")]
    pool = usable if usable else scores

    def key(row: dict[str, Any]) -> tuple[float, int, int, int, int, float, float, int]:
        deterministic = row.get("deterministic") or {}
        mqm = row.get("mqm") or {}
        style = row.get("style") or {}
        glossary_bad = 1 if deterministic.get("glossary_violation") else 0
        natural_words = len(re.findall(r"[A-Za-z0-9']+", str(row.get("text") or "")))
        return (
            float(row.get("candidate_quality_score") or 0.0),
            int(mqm.get("critical_errors") or 0),
            int(mqm.get("major_errors") or 0),
            len(deterministic.get("warnings") or []),
            glossary_bad,
            abs(natural_words - 12),
            float(row.get("cost_proxy") or 0.0),
            0 if row.get("agent") == baseline_agent else 1,
        )

    ranked = sorted(pool, key=key)
    if len(ranked) > 1 and ranked[1].get("agent") == baseline_agent:
        first = float(ranked[0].get("candidate_quality_score") or 0.0)
        second = float(ranked[1].get("candidate_quality_score") or 0.0)
        if second <= first + tolerance:
            ranked[0], ranked[1] = ranked[1], ranked[0]
    return ranked


def rejected_candidate_reason(row: dict[str, Any]) -> str:
    deterministic = row.get("deterministic") or {}
    if row.get("hard_reject"):
        flags = [key for key, value in deterministic.items() if value is True and key != "hard_fail"]
        return "hard reject: " + ", ".join(flags or ["hard_fail"])
    return "higher candidate score"
