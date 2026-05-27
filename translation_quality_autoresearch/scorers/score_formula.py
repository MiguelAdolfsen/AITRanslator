from __future__ import annotations

from pathlib import Path
from typing import Any

from translation_quality_autoresearch.common.io_utils import load_json

DEFAULT_WEIGHTS: dict[str, float] = {
    "hard_fail_penalty": 10000.0,
    "assistant_chatter_penalty": 2000.0,
    "japanese_leakage_penalty": 2000.0,
    "empty_output_penalty": 2000.0,
    "source_copied_penalty": 1600.0,
    "forbidden_pattern_penalty": 1200.0,
    "glossary_violation_penalty": 1400.0,
    "line_mapping_error_penalty": 1000.0,
    "oververbose_penalty": 120.0,
    "deterministic_warning_penalty": 25.0,
    "mqm_critical_penalty": 2500.0,
    "mqm_major_penalty": 500.0,
    "mqm_minor_penalty": 80.0,
    "reference_miss_penalty": 220.0,
    "style_length_penalty": 35.0,
    "style_punctuation_penalty": 20.0,
    "cost_weight": 0.25,
    "latency_ms_weight": 0.0001,
    "run_hard_failure_penalty": 2500.0,
    "run_average_case_weight": 1.0,
}


def load_weights(path: str | Path | None = None) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if path:
        loaded = load_json(path, default={}) or {}
    else:
        loaded = load_json(Path(__file__).resolve().parents[1] / "config" / "scoring_weights.json", default={}) or {}
    for key, value in loaded.items():
        try:
            weights[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return weights


def candidate_score(score_record: dict[str, Any], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    deterministic = score_record.get("deterministic") or {}
    mqm = score_record.get("mqm") or {}
    style = score_record.get("style") or {}
    mt_metrics = score_record.get("mt_metrics") or {}
    score = 0.0
    if deterministic.get("hard_fail"):
        score += weights["hard_fail_penalty"]
    for key, weight_key in [
        ("assistant_chatter", "assistant_chatter_penalty"),
        ("japanese_leakage", "japanese_leakage_penalty"),
        ("empty_output", "empty_output_penalty"),
        ("source_copied", "source_copied_penalty"),
        ("forbidden_pattern", "forbidden_pattern_penalty"),
        ("glossary_violation", "glossary_violation_penalty"),
        ("line_mapping_error", "line_mapping_error_penalty"),
    ]:
        if deterministic.get(key):
            score += weights[weight_key]
    if deterministic.get("oververbose"):
        score += weights["oververbose_penalty"]
    score += len(deterministic.get("warnings") or []) * weights["deterministic_warning_penalty"]
    score += int(mqm.get("critical_errors") or 0) * weights["mqm_critical_penalty"]
    score += int(mqm.get("major_errors") or 0) * weights["mqm_major_penalty"]
    score += int(mqm.get("minor_errors") or 0) * weights["mqm_minor_penalty"]
    reference_overlap = mt_metrics.get("reference_token_f1")
    if reference_overlap is not None:
        score += (1.0 - max(0.0, min(1.0, float(reference_overlap)))) * weights["reference_miss_penalty"]
    length_ratio = float(style.get("length_ratio") or 1.0)
    if length_ratio > 2.2:
        score += (length_ratio - 2.2) * weights["style_length_penalty"]
    if style.get("punctuation_ok") is False:
        score += weights["style_punctuation_penalty"]
    score += float(score_record.get("cost_proxy") or 0.0) * weights["cost_weight"]
    score += float(score_record.get("latency_ms") or 0.0) * weights["latency_ms_weight"]
    return round(score, 6)


def run_score(case_decisions: list[dict[str, Any]], candidate_scores: list[dict[str, Any]], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    if not case_decisions:
        return weights["hard_fail_penalty"]
    avg_case = sum(float(row.get("case_score") or 0.0) for row in case_decisions) / len(case_decisions)
    hard_failures = sum(1 for row in case_decisions if row.get("hard_failures"))
    critical = sum(int((row.get("mqm") or {}).get("critical_errors") or 0) for row in candidate_scores)
    return round(avg_case * weights["run_average_case_weight"] + hard_failures * weights["run_hard_failure_penalty"] + critical * weights["mqm_critical_penalty"], 6)
