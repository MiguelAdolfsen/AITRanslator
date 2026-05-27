from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from manga_local_translator.qwen_validation import extract_qwen_json_payloads
from manga_local_translator.text_filter import count_japanese_chars, suspected_bad_translation
from manga_local_translator.vision_validation import proper_noun_tokens


CRITICAL_VIOLATIONS = {
    "assistant_chatter",
    "japanese_leakage",
    "line_id_mismatch",
    "source_text_mismatch",
    "invalid_schema",
    "vision_hallucination",
    "invented_speaker_name",
    "invented_proper_noun",
}


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    main_delta: float
    holdout_delta: float
    holdout_pairwise_net: int


def deterministic_violations(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    output = str(row.get("final_translation", "") or "")
    source = str(row.get("source_text", "") or "")
    reference = str(row.get("reference_text", "") or "")
    context = row.get("translation_context") if isinstance(row.get("translation_context"), dict) else {}
    if not output.strip():
        violations.append("empty_translation")
    if assistant_chatter(output):
        violations.append("assistant_chatter")
    if count_japanese_chars(output) > 0:
        violations.append("japanese_leakage")
    if source and output.strip() == source.strip():
        violations.append("source_copied")
    if suspected_bad_translation(output):
        violations.append("suspected_bad_translation")
    if str(row.get("line_id", "")) != str(context.get("line_id", row.get("line_id", ""))):
        violations.append("line_id_mismatch")
    if context.get("vision_facts_structural_error"):
        violations.append("invalid_schema")
    if context.get("vision_facts_reject_reason") in {
        "contains_translation",
        "contains_translation_field",
        "invalid_visual_fact",
        "ungrounded_speaker",
        "ungrounded_speaker_anchor",
    }:
        violations.append("vision_hallucination")
    if invented_speaker_name(context):
        violations.append("invented_speaker_name")
    if invented_proper_nouns(output, reference, source):
        violations.append("invented_proper_noun")
    return list(dict.fromkeys(violations))


def assistant_chatter(text: str) -> bool:
    return bool(
        re.search(
            r"(\b(as an ai|i cannot|here(?:'s| is) (?:the )?translation|json)\b|\btranslation\s*:|```)",
            text,
            flags=re.IGNORECASE,
        )
    )


def invented_speaker_name(context: dict[str, Any]) -> bool:
    for key in ("speaker_anchor", "speaker_position"):
        value = str(context.get(key, "") or "")
        if not value or value.lower() in {"unknown", "narrator", "speaker", "character"}:
            continue
        if proper_noun_tokens(value):
            return True
    return False


def invented_proper_nouns(output: str, reference: str, source: str) -> bool:
    output_names = proper_noun_tokens(output)
    if not output_names:
        return False
    reference_names = proper_noun_tokens(reference)
    source_has_name_signal = bool(re.search(r"[\u30a0-\u30ff\u4e00-\u9fff\u3008\u3009\u300a\u300b\u30fb]", source))
    allowed = {name for name in output_names if name in reference_names}
    if source_has_name_signal:
        return False
    return bool(output_names - allowed)


def score_trace(
    row: dict[str, Any],
    *,
    judge=None,
    fake_judge: bool = False,
) -> dict[str, Any]:
    violations = deterministic_violations(row)
    if fake_judge or judge is None:
        judge_payload = heuristic_judge(row, violations)
    else:
        judge_payload = qwen_mqm_judge(judge, row, violations)
    quality = float(judge_payload.get("quality_score", 0.0))
    if violations:
        quality -= sum(violation_penalty(value) for value in violations)
    quality = max(0.0, min(100.0, round(quality, 3)))
    errors = list(judge_payload.get("mqm_errors", []) or [])
    critical_count = len([value for value in violations if value in CRITICAL_VIOLATIONS])
    critical_count += sum(1 for error in errors if str(error.get("severity", "")).lower() == "critical")
    return {
        "quality_score": quality,
        "deterministic_violations": violations,
        "mqm_errors": errors,
        "critical_error_count": critical_count,
        "hallucination_count": sum(1 for value in violations if value in {"vision_hallucination", "invented_speaker_name", "invented_proper_noun"}),
        "judge_reason": str(judge_payload.get("reason", "")),
    }


def heuristic_judge(row: dict[str, Any], violations: list[str]) -> dict[str, Any]:
    output = str(row.get("final_translation", "") or "")
    reference = str(row.get("reference_text", "") or "")
    overlap = lexical_overlap(output, reference)
    score = 72.0 + 24.0 * overlap
    if len(output.split()) > 28:
        score -= 4.0
    if not output.strip():
        score = 0.0
    return {
        "quality_score": round(score, 3),
        "mqm_errors": [{"category": "format", "severity": "critical", "detail": value} for value in violations if value in CRITICAL_VIOLATIONS],
        "reason": "heuristic_fake_judge",
    }


def lexical_overlap(left: str, right: str) -> float:
    left_words = content_words(left)
    right_words = content_words(right)
    if not left_words or not right_words:
        return 1.0 if left.strip().lower() == right.strip().lower() and left.strip() else 0.0
    return len(left_words & right_words) / max(1, len(right_words))


def content_words(text: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "it", "is", "are", "was", "were", "i", "you"}
    return {word for word in re.findall(r"[a-zA-Z][a-zA-Z']{1,}", text.lower()) if word not in stop}


def qwen_mqm_judge(judge, row: dict[str, Any], violations: list[str]) -> dict[str, Any]:
    prompt = build_mqm_prompt(row, violations)
    try:
        raw = judge._run_prompt(prompt, settings=judge_settings())
    except Exception as exc:
        return {"quality_score": 0.0, "mqm_errors": [{"category": "judge", "severity": "critical", "detail": str(exc)}], "reason": "judge_error"}
    for payload in extract_qwen_json_payloads(raw):
        if "quality_score" not in payload:
            continue
        errors = payload.get("mqm_errors", [])
        if not isinstance(errors, list):
            errors = []
        try:
            quality = float(payload.get("quality_score", 0.0))
        except (TypeError, ValueError):
            quality = 0.0
        return {
            "quality_score": quality,
            "mqm_errors": [error for error in errors if isinstance(error, dict)],
            "reason": str(payload.get("reason", "")),
            "raw_judge_response": raw,
        }
    return {"quality_score": 0.0, "mqm_errors": [{"category": "judge", "severity": "critical", "detail": "invalid_judge_json"}], "reason": raw[:200]}


def judge_settings():
    from manga_local_translator.qwen_types import QWEN_CRITIC_SETTINGS

    return QWEN_CRITIC_SETTINGS


def build_mqm_prompt(row: dict[str, Any], violations: list[str]) -> str:
    payload = {
        "source_text": row.get("source_text", ""),
        "reference_translation_semantic_anchor": row.get("reference_text", ""),
        "candidate_translation": row.get("final_translation", ""),
        "reference_role": row.get("reference_role", ""),
        "visual_facts_used": row.get("visual_facts_used", []),
        "deterministic_violations": violations,
    }
    return "\n".join(
        [
            "/no_think",
            "You are an MQM-style judge for Japanese-to-English manga dialogue.",
            "Score the candidate for semantic faithfulness, natural American manga English, terminology, and format.",
            "Use the reference as a semantic anchor, not as wording that must be copied.",
            "Critical errors: hallucinated facts, invented speakers/names/relationships, wrong polarity, line mismatch, untranslated Japanese, assistant chatter.",
            "Return compact JSON only: {\"quality_score\":0-100,\"mqm_errors\":[{\"category\":\"accuracy|fluency|terminology|style|format\",\"severity\":\"minor|major|critical\",\"detail\":\"...\"}],\"reason\":\"...\"}",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "JSON:",
        ]
    )


def pairwise_compare(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
    *,
    judge=None,
    fake_judge: bool = False,
) -> dict[str, Any]:
    if fake_judge or judge is None:
        base_score = float(baseline_row["score"]["quality_score"])
        candidate_score = float(candidate_row["score"]["quality_score"])
        if abs(base_score - candidate_score) < 0.001:
            winner = "tie"
        else:
            winner = "candidate" if candidate_score > base_score else "baseline"
        return {"winner": winner, "reason": "score_based_fake_pairwise"}
    prompt = build_pairwise_prompt(baseline_row, candidate_row)
    try:
        raw = judge._run_prompt(prompt, settings=judge_settings())
    except Exception as exc:
        return {"winner": "baseline", "reason": f"pairwise_judge_error:{exc}"}
    for payload in extract_qwen_json_payloads(raw):
        winner = str(payload.get("winner", "")).lower()
        if winner in {"baseline", "candidate", "tie"}:
            return {"winner": winner, "reason": str(payload.get("reason", "")), "raw_pairwise_response": raw}
    return {"winner": "baseline", "reason": "invalid_pairwise_json"}


def build_pairwise_prompt(baseline_row: dict[str, Any], candidate_row: dict[str, Any]) -> str:
    payload = {
        "source_text": candidate_row.get("source_text", ""),
        "reference_translation_semantic_anchor": candidate_row.get("reference_text", ""),
        "baseline_translation": baseline_row.get("final_translation", ""),
        "candidate_translation": candidate_row.get("final_translation", ""),
        "candidate_visual_facts": candidate_row.get("visual_facts_used", []),
    }
    return "\n".join(
        [
            "/no_think",
            "Choose the better English manga translation. Faithfulness beats fluency. Hallucination loses.",
            "Use the reference as a semantic anchor, not exact wording.",
            "Return compact JSON only: {\"winner\":\"baseline|candidate|tie\",\"reason\":\"...\"}",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "JSON:",
        ]
    )


def violation_penalty(violation: str) -> float:
    if violation in CRITICAL_VIOLATIONS:
        return 35.0
    if violation in {"source_copied", "empty_translation", "suspected_bad_translation"}:
        return 20.0
    return 8.0


def summarize_case_results(rows: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    split_rows = [row for row in rows if row.get("split") == split and not row.get("skip_reference")]
    if not split_rows:
        return {
            "split": split,
            "scored_count": 0,
            "quality_score": 0.0,
            "baseline_quality_score": 0.0,
            "critical_error_count": 0,
            "baseline_critical_error_count": 0,
            "hallucination_count": 0,
            "baseline_hallucination_count": 0,
            "pairwise_candidate_wins": 0,
            "pairwise_baseline_wins": 0,
            "pairwise_ties": 0,
            "pairwise_net": 0,
        }
    candidate_wins = sum(1 for row in split_rows if row.get("pairwise_winner") == "candidate")
    baseline_wins = sum(1 for row in split_rows if row.get("pairwise_winner") == "baseline")
    return {
        "split": split,
        "scored_count": len(split_rows),
        "quality_score": round(sum(float(row["candidate_score"]["quality_score"]) for row in split_rows) / len(split_rows), 3),
        "baseline_quality_score": round(sum(float(row["baseline_score"]["quality_score"]) for row in split_rows) / len(split_rows), 3),
        "critical_error_count": sum(int(row["candidate_score"]["critical_error_count"]) for row in split_rows),
        "baseline_critical_error_count": sum(int(row["baseline_score"]["critical_error_count"]) for row in split_rows),
        "hallucination_count": sum(int(row["candidate_score"]["hallucination_count"]) for row in split_rows),
        "baseline_hallucination_count": sum(int(row["baseline_score"]["hallucination_count"]) for row in split_rows),
        "pairwise_candidate_wins": candidate_wins,
        "pairwise_baseline_wins": baseline_wins,
        "pairwise_ties": sum(1 for row in split_rows if row.get("pairwise_winner") == "tie"),
        "pairwise_net": candidate_wins - baseline_wins,
    }


def promotion_decision(main: dict[str, Any], holdout: dict[str, Any]) -> PromotionDecision:
    main_delta = round(float(main["quality_score"]) - float(main["baseline_quality_score"]), 3)
    holdout_delta = round(float(holdout["quality_score"]) - float(holdout["baseline_quality_score"]), 3)
    holdout_pairwise_net = int(holdout.get("pairwise_net", 0))
    if main_delta < 0.5:
        return PromotionDecision(False, "main_quality_delta_below_0.5", main_delta, holdout_delta, holdout_pairwise_net)
    if int(main["critical_error_count"]) > int(main["baseline_critical_error_count"]):
        return PromotionDecision(False, "main_critical_errors_regressed", main_delta, holdout_delta, holdout_pairwise_net)
    if int(main["hallucination_count"]) > int(main["baseline_hallucination_count"]):
        return PromotionDecision(False, "main_hallucinations_regressed", main_delta, holdout_delta, holdout_pairwise_net)
    if holdout_delta < -0.25:
        return PromotionDecision(False, "holdout_quality_regressed", main_delta, holdout_delta, holdout_pairwise_net)
    if holdout_pairwise_net < 0:
        return PromotionDecision(False, "holdout_pairwise_net_negative", main_delta, holdout_delta, holdout_pairwise_net)
    if int(holdout["critical_error_count"]) > int(holdout["baseline_critical_error_count"]):
        return PromotionDecision(False, "holdout_critical_errors_regressed", main_delta, holdout_delta, holdout_pairwise_net)
    if int(holdout["hallucination_count"]) > int(holdout["baseline_hallucination_count"]):
        return PromotionDecision(False, "holdout_hallucinations_regressed", main_delta, holdout_delta, holdout_pairwise_net)
    return PromotionDecision(True, "promotion_gates_passed", main_delta, holdout_delta, holdout_pairwise_net)
