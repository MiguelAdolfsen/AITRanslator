from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_quality_autoresearch.common.io_utils import load_json, load_jsonl, load_jsonl_many, write_json, write_jsonl
from translation_quality_autoresearch.common.metrics import rank_candidates, rejected_candidate_reason, token_f1
from translation_quality_autoresearch.common.result_log import append_result_row
from translation_quality_autoresearch.common.schemas import Candidate, ReferenceRecord, TranslationCase
from translation_quality_autoresearch.scorers.comet_scorer import score_comet_disabled
from translation_quality_autoresearch.scorers.deterministic_checks import check_candidate
from translation_quality_autoresearch.scorers.manga_style_checks import style_metrics
from translation_quality_autoresearch.scorers.mqm_judge import judge_mqm_none
from translation_quality_autoresearch.scorers.score_formula import candidate_score, load_weights, run_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate translation-agent quality from frozen outputs or live traces.")
    parser.add_argument("--mode", choices=["replay", "live-trace"], required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--frozen-outputs", type=Path)
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--human-gold", type=Path)
    parser.add_argument("--config", type=Path, default=Path("translation_quality_autoresearch/config/default_eval.json"))
    parser.add_argument("--weights", type=Path, default=Path("translation_quality_autoresearch/config/scoring_weights.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag")
    parser.add_argument("--case-id")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--append-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mark-kept", action="store_true")
    parser.add_argument("--mqm-judge-provider", choices=["none", "local_qwen", "openai_optional"], default=None)
    parser.add_argument("--pairwise-judge-provider", choices=["none", "local_qwen", "openai_optional"], default=None)
    parser.add_argument("--enable-comet", action="store_true")
    parser.add_argument("--enable-cometkiwi", action="store_true")
    parser.add_argument("--enable-xcomet", action="store_true")
    parser.add_argument("--allow-metric-downloads", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_eval(args)
    except Exception as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


def run_eval(args: argparse.Namespace) -> None:
    config = load_json(args.config, default={}) or {}
    if args.mqm_judge_provider is not None:
        config["mqm_judge_provider"] = args.mqm_judge_provider
    if args.pairwise_judge_provider is not None:
        config["pairwise_judge_provider"] = args.pairwise_judge_provider
    config["enable_comet"] = bool(args.enable_comet or config.get("enable_comet"))
    config["enable_cometkiwi"] = bool(args.enable_cometkiwi or config.get("enable_cometkiwi"))
    config["enable_xcomet"] = bool(args.enable_xcomet or config.get("enable_xcomet"))
    config["allow_metric_downloads"] = bool(args.allow_metric_downloads or config.get("allow_metric_downloads"))
    weights = load_weights(args.weights)
    glossary = load_json(args.glossary, default={}) if args.glossary else {}
    cases = load_cases(args.cases, tag=args.tag, case_id=args.case_id, limit=args.limit)
    benchmark_metadata = load_benchmark_metadata(args.cases)
    references = {row.case_id: row for row in (ReferenceRecord.from_dict(item) for item in load_jsonl(args.references))}
    validate_reference_coverage(cases, references)
    candidate_map, source_hashes, source_rows = load_candidate_source(args)
    validate_candidate_source(cases, candidate_map, source_hashes)

    if args.output.exists():
        if not args.overwrite_output:
            raise RuntimeError(f"output already exists: {args.output}; pass --overwrite-output")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "artifacts").mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for case in cases:
        case_candidates = candidate_map.get(case.case_id, [])
        scored = [
            score_one_candidate(case, candidate, references.get(case.case_id), glossary, config, weights)
            for candidate in case_candidates
        ]
        candidate_rows.extend(scored)
        ranked = rank_candidates(scored, baseline_agent=str(config.get("baseline_agent") or "baseline"), tolerance=float(config.get("selection_tolerance", 0.001)))
        if not ranked:
            decisions.append(no_candidate_decision(case, weights))
            continue
        selected = ranked[0]
        rejected = [{"candidate_id": row["candidate_id"], "reason": rejected_candidate_reason(row)} for row in ranked[1:]]
        hard_failures = selected.get("deterministic", {}).get("warnings", []) if selected.get("hard_reject") else []
        pairwise_baseline = baseline_pairwise(selected, scored, str(config.get("baseline_agent") or "baseline"))
        decision = {
            "case_id": case.case_id,
            "source_hash": case.source_hash,
            "selected_candidate_id": selected["candidate_id"],
            "selected_agent": selected["agent"],
            "selected_text": selected["text"],
            "selection_reason": "lowest candidate score; no hard warnings" if not selected.get("hard_reject") else "all candidates hard-rejected; lowest score selected for inspection",
            "rejected_candidates": rejected,
            "pairwise_baseline": pairwise_baseline,
            "case_score": selected["candidate_quality_score"],
            "hard_failures": hard_failures,
            "warnings": selected.get("deterministic", {}).get("warnings", []),
        }
        decision = apply_human_gold_gate(decision, args, case)
        decisions.append(decision)
        traces.append(build_case_trace(case, source_rows.get(case.case_id, {}), scored, decision))

    summary = build_summary(args, cases, candidate_rows, decisions, weights, benchmark_metadata, config)
    write_jsonl(args.output / "candidate_scores.jsonl", candidate_rows)
    write_jsonl(args.output / "case_decisions.jsonl", decisions)
    write_jsonl(args.output / "case_results.jsonl", decisions)
    write_jsonl(args.output / "traces.jsonl", traces)
    write_jsonl(args.output / "failures.jsonl", failure_rows(candidate_rows, decisions))
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "artifacts" / "run_manifest.json", run_manifest(args, summary, config))
    write_summary_md(args.output / "summary.md", summary)
    write_failure_table(args.output / "failure_table.tsv", candidate_rows, decisions)
    if args.append_results:
        append_result_row(args.results, result_row(summary, args))
    print(f"translation_quality_score={summary['translation_quality_score']} cases={summary['case_count']} output={args.output}")


def load_cases(value: str, *, tag: str | None, case_id: str | None, limit: int | None) -> list[TranslationCase]:
    cases = [TranslationCase.from_dict(row) for row in load_jsonl_many(value)]
    if tag:
        cases = [case for case in cases if tag in case.tags]
    if case_id:
        cases = [case for case in cases if case.case_id == case_id]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise RuntimeError("no cases selected")
    errors = [error for case in cases for error in case.validate()]
    if errors:
        raise RuntimeError("; ".join(errors))
    return cases


def load_candidate_source(args: argparse.Namespace) -> tuple[dict[str, list[Candidate]], dict[str, str], dict[str, dict[str, Any]]]:
    if args.mode == "replay":
        if not args.frozen_outputs:
            raise RuntimeError("--frozen-outputs is required in replay mode")
        rows = load_jsonl(args.frozen_outputs)
    else:
        if not args.traces:
            raise RuntimeError("--traces is required in live-trace mode")
        rows = load_jsonl(args.traces)
    result: dict[str, list[Candidate]] = defaultdict(list)
    source_hashes: dict[str, str] = {}
    source_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        source_hashes[case_id] = str(row.get("source_hash") or "")
        source_rows[case_id] = row
        for payload in row.get("candidates") or []:
            if isinstance(payload, dict) and payload.get("candidate_id") and payload.get("agent") is not None:
                result[case_id].append(Candidate.from_dict(payload))
    return dict(result), source_hashes, source_rows


def validate_candidate_source(
    cases: list[TranslationCase],
    candidate_map: dict[str, list[Candidate]],
    source_hashes: dict[str, str],
) -> None:
    missing = [case.case_id for case in cases if case.case_id not in candidate_map]
    if missing:
        raise RuntimeError(f"missing candidates for case(s): {', '.join(missing[:10])}")
    mismatched = [
        case.case_id
        for case in cases
        if source_hashes.get(case.case_id, "") != case.source_hash
    ]
    if mismatched:
        raise RuntimeError(f"candidate source_hash mismatch for case(s): {', '.join(mismatched[:10])}")


def validate_reference_coverage(cases: list[TranslationCase], references: dict[str, ReferenceRecord]) -> None:
    missing = [case.case_id for case in cases if case.case_id not in references]
    if missing:
        raise RuntimeError(f"missing references for case(s): {', '.join(missing[:10])}")
    selected_case_ids = {case.case_id for case in cases}
    empty = [case_id for case_id, row in references.items() if case_id in selected_case_ids and not row.reference_translations]
    if empty:
        raise RuntimeError(f"empty reference_translations for case(s): {', '.join(empty[:10])}")


def benchmark_root_from_cases(value: str | Path) -> Path | None:
    first = str(value).split(",", 1)[0].strip()
    if not first:
        return None
    return Path(first).parent


def load_benchmark_metadata(cases_value: str | Path) -> dict[str, Any]:
    root = benchmark_root_from_cases(cases_value)
    version_path = root / "VERSION.json" if root else None
    metadata = load_json(version_path, default={}) if version_path else {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "benchmark_id": str(metadata.get("benchmark_id") or (root.name if root else "unknown")),
        "reference_policy": str(metadata.get("reference_policy") or "seed_synthetic"),
        "benchmark_version_path": str(version_path) if version_path and version_path.exists() else "",
    }


def score_one_candidate(
    case: TranslationCase,
    candidate: Candidate,
    reference: ReferenceRecord | None,
    glossary: dict[str, Any] | None,
    config: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, Any]:
    deterministic = check_candidate(case, candidate, glossary=glossary, config=config)
    mqm = judge_mqm_none()
    mt_metrics = score_comet_disabled()
    mt_metrics["reference_token_f1"] = token_f1(candidate.text, reference.reference_translations if reference else [])
    style = style_metrics(case, candidate.text, max_target_words_default=int(config.get("max_target_words_default", 28)))
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "candidate_id": candidate.candidate_id,
        "agent": candidate.agent,
        "text": candidate.text,
        "deterministic": deterministic,
        "mqm": mqm,
        "mt_metrics": mt_metrics,
        "style": style,
        "cost_proxy": candidate.cost_proxy,
        "latency_ms": candidate.latency_ms,
        "candidate_quality_score": 0.0,
        "hard_reject": bool(deterministic.get("hard_fail")),
    }
    row["candidate_quality_score"] = candidate_score(row, weights)
    return row


def no_candidate_decision(case: TranslationCase, weights: dict[str, float]) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "source_hash": case.source_hash,
        "selected_candidate_id": "",
        "selected_agent": "",
        "selected_text": "",
        "selection_reason": "no candidates",
        "rejected_candidates": [],
        "pairwise_baseline": {"enabled": False, "reason": "no selected candidate"},
        "case_score": weights["hard_fail_penalty"],
        "hard_failures": ["no_candidates"],
        "warnings": ["no_candidates"],
    }


def baseline_pairwise(selected: dict[str, Any], scored: list[dict[str, Any]], baseline_agent: str) -> dict[str, Any]:
    baseline_candidates = [row for row in scored if row.get("agent") == baseline_agent]
    if not baseline_candidates:
        return {"enabled": False, "reason": f"baseline agent not present: {baseline_agent}"}
    baseline = sorted(baseline_candidates, key=lambda row: float(row.get("candidate_quality_score") or 0.0))[0]
    selected_score = float(selected.get("candidate_quality_score") or 0.0)
    baseline_score = float(baseline.get("candidate_quality_score") or 0.0)
    delta = round(selected_score - baseline_score, 6)
    if selected.get("candidate_id") == baseline.get("candidate_id"):
        outcome = "selected_baseline"
    elif delta < 0:
        outcome = "selected_better_than_baseline"
    elif delta > 0:
        outcome = "selected_worse_than_baseline"
    else:
        outcome = "tie_with_baseline"
    return {
        "enabled": True,
        "judge_provider": "deterministic_score_delta",
        "baseline_agent": baseline_agent,
        "baseline_candidate_id": baseline.get("candidate_id"),
        "baseline_text": baseline.get("text"),
        "baseline_score": baseline_score,
        "selected_score": selected_score,
        "score_delta_selected_minus_baseline": delta,
        "outcome": outcome,
    }


def build_case_trace(
    case: TranslationCase,
    source_row: dict[str, Any],
    scored: list[dict[str, Any]],
    decision: dict[str, Any],
) -> dict[str, Any]:
    candidate_translations = []
    for row in scored:
        candidate_translations.append(
            {
                "candidate_id": row.get("candidate_id"),
                "agent": row.get("agent"),
                "text": row.get("text"),
                "candidate_quality_score": row.get("candidate_quality_score"),
                "hard_reject": row.get("hard_reject"),
                "deterministic": row.get("deterministic"),
                "mqm": row.get("mqm"),
                "mt_metrics": row.get("mt_metrics"),
                "style": row.get("style"),
                "cost_proxy": row.get("cost_proxy"),
                "latency_ms": row.get("latency_ms"),
            }
        )
    return {
        "case_id": case.case_id,
        "source_hash": case.source_hash,
        "source_text": case.source_text,
        "context_before": case.context_before,
        "context_after": case.context_after,
        "glossary_terms": case.glossary_terms,
        "candidate_translations": candidate_translations,
        "agent_trace": {
            key: value
            for key, value in source_row.items()
            if key not in {"candidates"}
        },
        "critic_repair_decisions": source_row.get("critic_repair_decisions") or [],
        "reranker_scores": [
            {
                "candidate_id": row.get("candidate_id"),
                "candidate_quality_score": row.get("candidate_quality_score"),
                "hard_reject": row.get("hard_reject"),
            }
            for row in scored
        ],
        "final_accepted_translation": decision.get("selected_text"),
        "final_candidate_id": decision.get("selected_candidate_id"),
        "final_agent": decision.get("selected_agent"),
        "reason_final_candidate_was_selected": decision.get("selection_reason"),
        "rejected_candidates": decision.get("rejected_candidates") or [],
        "pairwise_baseline": decision.get("pairwise_baseline") or {},
        "hard_failures": decision.get("hard_failures") or [],
        "warnings": decision.get("warnings") or [],
    }


def failure_rows(candidate_rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.get("hard_failures") or decision.get("warnings"):
            failures.append(
                {
                    "case_id": decision.get("case_id"),
                    "scope": "selected_decision",
                    "selected_candidate_id": decision.get("selected_candidate_id"),
                    "hard_failures": decision.get("hard_failures") or [],
                    "warnings": decision.get("warnings") or [],
                    "selected_text": decision.get("selected_text"),
                }
            )
    for row in candidate_rows:
        deterministic = row.get("deterministic") or {}
        if row.get("hard_reject") or deterministic.get("warnings"):
            failures.append(
                {
                    "case_id": row.get("case_id"),
                    "scope": "candidate",
                    "candidate_id": row.get("candidate_id"),
                    "agent": row.get("agent"),
                    "hard_reject": row.get("hard_reject"),
                    "warnings": deterministic.get("warnings") or [],
                    "deterministic": deterministic,
                    "text": row.get("text"),
                }
            )
    return failures


def apply_human_gold_gate(decision: dict[str, Any], args: argparse.Namespace, case: TranslationCase) -> dict[str, Any]:
    gold_path = args.human_gold
    if gold_path is None:
        root = benchmark_root_from_cases(args.cases)
        gold_path = (root / "human_gold.jsonl") if root else Path("translation_quality_autoresearch/benchmark/human_gold.jsonl")
    if not gold_path.exists():
        return decision
    for row in load_jsonl(gold_path):
        if str(row.get("case_id") or "") != case.case_id:
            continue
        gold = row.get("gold_rating") or {}
        selected = str(decision.get("selected_text") or "")
        forbidden = [str(value) for value in gold.get("forbid") or []]
        exact = [str(value) for value in gold.get("must_select_exact") or []]
        one_of = [str(value) for value in gold.get("must_select_one_of") or []]
        failed = any(value and value.lower() in selected.lower() for value in forbidden)
        if exact and selected not in exact:
            failed = True
        if one_of and not any(selected.lower() == value.lower() for value in one_of):
            failed = True
        if failed and str(row.get("severity_if_failed") or "") == "critical":
            decision.setdefault("hard_failures", []).append("human_gold_critical")
            decision.setdefault("warnings", []).append("human_gold_critical")
    return decision


def build_summary(
    args: argparse.Namespace,
    cases: list[TranslationCase],
    candidate_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    weights: dict[str, float],
    benchmark_metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    deterministic_rows = [row.get("deterministic") or {} for row in candidate_rows]
    tag_slices: dict[str, dict[str, Any]] = {}
    case_by_id = {case.case_id: case for case in cases}
    for tag in sorted({tag for case in cases for tag in case.tags}):
        tagged = [row for row in decisions if tag in case_by_id[row["case_id"]].tags]
        if tagged:
            tag_slices[tag] = {
                "case_count": len(tagged),
                "average_case_score": round(sum(float(row.get("case_score") or 0.0) for row in tagged) / len(tagged), 6),
                "hard_failure_count": sum(1 for row in tagged if row.get("hard_failures")),
            }
    score = run_score(decisions, candidate_rows, weights)
    hard_failure_count = sum(1 for row in decisions if row.get("hard_failures"))
    pairwise_rows = [row.get("pairwise_baseline") or {} for row in decisions]
    pairwise_enabled = [row for row in pairwise_rows if row.get("enabled")]
    return {
        "run_id": args.run_id,
        "mode": args.mode,
        "benchmark_id": benchmark_metadata["benchmark_id"],
        "reference_policy": benchmark_metadata["reference_policy"],
        "benchmark_version_path": benchmark_metadata["benchmark_version_path"],
        "experiment_name": args.experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "translation_quality_score": score,
        "case_count": len(cases),
        "candidate_count": len(candidate_rows),
        "hard_reject": hard_failure_count > 0,
        "hard_failure_count": hard_failure_count,
        "hard_failure_rate": round(hard_failure_count / max(1, len(cases)), 6),
        "critical_mqm_errors": sum(int((row.get("mqm") or {}).get("critical_errors") or 0) for row in candidate_rows),
        "japanese_leakage_count": sum(1 for row in deterministic_rows if row.get("japanese_leakage")),
        "assistant_chatter_count": sum(1 for row in deterministic_rows if row.get("assistant_chatter")),
        "glossary_violation_count": sum(1 for row in deterministic_rows if row.get("glossary_violation")),
        "pairwise_baseline": {
            "enabled_count": len(pairwise_enabled),
            "selected_better_than_baseline": sum(1 for row in pairwise_enabled if row.get("outcome") == "selected_better_than_baseline"),
            "selected_worse_than_baseline": sum(1 for row in pairwise_enabled if row.get("outcome") == "selected_worse_than_baseline"),
            "selected_baseline": sum(1 for row in pairwise_enabled if row.get("outcome") == "selected_baseline"),
            "tie_with_baseline": sum(1 for row in pairwise_enabled if row.get("outcome") == "tie_with_baseline"),
        },
        "scoring_layers": {
            "deterministic_hard_checks": True,
            "mt_metrics": {
                "reference_token_f1": True,
                "comet": bool(config.get("enable_comet")),
                "cometkiwi": bool(config.get("enable_cometkiwi")),
                "xcomet": bool(config.get("enable_xcomet")),
            },
            "mqm_judge_provider": str(config.get("mqm_judge_provider") or "none"),
            "pairwise_judge_provider": str(config.get("pairwise_judge_provider") or "none"),
            "human_gold_gate": human_gold_path(args).exists(),
        },
        "artifacts": {
            "summary": "summary.json",
            "case_results": "case_results.jsonl",
            "traces": "traces.jsonl",
            "failures": "failures.jsonl",
            "candidate_scores_legacy": "candidate_scores.jsonl",
            "case_decisions_legacy": "case_decisions.jsonl",
            "manifest": "artifacts/run_manifest.json",
        },
        "selected_agents": count_values(row.get("selected_agent") for row in decisions),
        "tag_slices": tag_slices,
        "notes": args.notes,
    }


def human_gold_path(args: argparse.Namespace) -> Path:
    if args.human_gold is not None:
        return args.human_gold
    root = benchmark_root_from_cases(args.cases)
    return (root / "human_gold.jsonl") if root else Path("translation_quality_autoresearch/benchmark/human_gold.jsonl")


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Translation Quality Summary",
        "",
        f"- run_id: {summary['run_id']}",
        f"- mode: {summary.get('mode', '')}",
        f"- benchmark_id: {summary.get('benchmark_id', '')}",
        f"- reference_policy: {summary.get('reference_policy', '')}",
        f"- translation_quality_score: {summary['translation_quality_score']}",
        f"- cases: {summary['case_count']}",
        f"- candidates: {summary['candidate_count']}",
        f"- hard_reject: {summary['hard_reject']}",
        f"- hard_failure_rate: {summary['hard_failure_rate']}",
        f"- japanese_leakage_count: {summary['japanese_leakage_count']}",
        f"- assistant_chatter_count: {summary['assistant_chatter_count']}",
        f"- glossary_violation_count: {summary['glossary_violation_count']}",
        f"- pairwise_baseline: {summary.get('pairwise_baseline', {})}",
        "",
        "## Tag Slices",
    ]
    for tag, row in summary.get("tag_slices", {}).items():
        lines.append(f"- {tag}: score={row['average_case_score']} cases={row['case_count']} hard_failures={row['hard_failure_count']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_table(path: Path, candidate_rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "candidate_id", "agent", "candidate_quality_score", "hard_reject", "warnings", "text"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in candidate_rows:
            deterministic = row.get("deterministic") or {}
            if not row.get("hard_reject") and not deterministic.get("warnings"):
                continue
            writer.writerow(
                {
                    "case_id": row.get("case_id"),
                    "candidate_id": row.get("candidate_id"),
                    "agent": row.get("agent"),
                    "candidate_quality_score": row.get("candidate_quality_score"),
                    "hard_reject": row.get("hard_reject"),
                    "warnings": ",".join(deterministic.get("warnings") or []),
                    "text": row.get("text"),
                }
            )


def result_row(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "timestamp": summary["timestamp"],
        "run_id": args.run_id,
        "benchmark_id": summary.get("benchmark_id", ""),
        "reference_policy": summary.get("reference_policy", ""),
        "experiment_name": args.experiment_name,
        "translation_quality_score": summary["translation_quality_score"],
        "case_count": summary["case_count"],
        "candidate_count": summary["candidate_count"],
        "hard_reject": summary["hard_reject"],
        "hard_failure_rate": summary["hard_failure_rate"],
        "critical_mqm_errors": summary["critical_mqm_errors"],
        "japanese_leakage_count": summary["japanese_leakage_count"],
        "assistant_chatter_count": summary["assistant_chatter_count"],
        "glossary_violation_count": summary["glossary_violation_count"],
        "notes": args.notes,
    }


def run_manifest(args: argparse.Namespace, summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "mode": args.mode,
        "benchmark_id": summary.get("benchmark_id", ""),
        "reference_policy": summary.get("reference_policy", ""),
        "cases": str(args.cases),
        "references": str(args.references),
        "frozen_outputs": str(args.frozen_outputs or ""),
        "traces": str(args.traces or ""),
        "glossary": str(args.glossary or ""),
        "weights": str(args.weights),
        "config": str(args.config),
        "effective_scoring_layers": summary.get("scoring_layers") or {},
        "effective_config_flags": {
            "mqm_judge_provider": config.get("mqm_judge_provider"),
            "pairwise_judge_provider": config.get("pairwise_judge_provider"),
            "enable_comet": config.get("enable_comet"),
            "enable_cometkiwi": config.get("enable_cometkiwi"),
            "enable_xcomet": config.get("enable_xcomet"),
            "allow_metric_downloads": config.get("allow_metric_downloads"),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
