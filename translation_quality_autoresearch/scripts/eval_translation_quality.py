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
    weights = load_weights(args.weights)
    glossary = load_json(args.glossary, default={}) if args.glossary else {}
    cases = load_cases(args.cases, tag=args.tag, case_id=args.case_id, limit=args.limit)
    benchmark_metadata = load_benchmark_metadata(args.cases)
    references = {row.case_id: row for row in (ReferenceRecord.from_dict(item) for item in load_jsonl(args.references))}
    candidate_map, source_hashes = load_candidate_source(args)
    validate_candidate_source(cases, candidate_map, source_hashes)

    if args.output.exists():
        if not args.overwrite_output:
            raise RuntimeError(f"output already exists: {args.output}; pass --overwrite-output")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
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
        decision = {
            "case_id": case.case_id,
            "source_hash": case.source_hash,
            "selected_candidate_id": selected["candidate_id"],
            "selected_agent": selected["agent"],
            "selected_text": selected["text"],
            "selection_reason": "lowest candidate score; no hard warnings" if not selected.get("hard_reject") else "all candidates hard-rejected; lowest score selected for inspection",
            "rejected_candidates": rejected,
            "case_score": selected["candidate_quality_score"],
            "hard_failures": hard_failures,
            "warnings": selected.get("deterministic", {}).get("warnings", []),
        }
        decisions.append(apply_human_gold_gate(decision, args, case))

    summary = build_summary(args, cases, candidate_rows, decisions, weights, benchmark_metadata)
    write_jsonl(args.output / "candidate_scores.jsonl", candidate_rows)
    write_jsonl(args.output / "case_decisions.jsonl", decisions)
    write_json(args.output / "summary.json", summary)
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


def load_candidate_source(args: argparse.Namespace) -> tuple[dict[str, list[Candidate]], dict[str, str]]:
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
    for row in rows:
        case_id = str(row.get("case_id") or "")
        source_hashes[case_id] = str(row.get("source_hash") or "")
        for payload in row.get("candidates") or []:
            if isinstance(payload, dict) and payload.get("candidate_id") and payload.get("agent") is not None:
                result[case_id].append(Candidate.from_dict(payload))
    return dict(result), source_hashes


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
        "case_score": weights["hard_fail_penalty"],
        "hard_failures": ["no_candidates"],
        "warnings": ["no_candidates"],
    }


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
    return {
        "run_id": args.run_id,
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
        "selected_agents": count_values(row.get("selected_agent") for row in decisions),
        "tag_slices": tag_slices,
        "notes": args.notes,
    }


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


if __name__ == "__main__":
    raise SystemExit(main())
