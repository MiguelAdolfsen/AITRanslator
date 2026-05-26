from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from . import io_adapters
    from . import score
    from .validate_fixtures import validate_benchmark
except ImportError:  # pragma: no cover - direct script execution path
    import io_adapters
    import score
    from validate_fixtures import validate_benchmark


RESULT_HEADER = [
    "run_id",
    "timestamp",
    "commit",
    "parent_commit",
    "experiment_name",
    "changed_files",
    "fixture_validation_ok",
    "benchmark_set",
    "cat_response_score",
    "cat_quality_score",
    "cat_latency_score",
    "baseline_score",
    "delta_score",
    "cases_total",
    "repeat_count",
    "evaluations_total",
    "cat_approved_count",
    "cat_approval_rate",
    "retried_count",
    "retry_rate",
    "primary_rejected_count",
    "primary_rejected_rate",
    "retry_rescued_accept_count",
    "retry_wasted_safe_reject_count",
    "retry_failed_count",
    "clean_primary_accept_count",
    "expected_accept_count",
    "expected_reject_count",
    "accepted_count",
    "rejected_count",
    "clean_accept_count",
    "safe_reject_count",
    "false_reject_count",
    "unsafe_accept_count",
    "weak_accept_count",
    "unclear_fail_count",
    "category_min_approval_rate",
    "low_category_count",
    "low_categories",
    "accepted_prompt_chatter_count",
    "accepted_japanese_leakage_count",
    "accepted_prompt_fragment_count",
    "accepted_schema_fragment_count",
    "source_text_mutation_count",
    "accepted_forbidden_pattern_count",
    "accepted_overlong_fragment_count",
    "accepted_repetitive_count",
    "accepted_explanatory_output_count",
    "empty_output_count",
    "verbose_output_count",
    "model_error_count",
    "mean_latency_ms",
    "hard_failure",
    "kept",
    "notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def git_value(args: list[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=10)
    except Exception:
        return "UNKNOWN"
    if completed.returncode != 0:
        return "UNKNOWN"
    return completed.stdout.strip() or "UNKNOWN"


def changed_files() -> str:
    try:
        completed = subprocess.run(["git", "status", "--porcelain"], text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=10)
    except Exception:
        return "UNKNOWN"
    if completed.returncode != 0:
        return "UNKNOWN"
    return ";".join(line[3:].strip() for line in completed.stdout.splitlines() if len(line) > 3)


def benchmark_fingerprint(benchmark: Path) -> str:
    digest = hashlib.sha256()
    for name in ("cases.jsonl", "references.jsonl"):
        path = benchmark / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_results_header(results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    expected = "\t".join(RESULT_HEADER)
    if not results_path.exists() or results_path.stat().st_size == 0:
        results_path.write_text(expected + "\n", encoding="utf-8")
        return
    first = results_path.read_text(encoding="utf-8-sig").splitlines()[0]
    if first != expected:
        raise RuntimeError(f"results header mismatch: {results_path}")


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    ensure_results_header(results_path)
    needs_newline = results_path.read_bytes()[-1:] not in {b"\n", b"\r"}
    with results_path.open("a", encoding="utf-8", newline="") as handle:
        if needs_newline:
            handle.write("\n")
        writer = csv.DictWriter(handle, fieldnames=RESULT_HEADER, delimiter="\t", lineterminator="\n")
        writer.writerow({key: row.get(key, 0) for key in RESULT_HEADER})


def load_best(best_path: Path) -> dict[str, Any]:
    if not best_path.exists():
        return {
            "schema_version": 1,
            "best_run_id": None,
            "best_score": None,
            "best_quality_score": None,
            "best_commit": None,
            "benchmark_fingerprint": None,
        }
    return json.loads(best_path.read_text(encoding="utf-8-sig"))


def maybe_update_best(
    best_path: Path,
    run_id: str,
    commit: str,
    metrics: dict[str, Any],
    *,
    benchmark_hash: str,
    disabled: bool,
    min_improvement: float,
) -> bool:
    if disabled or metrics.get("hard_failure"):
        return False
    best = load_best(best_path)
    current = best.get("best_quality_score", best.get("best_score"))
    current_approval = best.get("cat_approval_rate")
    current_hash = best.get("benchmark_fingerprint")
    run_quality = float(metrics.get("cat_quality_score", metrics["cat_response_score"]))
    run_response = float(metrics.get("cat_response_score", run_quality) or run_quality)
    run_approval = float(metrics.get("cat_approval_rate", 0.0) or 0.0)
    run_retry_rate = float(metrics.get("retry_rate", 0.0) or 0.0)
    keep_reason = "first_valid_run"
    if current_hash == benchmark_hash and current is not None:
        current_quality = float(current)
        best_has_retry_rate = best.get("retry_rate") is not None
        run_has_retry_rate = metrics.get("retry_rate") is not None
        current_retry_rate = float(best.get("retry_rate", 1.0) if best_has_retry_rate else 1.0)
        current_response = best.get("best_response_score")
        approval_improved = current_approval is not None and run_approval > float(current_approval)
        if current_quality <= 0:
            quality_improved = run_quality < current_quality
        else:
            quality_improved = run_quality <= current_quality * (1.0 - min_improvement)
        quality_tied = abs(run_quality - current_quality) <= 1e-9
        approval_tied = current_approval is not None and abs(run_approval - float(current_approval)) <= 1e-9
        retry_tied = best_has_retry_rate and run_has_retry_rate and abs(run_retry_rate - current_retry_rate) <= 1e-9
        retry_improved = best_has_retry_rate and run_has_retry_rate and quality_tied and approval_tied and run_retry_rate < current_retry_rate
        response_improved = (
            quality_tied
            and approval_tied
            and retry_tied
            and current_response is not None
            and run_response <= float(current_response) * (1.0 - min_improvement)
        )
        if not approval_improved and not quality_improved and not retry_improved and not response_improved:
            return False
        if approval_improved:
            keep_reason = "approval_improved"
        elif quality_improved:
            keep_reason = "quality_margin_improved"
        elif retry_improved:
            keep_reason = "retry_dependence_improved"
        else:
            keep_reason = "response_score_tiebreak_improved"
    best.update(
        {
            "schema_version": 1,
            "best_run_id": run_id,
            "best_score": run_quality,
            "best_quality_score": run_quality,
            "best_response_score": metrics.get("cat_response_score"),
            "best_latency_score": metrics.get("cat_latency_score"),
            "best_commit": commit,
            "benchmark_fingerprint": benchmark_hash,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cat_approval_rate": metrics.get("cat_approval_rate"),
            "retry_rate": metrics.get("retry_rate"),
            "retried_count": metrics.get("retried_count"),
            "primary_rejected_rate": metrics.get("primary_rejected_rate"),
            "keep_reason": keep_reason,
            "min_improvement": min_improvement,
        }
    )
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def write_human_review(path: Path, per_case: list[dict[str, Any]], raw_outputs: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    raw_by_case = {raw_key(row): row for row in raw_outputs}
    lines = [
        "# CAT Response Review",
        "",
        f"- CAT approval rate: **{metrics.get('cat_approval_rate')}**",
        f"- CAT response score: **{metrics.get('cat_response_score')}**",
        f"- Retry rate: **{metrics.get('retry_rate', 0)}** ({metrics.get('retried_count', 0)} / {metrics.get('evaluations_total', metrics.get('cases_total', 0))})",
        f"- Retry rescued accepts: **{metrics.get('retry_rescued_accept_count', 0)}**",
        f"- Retry wasted safe rejects: **{metrics.get('retry_wasted_safe_reject_count', 0)}**",
        f"- Hard failure: **{metrics.get('hard_failure')}**",
        f"- Low categories: **{metrics.get('low_categories', '') or 'none'}**",
        "",
        "## Category Metrics",
        "",
        "| Source type | Evaluations | Approval rate | Retry rate | False rejects | Unsafe accepts | Weak accepts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    category_metrics = metrics.get("category_metrics") if isinstance(metrics.get("category_metrics"), dict) else {}
    if not category_metrics:
        lines.append("| _none_ | 0 |  |  |  |  |  |")
    for source_type, data in sorted(category_metrics.items()):
        lines.append(
            "| {source_type} | {evaluations} | {approval_rate} | {retry_rate} | {false_rejects} | {unsafe_accepts} | {weak_accepts} |".format(
                source_type=escape_md(str(source_type)),
                evaluations=data.get("evaluations", 0),
                approval_rate=data.get("approval_rate", ""),
                retry_rate=data.get("retry_rate", ""),
                false_rejects=data.get("false_reject_count", 0),
                unsafe_accepts=data.get("unsafe_accept_count", 0),
                weak_accepts=data.get("weak_accept_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
            "| Case | Repeat | Outcome | Source type | Violations | Source | Raw | Final | Reject reason |",
            "|---|---:|---|---|---|---|---|---|---|",
        ]
    )
    failures = [row for row in per_case if row.get("violations")]
    if not failures:
        lines.append("| _none_ |  |  |  |  |  |  |  |  |")
    for row in failures:
        raw = raw_by_case.get(raw_key(row), {})
        lines.append(
            "| {case} | {repeat} | {outcome} | {source_type} | {violations} | {source} | {raw_output} | {final} | {reason} |".format(
                case=escape_md(str(row.get("case_id", ""))),
                repeat=escape_md(str(row.get("repeat_index", ""))),
                outcome=escape_md(str(row.get("outcome_class", ""))),
                source_type=escape_md(str(row.get("source_type", ""))),
                violations=escape_md(", ".join(str(v) for v in row.get("violations", []))),
                source=escape_md(str(raw.get("source_text", ""))),
                raw_output=escape_md(str(raw.get("raw_output", ""))[:160]),
                final=escape_md(str(row.get("final_output", ""))[:120]),
                reason=escape_md(str(row.get("reject_reason", ""))),
            )
        )
    lines.extend(
        [
            "",
            "## All Cases",
            "",
            "| Case | Repeat | Outcome | Source type | Source | Primary reject | Primary final | Final | Retried | Raw summary |",
            "|---|---:|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in per_case:
        raw = raw_by_case.get(raw_key(row), {})
        lines.append(
            "| {case} | {repeat} | {outcome} | {source_type} | {source} | {primary_reject} | {primary_final} | {final} | {retried} | {raw_output} |".format(
                case=escape_md(str(row.get("case_id", ""))),
                repeat=escape_md(str(row.get("repeat_index", ""))),
                outcome=escape_md(str(row.get("outcome_class", ""))),
                source_type=escape_md(str(row.get("source_type", ""))),
                source=escape_md(str(raw.get("source_text", ""))),
                primary_reject=escape_md(str(raw.get("primary_reject_reason", ""))),
                primary_final=escape_md(str(raw.get("primary_final_output", ""))[:80]),
                final=escape_md(str(row.get("final_output", ""))[:80]),
                retried=escape_md(str(raw.get("retried", ""))),
                raw_output=escape_md(str(raw.get("raw_output", ""))[:120]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def raw_key(row: dict[str, Any]) -> str:
    return f"{int(row.get('repeat_index', 1) or 1)}:{row.get('case_id')}"


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def average_repeat_metrics(repeat_metrics: list[dict[str, Any]], *, cases_total: int) -> dict[str, Any]:
    if not repeat_metrics:
        return score.summarize_metrics([])
    averaged: dict[str, Any] = {"cases_total": cases_total, "repeat_count": len(repeat_metrics)}
    averaged["evaluations_total"] = cases_total * len(repeat_metrics)
    keys = sorted({key for metrics in repeat_metrics for key in metrics.keys()})
    for key in keys:
        if key in {"cases_total", "hard_failure"}:
            continue
        values = [metrics.get(key) for metrics in repeat_metrics]
        if all(isinstance(value, (int, float, bool)) for value in values):
            averaged[key] = round(sum(float(value) for value in values) / len(values), 6)
    averaged["hard_failure"] = any(bool(metrics.get("hard_failure")) for metrics in repeat_metrics)
    return averaged


def write_comparison_report(path: Path, metrics: dict[str, Any], best: dict[str, Any], *, benchmark_hash: str) -> None:
    best_score = best.get("best_quality_score", best.get("best_score"))
    best_rate = best.get("cat_approval_rate")
    lines = [
        "# CAT Response Comparison",
        "",
        f"- Current score: **{metrics.get('cat_response_score')}**",
        f"- Current quality score: **{metrics.get('cat_quality_score')}**",
        f"- Current latency score: **{metrics.get('cat_latency_score')}**",
        f"- Current approval rate: **{metrics.get('cat_approval_rate')}**",
        f"- Current retry rate: **{metrics.get('retry_rate', 0)}**",
        f"- Current low categories: **{metrics.get('low_categories', '') or 'none'}**",
        f"- Previous best run: **{best.get('best_run_id') or 'none'}**",
        f"- Previous best quality score: **{best_score if best_score is not None else 'none'}**",
        f"- Previous best approval rate: **{best_rate if best_rate is not None else 'none'}**",
        f"- Previous best retry rate: **{best.get('retry_rate') if best.get('retry_rate') is not None else 'none'}**",
        f"- Same benchmark fingerprint: **{best.get('benchmark_fingerprint') == benchmark_hash}**",
        "",
    ]
    if best_score is not None:
        lines.append(f"- Quality score delta vs best: **{round(float(metrics.get('cat_quality_score', 0.0)) - float(best_score), 6)}**")
    if best_rate is not None:
        lines.append(f"- Approval delta vs best: **{round(float(metrics.get('cat_approval_rate', 0.0)) - float(best_rate), 6)}**")
    lines.extend(
        [
            "",
            "This report compares run-level metrics only. Inspect `human_review.md`, `failures.jsonl`, and per-category rows before keeping any profile.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    for attempt in range(8):
        temp_path = path.with_name(f"{path.name}.{time.time_ns()}.{attempt}.tmp")
        try:
            temp_path.write_text(payload_text, encoding="utf-8")
            temp_path.replace(path)
            return
        except OSError:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
    # Progress is diagnostic only. A dashboard read lock on Windows should not fail a benchmark run.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CAT response reliability over frozen source-text cases.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--profile", default="official")
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--fake-outputs", type=Path)
    parser.add_argument("--baseline-score", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=4, help="Run the full benchmark this many times and score the average.")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.002,
        help="Minimum quality-score improvement ratio required to replace best when approval does not improve.",
    )
    parser.add_argument("--no-update-best", action="store_true")
    parser.add_argument("--allow-hard-failure", action="store_true")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    validation_errors = validate_benchmark(args.benchmark)
    if validation_errors:
        for error in validation_errors:
            print(error, file=sys.stderr)
        return 1

    cases = read_jsonl(args.benchmark / "cases.jsonl")
    references = {str(row["case_id"]): row for row in read_jsonl(args.benchmark / "references.jsonl")}
    fake_outputs = io_adapters.load_fake_outputs(args.fake_outputs)
    profile = io_adapters.load_profile(args.profile, args.profile_json)

    raw_outputs: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    repeat_metrics: list[dict[str, Any]] = []
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    total_evaluations = len(cases) * args.repeats
    completed_evaluations = 0
    started_at = time.time()
    progress_path = output / "progress.json"
    write_progress(
        progress_path,
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "running",
            "benchmark_set": args.benchmark.name,
            "profile_name": profile.get("name", "custom"),
            "profile_hash": io_adapters.profile_hash(profile),
            "repeat_count": args.repeats,
            "cases_total": len(cases),
            "evaluations_total": total_evaluations,
            "completed_evaluations": completed_evaluations,
            "progress_rate": 0.0,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    for repeat_index in range(1, args.repeats + 1):
        repeat_rows: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases, start=1):
            raw = io_adapters.run_cat_case(case, profile, fake_outputs=fake_outputs)
            raw["repeat_index"] = repeat_index
            raw_outputs.append(raw)
            scored = score.score_case(case, raw, references[str(case["case_id"])])
            scored["repeat_index"] = repeat_index
            per_case.append(scored)
            repeat_rows.append(scored)
            completed_evaluations += 1
            elapsed = max(0.001, time.time() - started_at)
            write_progress(
                progress_path,
                {
                    "schema_version": 1,
                    "run_id": args.run_id,
                    "status": "running",
                    "benchmark_set": args.benchmark.name,
                    "profile_name": profile.get("name", "custom"),
                    "profile_hash": io_adapters.profile_hash(profile),
                    "repeat_count": args.repeats,
                    "current_repeat": repeat_index,
                    "cases_total": len(cases),
                    "current_case_index": case_index,
                    "current_case_id": case.get("case_id"),
                    "evaluations_total": total_evaluations,
                    "completed_evaluations": completed_evaluations,
                    "progress_rate": round(completed_evaluations / max(1, total_evaluations), 6),
                    "elapsed_seconds": round(elapsed, 3),
                    "evaluations_per_second": round(completed_evaluations / elapsed, 6),
                    "started_at": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        repeat_metrics.append(score.summarize_metrics(repeat_rows))

    metrics = average_repeat_metrics(repeat_metrics, cases_total=len(cases))
    metrics["category_metrics"] = score.summarize_category_metrics(per_case)
    low_categories = [
        name
        for name, data in sorted(metrics["category_metrics"].items())
        if int(data.get("evaluations", 0)) >= score.CATEGORY_MIN_EVALUATIONS
        and float(data.get("approval_rate", 0.0)) < score.CATEGORY_MIN_APPROVAL_RATE
    ]
    metrics["category_min_approval_rate"] = round(
        min((float(data.get("approval_rate", 0.0)) for data in metrics["category_metrics"].values()), default=1.0),
        6,
    )
    metrics["low_category_count"] = len(low_categories)
    metrics["low_categories"] = ",".join(low_categories)
    metrics["hard_failure"] = bool(metrics.get("hard_failure")) or bool(low_categories)
    write_jsonl(output / "raw_outputs.jsonl", raw_outputs)
    write_jsonl(output / "per_case_metrics.jsonl", per_case)
    write_jsonl(output / "failures.jsonl", [row for row in per_case if row.get("violations")])
    write_human_review(output / "human_review.md", per_case, raw_outputs, metrics)

    commit = git_value(["rev-parse", "HEAD"])
    parent_commit = git_value(["rev-parse", "HEAD^"])
    benchmark_hash = benchmark_fingerprint(args.benchmark)
    previous_best = load_best(args.results.parent / "best.json")
    write_comparison_report(output / "comparison.md", metrics, previous_best, benchmark_hash=benchmark_hash)
    kept = maybe_update_best(
        args.results.parent / "best.json",
        args.run_id,
        commit,
        metrics,
        benchmark_hash=benchmark_hash,
        disabled=args.no_update_best,
        min_improvement=args.min_improvement,
    )
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "benchmark_set": args.benchmark.name,
        "profile": profile,
        "profile_hash": io_adapters.profile_hash(profile),
        "benchmark_fingerprint": benchmark_hash,
        "cat_response_score": metrics["cat_response_score"],
        "cat_quality_score": metrics["cat_quality_score"],
        "cat_latency_score": metrics["cat_latency_score"],
        "cat_approval_rate": metrics["cat_approval_rate"],
        "hard_failure": metrics["hard_failure"],
        "metrics": metrics,
        "repeat_metrics": repeat_metrics,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(
        progress_path,
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "complete",
            "benchmark_set": args.benchmark.name,
            "profile_name": profile.get("name", "custom"),
            "profile_hash": io_adapters.profile_hash(profile),
            "repeat_count": args.repeats,
            "cases_total": len(cases),
            "evaluations_total": total_evaluations,
            "completed_evaluations": total_evaluations,
            "progress_rate": 1.0,
            "started_at": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "summary_path": str(output / "summary.json"),
            "cat_response_score": metrics["cat_response_score"],
            "cat_quality_score": metrics["cat_quality_score"],
            "cat_latency_score": metrics["cat_latency_score"],
            "cat_approval_rate": metrics["cat_approval_rate"],
            "hard_failure": metrics["hard_failure"],
            "low_categories": metrics.get("low_categories", ""),
        },
    )
    row = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "parent_commit": parent_commit,
        "experiment_name": args.experiment_name or args.run_id,
        "changed_files": changed_files(),
        "fixture_validation_ok": True,
        "benchmark_set": args.benchmark.name,
        "baseline_score": args.baseline_score,
        "delta_score": round(float(metrics["cat_quality_score"]) - float(args.baseline_score), 6),
        "kept": kept,
        "notes": f"profile={profile.get('name', 'custom')} hash={io_adapters.profile_hash(profile)}",
        **metrics,
    }
    append_result(args.results, row)
    if metrics["hard_failure"] and not args.allow_hard_failure:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
