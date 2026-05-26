from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def score_value(row: dict[str, str]) -> float | None:
    value = row.get("source_extraction_score", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_smoke_row(row: dict[str, str]) -> bool:
    return row.get("notes") == "adapter_smoke"


def is_draft_row(row: dict[str, str]) -> bool:
    return "_draft" in row.get("benchmark_set", "").lower() or row.get("notes") == "draft_benchmark"


def is_decision_row(row: dict[str, str]) -> bool:
    return score_value(row) is not None and not is_smoke_row(row) and not is_draft_row(row)


def describe_row(prefix: str, row: dict[str, str]) -> str:
    score = score_value(row)
    score_text = f"{score:.6f}" if score is not None else "n/a"
    return (
        f"{prefix}: {row.get('run_id', '')} score={score_text} "
        f"kept={row.get('kept', '')} notes={row.get('notes', '')}"
    )


def matrix_prefix(run_id: str) -> str | None:
    suffixes = ("_synthetic", "_v2", "_spy_v1", "_hard_v1")
    for suffix in suffixes:
        if run_id.endswith(suffix):
            return run_id[: -len(suffix)]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize source-extraction results.tsv.")
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.results.exists():
        print(f"missing results file: {args.results}")
        return 1
    with args.results.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        print("no result rows")
        return 0
    scored = [row for row in rows if score_value(row) is not None]
    decision_rows = [row for row in rows if is_decision_row(row)]
    latest = rows[-1]
    kept = [row for row in rows if row.get("kept") == "TRUE"]
    failed = [row for row in rows if "hard_failure" in str(row.get("notes", ""))]
    smoke = [row for row in rows if is_smoke_row(row)]
    draft = [row for row in rows if is_draft_row(row)]
    print(f"rows: {len(rows)} decision: {len(decision_rows)} kept: {len(kept)} smoke: {len(smoke)} draft: {len(draft)} hard-failure: {len(failed)}")
    print(describe_row("latest row", latest))
    if decision_rows:
        best_decision = min(decision_rows, key=lambda row: score_value(row) or float("inf"))
        print(describe_row("best decision row", best_decision))

    rows_by_benchmark: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_benchmark[row.get("benchmark_set") or "UNKNOWN"].append(row)
    print()
    print("by benchmark:")
    for benchmark in sorted(rows_by_benchmark):
        benchmark_rows = rows_by_benchmark[benchmark]
        benchmark_decision_rows = [row for row in benchmark_rows if is_decision_row(row)]
        benchmark_latest = benchmark_rows[-1]
        if benchmark_decision_rows:
            benchmark_best = min(benchmark_decision_rows, key=lambda row: score_value(row) or float("inf"))
            best_score = score_value(benchmark_best)
            best_text = f"{benchmark_best.get('run_id')} score={best_score:.6f}" if best_score is not None else "n/a"
        else:
            best_text = "no decision rows"
        latest_score = score_value(benchmark_latest)
        latest_text = f"{benchmark_latest.get('run_id')} score={latest_score:.6f}" if latest_score is not None else "n/a"
        print(
            f"- {benchmark}: rows={len(benchmark_rows)} "
            f"decision={len(benchmark_decision_rows)} best={best_text} latest={latest_text}"
        )

    if len(scored) >= 2:
        first_score = score_value(scored[0]) or 0.0
        last_score = score_value(scored[-1]) or 0.0
        print()
        print(f"raw score trend: first={first_score:.6f} latest={last_score:.6f} delta={last_score - first_score:.6f}")

    matrix_rows_by_prefix: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        prefix = matrix_prefix(row.get("run_id", ""))
        if prefix:
            matrix_rows_by_prefix[prefix].append(row)
    if matrix_rows_by_prefix:
        latest_prefix = sorted(matrix_rows_by_prefix, key=lambda key: max(row.get("timestamp", "") for row in matrix_rows_by_prefix[key]))[-1]
        print()
        print(f"latest matrix: {latest_prefix}")
        for row in sorted(matrix_rows_by_prefix[latest_prefix], key=lambda item: item.get("run_id", "")):
            score = score_value(row)
            score_text = f"{score:.6f}" if score is not None else "n/a"
            print(f"- {row.get('run_id')}: benchmark={row.get('benchmark_set')} score={score_text} kept={row.get('kept')} notes={row.get('notes')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
