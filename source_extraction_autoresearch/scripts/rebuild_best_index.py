from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def score_value(row: dict[str, str]) -> float | None:
    value = row.get("source_extraction_score", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def numeric(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def quality_score_value(row: dict[str, str]) -> float | None:
    value = score_value(row)
    if value is None:
        return None
    return round(value - numeric(row, "p95_extraction_ms_per_page"), 6)


def row_score(row: dict[str, str], *, use_quality_score: bool) -> float | None:
    return quality_score_value(row) if use_quality_score else score_value(row)


def sort_key(row: dict[str, str], *, use_quality_score: bool) -> tuple[float, int, int, float, int, int, float, int]:
    merge_order_orientation = (
        int(numeric(row, "overmerge_count"))
        + int(numeric(row, "undermerge_count"))
        + int(numeric(row, "reading_order_error_count"))
        + int(numeric(row, "orientation_error_count"))
    )
    return (
        row_score(row, use_quality_score=use_quality_score) or float("inf"),
        int(numeric(row, "missed_dialogue_region_count")),
        int(numeric(row, "destructive_false_positive_count")),
        numeric(row, "mean_ocr_cer"),
        int(numeric(row, "severe_ocr_error_count")),
        merge_order_orientation,
        numeric(row, "p95_extraction_ms_per_page"),
        len([item for item in row.get("changed_files", "").split(";") if item]),
    )


def best_path_for_benchmark(results_dir: Path, benchmark_set: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", benchmark_set).strip("._")
    if not safe_name:
        safe_name = "unknown"
    return results_dir / f"best.{safe_name}.json"


def is_decision_row(row: dict[str, str]) -> bool:
    notes = row.get("notes", "")
    benchmark = row.get("benchmark_set", "")
    return score_value(row) is not None and notes != "adapter_smoke" and "_draft" not in benchmark.lower() and notes != "draft_benchmark"


def best_rows(rows: list[dict[str, str]], *, use_quality_score: bool) -> dict[str, dict[str, str]]:
    best_by_benchmark: dict[str, dict[str, str]] = {}
    for row in rows:
        if not is_decision_row(row):
            continue
        benchmark = row.get("benchmark_set") or "UNKNOWN"
        current = best_by_benchmark.get(benchmark)
        if current is None or sort_key(row, use_quality_score=use_quality_score) < sort_key(current, use_quality_score=use_quality_score):
            best_by_benchmark[benchmark] = row
    return best_by_benchmark


def write_best_file(results_dir: Path, benchmark: str, row: dict[str, str], *, dry_run: bool, use_quality_score: bool) -> Path:
    path = best_path_for_benchmark(results_dir, benchmark)
    score = row_score(row, use_quality_score=use_quality_score)
    payload = {
        "schema_version": 1,
        "benchmark_set": benchmark,
        "benchmark_fingerprint": None,
        "best_run_id": row.get("run_id"),
        "best_score": score,
        "best_commit": row.get("commit"),
        "best_sort_key": list(sort_key(row, use_quality_score=use_quality_score)),
        "best_row_timestamp": row.get("timestamp"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "notes": "Rebuilt from results.tsv; smoke and draft rows excluded. Timing is a tie-breaker.",
    }
    if not dry_run:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild per-benchmark best JSON files from results.tsv.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-quality-score", action="store_true")
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"missing results file: {args.results}")
        return 1

    with args.results.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    best = best_rows(rows, use_quality_score=args.use_quality_score)
    if not best:
        print("no decision rows found")
        return 0

    for benchmark in sorted(best):
        row = best[benchmark]
        path = write_best_file(args.results.parent, benchmark, row, dry_run=args.dry_run, use_quality_score=args.use_quality_score)
        score = row_score(row, use_quality_score=args.use_quality_score)
        action = "would write" if args.dry_run else "wrote"
        print(f"{action} {path}: run_id={row.get('run_id')} score={score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
