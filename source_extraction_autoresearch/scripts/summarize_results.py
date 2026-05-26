from __future__ import annotations

import argparse
import csv
from pathlib import Path


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
    scored = [row for row in rows if row.get("source_extraction_score")]
    best = min(scored, key=lambda row: float(row["source_extraction_score"])) if scored else None
    latest = rows[-1]
    if best:
        print(f"best run: {best.get('run_id')} score={best.get('source_extraction_score')} kept={best.get('kept')}")
    print(f"latest run: {latest.get('run_id')} score={latest.get('source_extraction_score')} notes={latest.get('notes')}")
    kept = [row for row in rows if row.get("kept") == "TRUE"]
    failed = [row for row in rows if "hard_failure" in str(row.get("notes", ""))]
    print(f"rows: {len(rows)} kept: {len(kept)} hard-failure: {len(failed)}")
    if len(scored) >= 2:
        first = float(scored[0]["source_extraction_score"])
        last = float(scored[-1]["source_extraction_score"])
        print(f"score trend: first={first:.6f} latest={last:.6f} delta={last - first:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

