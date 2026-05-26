from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    if not args.results.exists():
        print(f"No results file: {args.results}")
        return 1
    rows = list(csv.DictReader(args.results.open("r", encoding="utf-8"), delimiter="\t"))
    rows = [row for row in rows if row.get("run_id")]
    rows.sort(key=lambda row: (float(row.get("cat_quality_score") or row.get("cat_response_score") or 999999), -float(row.get("cat_approval_rate") or 0)))
    print("run_id\tcat_quality_score\tcat_response_score\tcat_approval_rate\tretry_rate\tlow_categories\thard_failure\tkept\tnotes")
    for row in rows[: args.limit]:
        print(
            "\t".join(
                [
                    row.get("run_id", ""),
                    row.get("cat_quality_score", ""),
                    row.get("cat_response_score", ""),
                    row.get("cat_approval_rate", ""),
                    row.get("retry_rate", ""),
                    row.get("low_categories", ""),
                    row.get("hard_failure", ""),
                    row.get("kept", ""),
                    row.get("notes", ""),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
