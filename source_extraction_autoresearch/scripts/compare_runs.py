from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8-sig"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two source-extraction run folders.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    before = load_summary(args.before)
    after = load_summary(args.after)
    before_metrics = before.get("metrics", {})
    after_metrics = after.get("metrics", {})
    before_score = float(before_metrics.get("source_extraction_score", before.get("source_extraction_score", 0.0)))
    after_score = float(after_metrics.get("source_extraction_score", after.get("source_extraction_score", 0.0)))
    print(f"score: {before_score:.6f} -> {after_score:.6f} delta={after_score - before_score:.6f}")
    for key in sorted(set(before_metrics) | set(after_metrics)):
        if not isinstance(before_metrics.get(key, 0), (int, float)) and not isinstance(after_metrics.get(key, 0), (int, float)):
            continue
        delta = float(after_metrics.get(key, 0) or 0) - float(before_metrics.get(key, 0) or 0)
        if delta:
            print(f"{key}: {before_metrics.get(key, 0)} -> {after_metrics.get(key, 0)} delta={delta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

