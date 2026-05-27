from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_quality_autoresearch.common.io_utils import load_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print and rewrite a concise run summary.")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = load_json(args.run / "summary.json", default=None)
    if not summary:
        raise SystemExit(f"missing summary.json under {args.run}")
    lines = [
        f"run_id: {summary.get('run_id')}",
        f"translation_quality_score: {summary.get('translation_quality_score')}",
        f"cases: {summary.get('case_count')}",
        f"candidates: {summary.get('candidate_count')}",
        f"hard_reject: {summary.get('hard_reject')}",
        f"hard_failure_rate: {summary.get('hard_failure_rate')}",
    ]
    text = "\n".join(lines)
    print(text)
    (args.run / "summary.md").write_text("# Translation Quality Summary\n\n" + text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
