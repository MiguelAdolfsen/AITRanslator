from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_quality_autoresearch.common.io_utils import load_jsonl, write_jsonl
from translation_quality_autoresearch.common.metrics import rank_candidates, rejected_candidate_reason


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rerank candidate scores per case.")
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    grouped = defaultdict(list)
    for row in load_jsonl(args.candidate_scores):
        grouped[str(row.get("case_id"))].append(row)
    rows = []
    for case_id, scores in sorted(grouped.items()):
        ranked = rank_candidates(scores)
        if not ranked:
            continue
        rows.append(
            {
                "case_id": case_id,
                "selected_candidate_id": ranked[0]["candidate_id"],
                "selected_agent": ranked[0]["agent"],
                "selected_text": ranked[0]["text"],
                "case_score": ranked[0]["candidate_quality_score"],
                "rejected_candidates": [
                    {"candidate_id": row["candidate_id"], "reason": rejected_candidate_reason(row)}
                    for row in ranked[1:]
                ],
            }
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} reranked decisions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
