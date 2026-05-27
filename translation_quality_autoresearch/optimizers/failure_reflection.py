from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_quality_autoresearch.common.io_utils import load_json, load_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize worst failures and suggest prompt/rule changes.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    scores = load_jsonl(args.run / "candidate_scores.jsonl")
    summary = load_json(args.run / "summary.json", default={}) or {}
    hard = [row for row in scores if row.get("hard_reject")]
    warnings = Counter(warning for row in scores for warning in (row.get("deterministic") or {}).get("warnings", []))
    agents = Counter(row.get("agent") for row in hard)
    lines = [
        "# Failure Reflection",
        "",
        f"- run_id: {summary.get('run_id')}",
        f"- translation_quality_score: {summary.get('translation_quality_score')}",
        f"- hard rejected candidates: {len(hard)}",
        "",
        "## Common Hard-Failure Agents",
    ]
    for agent, count in agents.most_common():
        lines.append(f"- {agent}: {count}")
    lines.append("")
    lines.append("## Common Warnings")
    for warning, count in warnings.most_common():
        lines.append(f"- {warning}: {count}")
    lines.append("")
    lines.append("## Possible Changes")
    lines.extend(
        [
            "- Tighten prompts against assistant chatter if chatter appears.",
            "- Improve glossary injection when glossary violations appear.",
            "- Add repair triggers for Japanese leakage or copied source.",
            "- Inspect oververbose outputs before reducing max token limits.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
