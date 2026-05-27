from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_quality_autoresearch.common.io_utils import load_json


REGRESSION_METRICS = [
    "critical_mqm_errors",
    "japanese_leakage_count",
    "assistant_chatter_count",
    "glossary_violation_count",
    "hard_failure_count",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two translation-quality run summaries.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = load_json(args.baseline, default={}) or {}
    candidate = load_json(args.candidate, default={}) or {}
    try:
        comparison = compare_summaries(baseline, candidate)
    except ValueError as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 1
    lines = render_comparison(baseline, candidate, comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"recommendation={comparison['recommendation']} delta={comparison['score_delta']:.6f}")
    return 0


def compare_summaries(baseline: dict, candidate: dict) -> dict:
    baseline_benchmark = str(baseline.get("benchmark_id") or "")
    candidate_benchmark = str(candidate.get("benchmark_id") or "")
    if baseline_benchmark and candidate_benchmark and baseline_benchmark != candidate_benchmark:
        raise ValueError(f"benchmark_id mismatch: {baseline_benchmark} != {candidate_benchmark}")
    delta = float(candidate.get("translation_quality_score") or 0.0) - float(
        baseline.get("translation_quality_score") or 0.0
    )
    metric_deltas = {
        metric: int(candidate.get(metric) or 0) - int(baseline.get(metric) or 0)
        for metric in REGRESSION_METRICS
    }
    hard_regressed = bool(candidate.get("hard_reject")) and not bool(baseline.get("hard_reject"))
    metric_regressions = {
        metric: value
        for metric, value in metric_deltas.items()
        if value > 0
    }
    if delta < 0 and not hard_regressed and not metric_regressions:
        recommendation = "KEEP"
    elif delta > 0 or hard_regressed or metric_regressions:
        recommendation = "REVERT"
    else:
        recommendation = "MANUAL_REVIEW"
    return {
        "score_delta": delta,
        "hard_regressed": hard_regressed,
        "metric_deltas": metric_deltas,
        "metric_regressions": metric_regressions,
        "recommendation": recommendation,
    }


def render_comparison(baseline: dict, candidate: dict, comparison: dict) -> list[str]:
    delta = comparison["score_delta"]
    metric_deltas = comparison["metric_deltas"]
    lines = [
        "# Run Comparison",
        "",
        f"- baseline: {baseline.get('run_id')}",
        f"- candidate: {candidate.get('run_id')}",
        f"- score delta: {delta:.6f}",
        f"- hard reject changed: {baseline.get('hard_reject')} -> {candidate.get('hard_reject')}",
        f"- recommendation: {comparison['recommendation']}",
        "",
        "## Guardrail Deltas",
    ]
    for metric in REGRESSION_METRICS:
        lines.append(f"- {metric}: {metric_deltas[metric]}")
    if comparison["metric_regressions"]:
        lines.extend(["", "## Regressed Guardrails"])
        for metric, value in comparison["metric_regressions"].items():
            lines.append(f"- {metric}: +{value}")
    lines.extend(
        [
            "",
            "## Tag-Level Deltas",
        ]
    )
    base_tags = baseline.get("tag_slices") or {}
    cand_tags = candidate.get("tag_slices") or {}
    for tag in sorted(set(base_tags) | set(cand_tags)):
        before = float((base_tags.get(tag) or {}).get("average_case_score") or 0.0)
        after = float((cand_tags.get(tag) or {}).get("average_case_score") or 0.0)
        lines.append(f"- {tag}: {after - before:.6f}")
    lines.extend(
        [
            "",
            "## Agent Selection",
        ]
    )
    base_agents = baseline.get("selected_agents") or {}
    cand_agents = candidate.get("selected_agents") or {}
    for agent in sorted(set(base_agents) | set(cand_agents)):
        before = int(base_agents.get(agent) or 0)
        after = int(cand_agents.get(agent) or 0)
        lines.append(f"- {agent}: {before} -> {after}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
