from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_quality_autoresearch.common.io_utils import load_jsonl, split_path_list


DEFAULT_TARGETS = {
    "total_cases": 60,
    "dialogue": 20,
    "sfx": 8,
    "short_fragment": 10,
    "reaction": 10,
    "ambiguous_subject": 5,
    "ambiguous_speaker": 5,
    "names_and_honorifics": 8,
    "glossary_sensitive": 8,
    "ocr_noisy_but_salvageable": 10,
    "hallucination_trap": 8,
    "page_context_required": 8,
    "line_mapping": 5,
    "oververbose_guard": 5,
    "human_gold": 10,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report benchmark coverage and readiness gaps.")
    parser.add_argument(
        "--cases",
        default=(
            "translation_quality_autoresearch/benchmark/cases.jsonl,"
            "translation_quality_autoresearch/benchmark/adversarial_cases.jsonl,"
            "translation_quality_autoresearch/benchmark/page_context_cases.jsonl"
        ),
        help="Case JSONL file or comma-separated list.",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("translation_quality_autoresearch/benchmark/references.jsonl"),
    )
    parser.add_argument(
        "--human-gold",
        type=Path,
        default=Path("translation_quality_autoresearch/benchmark/human_gold.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-total-cases", type=int, default=DEFAULT_TARGETS["total_cases"])
    parser.add_argument(
        "--readiness-profile",
        choices=["reviewed", "machine_assisted"],
        default="reviewed",
        help="Use reviewed production targets or a machine-assisted handoff gate.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when minimum readiness targets are not met.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_cases(args.cases)
    references = load_jsonl(args.references)
    human_gold = load_jsonl(args.human_gold)
    report = coverage_report(
        cases,
        references=references,
        human_gold=human_gold,
        min_total_cases=args.min_total_cases,
        readiness_profile=args.readiness_profile,
    )
    text = render_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    ready_status = "READY_FOR_MACHINE_ASSISTED_AGENT_HANDOFF" if args.readiness_profile == "machine_assisted" else "READY_FOR_AGENT_OPTIMIZATION"
    if args.strict and report["readiness"] != ready_status:
        return 1
    return 0


def load_cases(value: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in split_path_list(value):
        rows.extend(load_jsonl(path))
    return rows


def coverage_report(
    cases: list[dict[str, Any]],
    *,
    references: list[dict[str, Any]],
    human_gold: list[dict[str, Any]],
    min_total_cases: int,
    readiness_profile: str = "reviewed",
) -> dict[str, Any]:
    tags = Counter(tag for case in cases for tag in case.get("tags", []))
    source_types = Counter(str(case.get("source_type") or "unknown") for case in cases)
    ocr_risks = Counter(str(case.get("ocr_risk") or "unknown") for case in cases)
    ref_ids = {str(row.get("case_id") or "") for row in references}
    gold_ids = {str(row.get("case_id") or "") for row in human_gold}
    case_ids = [str(case.get("case_id") or "") for case in cases]
    case_id_set = set(case_ids)

    derived = {
        "total_cases": len(cases),
        "with_references": sum(1 for case_id in case_ids if case_id in ref_ids),
        "with_human_gold": sum(1 for case_id in case_ids if case_id in gold_ids),
        "with_context": sum(1 for case in cases if case.get("context_before") or case.get("context_after")),
        "with_glossary_terms": sum(1 for case in cases if case.get("glossary_terms")),
        "with_must_preserve": sum(1 for case in cases if case.get("must_preserve")),
        "multi_line_source": sum(1 for case in cases if "\n" in str(case.get("source_text") or "")),
        "missing_references": sorted(case_id for case_id in case_id_set if case_id not in ref_ids),
        "unknown_references": sorted(case_id for case_id in ref_ids if case_id and case_id not in case_id_set),
    }

    if readiness_profile == "machine_assisted":
        target_values = {"total_cases": min_total_cases}
    else:
        target_values = dict(DEFAULT_TARGETS)
        target_values["total_cases"] = min_total_cases
    actuals = {
        "total_cases": derived["total_cases"],
        "dialogue": source_types.get("dialogue", 0),
        "sfx": source_types.get("sfx", 0),
        "short_fragment": tags.get("short_fragment", 0) + tags.get("short", 0),
        "reaction": tags.get("reaction", 0),
        "ambiguous_subject": tags.get("ambiguous_subject", 0),
        "ambiguous_speaker": tags.get("ambiguous_speaker", 0),
        "names_and_honorifics": tags.get("names_and_honorifics", 0),
        "glossary_sensitive": tags.get("glossary_sensitive", 0),
        "ocr_noisy_but_salvageable": tags.get("ocr_noisy_but_salvageable", 0),
        "hallucination_trap": tags.get("hallucination_trap", 0),
        "page_context_required": tags.get("page_context_required", 0),
        "line_mapping": tags.get("line_mapping", 0) + derived["multi_line_source"],
        "oververbose_guard": tags.get("oververbose_guard", 0),
        "human_gold": derived["with_human_gold"],
    }
    gaps = {
        name: {"actual": actual, "target": target_values[name], "missing": max(0, target_values[name] - actual)}
        for name, actual in actuals.items()
        if name in target_values
        if actual < target_values[name]
    }
    blocking_gaps = bool(gaps) or bool(derived["missing_references"])
    if blocking_gaps:
        readiness = "NEEDS_REVIEWED_BENCHMARK_EXPANSION"
    elif readiness_profile == "machine_assisted":
        readiness = "READY_FOR_MACHINE_ASSISTED_AGENT_HANDOFF"
    else:
        readiness = "READY_FOR_AGENT_OPTIMIZATION"
    return {
        "readiness": readiness,
        "readiness_profile": readiness_profile,
        "derived": derived,
        "source_types": dict(sorted(source_types.items())),
        "ocr_risks": dict(sorted(ocr_risks.items())),
        "tags": dict(sorted(tags.items())),
        "targets": target_values,
        "actuals": actuals,
        "gaps": gaps,
    }


def render_report(report: dict[str, Any]) -> str:
    derived = report["derived"]
    lines = [
        "# Benchmark Coverage Report",
        "",
        f"- readiness: {report['readiness']}",
        f"- readiness_profile: {report.get('readiness_profile', 'reviewed')}",
        f"- total_cases: {derived['total_cases']}",
        f"- with_references: {derived['with_references']}",
        f"- with_human_gold: {derived['with_human_gold']}",
        f"- with_context: {derived['with_context']}",
        f"- with_glossary_terms: {derived['with_glossary_terms']}",
        f"- with_must_preserve: {derived['with_must_preserve']}",
        f"- multi_line_source: {derived['multi_line_source']}",
        "",
        "## Source Types",
    ]
    lines.extend(f"- {name}: {count}" for name, count in report["source_types"].items())
    lines.extend(["", "## OCR Risks"])
    lines.extend(f"- {name}: {count}" for name, count in report["ocr_risks"].items())
    lines.extend(["", "## Tags"])
    lines.extend(f"- {name}: {count}" for name, count in report["tags"].items())
    lines.extend(["", "## Readiness Gaps"])
    if report["gaps"]:
        for name, row in report["gaps"].items():
            lines.append(f"- {name}: actual={row['actual']} target={row['target']} missing={row['missing']}")
    else:
        lines.append("- none")
    if derived["missing_references"]:
        lines.extend(["", "## Missing References"])
        lines.extend(f"- {case_id}" for case_id in derived["missing_references"])
    if derived["unknown_references"]:
        lines.extend(["", "## References Outside Selected Cases"])
        lines.extend(f"- {case_id}" for case_id in derived["unknown_references"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
