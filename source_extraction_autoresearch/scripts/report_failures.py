from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PAGE_FAILURE_KEYS = [
    "missed_dialogue_region_count",
    "destructive_false_positive_count",
    "harmless_false_positive_count",
    "wrong_text_region_match_count",
    "severe_ocr_error_count",
    "empty_ocr_count",
    "overmerge_count",
    "undermerge_count",
    "reading_order_error_count",
    "orientation_error_count",
    "bad_crop_count",
    "duplicate_region_count",
    "non_japanese_noise_count",
]

BREAKDOWN_KEYS = [
    "missed",
    "destructive_false_positive",
    "ocr",
    "grouping",
    "order",
    "crop_duplicate_noise",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nonzero_counts(row: dict[str, Any], keys: list[str]) -> str:
    parts = []
    for key in keys:
        value = row.get(key, 0)
        if isinstance(value, (int, float)) and value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "no counted failures"


def short_text(value: Any, max_len: int = 80) -> str:
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def interesting_match(row: dict[str, Any], cer_threshold: float) -> bool:
    if not row.get("matched", False):
        return True
    if not row.get("orientation_ok", True):
        return True
    if float(row.get("cer", 0.0) or 0.0) >= cer_threshold:
        return True
    if float(row.get("box_iou", 1.0) or 0.0) < 0.45:
        return True
    if row.get("violations"):
        return True
    return False


def match_reason(row: dict[str, Any], cer_threshold: float) -> str:
    reasons: list[str] = []
    if not row.get("matched", False):
        reasons.append("missed")
    if not row.get("orientation_ok", True):
        reasons.append("orientation")
    if float(row.get("cer", 0.0) or 0.0) >= cer_threshold:
        reasons.append(f"cer={float(row.get('cer', 0.0) or 0.0):.3f}")
    if float(row.get("box_iou", 1.0) or 0.0) < 0.45:
        reasons.append(f"iou={float(row.get('box_iou', 0.0) or 0.0):.3f}")
    violations = row.get("violations") or []
    if violations:
        reasons.append("violations=" + ",".join(str(item) for item in violations))
    return ", ".join(reasons) if reasons else "ok"


def false_positive_reason(row: dict[str, Any]) -> str:
    reasons = [str(row.get("type", "false_positive"))]
    if row.get("in_ignore_region"):
        reasons.append("in_ignore")
    if row.get("overlaps_non_extractable"):
        reasons.append("overlaps_non_extractable")
    if row.get("plausible_text"):
        reasons.append("plausible_text")
    nearest = row.get("nearest_region_id")
    if nearest:
        reasons.append(
            f"nearest={nearest}/{row.get('nearest_kind')} iou={float(row.get('nearest_iou', 0.0) or 0.0):.3f}"
        )
    return ", ".join(reasons)


def score_breakdown_text(row: dict[str, Any]) -> str:
    breakdown = row.get("page_score_breakdown") or {}
    parts = []
    for key in BREAKDOWN_KEYS:
        value = breakdown.get(key, 0.0)
        if isinstance(value, (int, float)) and value:
            parts.append(f"{key}={value:.3f}")
    return ", ".join(parts) if parts else "no quality contribution"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Report the main failures from a source-extraction run folder.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--top-pages", type=int, default=10)
    parser.add_argument("--top-regions", type=int, default=25)
    parser.add_argument("--cer-threshold", type=float, default=0.50)
    args = parser.parse_args(argv)

    summary_path = args.run / "summary.json"
    if not summary_path.exists():
        print(f"missing run summary: {summary_path}")
        return 1

    summary = read_json(summary_path)
    pages = read_jsonl(args.run / "per_page_metrics.jsonl")
    matches = read_jsonl(args.run / "region_matches.jsonl")
    failures = read_jsonl(args.run / "failures.jsonl")

    print(f"# Failure Report: {summary.get('run_id', args.run.name)}")
    print()
    print(f"benchmark: {summary.get('benchmark_set', 'UNKNOWN')}")
    print(f"score: {summary.get('source_extraction_score')}")
    metrics = summary.get("metrics", {}) or {}
    if "source_extraction_quality_score" in metrics or "timing_score_component" in metrics:
        print(f"quality_score: {metrics.get('source_extraction_quality_score', 'n/a')}")
        print(f"timing_score_component: {metrics.get('timing_score_component', 'n/a')}")
    print(f"hard_failure: {summary.get('hard_failure')}")
    print()

    print("## Top Pages By Quality")
    ranked_pages = sorted(pages, key=lambda row: float(row.get("source_extraction_quality_score", row.get("source_extraction_score", 0.0)) or 0.0), reverse=True)
    for page in ranked_pages[: max(0, args.top_pages)]:
        page_id = str(page.get("page_id", "UNKNOWN"))
        overlay = args.run / "overlays" / f"{page_id}.overlay.png"
        quality = page.get("source_extraction_quality_score")
        timing = page.get("timing_score_component")
        score_parts = f"score={float(page.get('source_extraction_score', 0.0) or 0.0):.6f}"
        if quality is not None or timing is not None:
            score_parts += f" quality={float(quality or 0.0):.6f} timing={float(timing or 0.0):.6f}"
        print(
            f"- {page_id}: {score_parts}; "
            f"{nonzero_counts(page, PAGE_FAILURE_KEYS)}; {score_breakdown_text(page)}; overlay={overlay}"
        )
    print()

    print("## Slowest Pages")
    slow_pages = sorted(pages, key=lambda row: float(row.get("timing_score_component", row.get("total_ms", 0.0)) or 0.0), reverse=True)
    for page in slow_pages[: max(0, min(args.top_pages, 8))]:
        print(
            f"- {page.get('page_id', 'UNKNOWN')}: timing={float(page.get('timing_score_component', page.get('total_ms', 0.0)) or 0.0):.6f} "
            f"quality={float(page.get('source_extraction_quality_score', page.get('source_extraction_score', 0.0)) or 0.0):.6f}"
        )
    print()

    interesting = [row for row in matches if interesting_match(row, args.cer_threshold)]
    interesting.sort(
        key=lambda row: (
            not row.get("matched", False),
            float(row.get("cer", 0.0) or 0.0),
            1.0 - float(row.get("box_iou", 1.0) or 0.0),
        ),
        reverse=True,
    )
    print("## Region Issues")
    if not interesting:
        print("- no missed, severe OCR, orientation, bad-crop, or violation rows found")
    for row in interesting[: max(0, args.top_regions)]:
        print(
            f"- {row.get('page_id')} {row.get('region_id', '')}: {match_reason(row, args.cer_threshold)}; "
            f"pred={row.get('pred_region_id', '')}; "
            f"gt={short_text(row.get('gt_text'))}; ocr={short_text(row.get('pred_text'))}"
        )
    print()

    false_positives = [row for row in failures if str(row.get("type", "")).endswith("false_positive")]
    print("## False Positives")
    if not false_positives:
        print("- no structured false-positive rows found")
    for row in false_positives[: max(0, args.top_regions)]:
        print(
            f"- {row.get('page_id')} {row.get('pred_region_id')}: {false_positive_reason(row)}; "
            f"box={row.get('box')}; text={short_text(row.get('pred_text'))}"
        )
    print()

    group_failures = [row for row in failures if row.get("type") in {"overmerge", "undermerge"}]
    print("## Group Failures")
    if not group_failures:
        print("- no structured group failure rows found")
    for row in group_failures[: max(0, args.top_regions)]:
        if row.get("type") == "overmerge":
            print(
                f"- {row.get('page_id')} overmerge pred_group={row.get('pred_group_id')} "
                f"gt_groups={row.get('gt_group_ids')} members={row.get('member_pred_region_ids')}"
            )
        else:
            print(
                f"- {row.get('page_id')} undermerge gt_group={row.get('gt_group_id')} "
                f"pred_groups={row.get('pred_group_ids')} regions={row.get('region_ids')}"
            )
    print()

    order_failures = [row for row in failures if row.get("type") == "reading_order_inversion"]
    print("## Reading Order Inversions")
    if not order_failures:
        print("- no structured reading-order rows found")
    for row in order_failures[: max(0, args.top_regions)]:
        print(
            f"- {row.get('page_id')}: {row.get('left_region_id')} gt={row.get('left_gt_order')} pred={row.get('left_pred_order')} "
            f"vs {row.get('right_region_id')} gt={row.get('right_gt_order')} pred={row.get('right_pred_order')}"
        )
    print()

    if failures:
        print("## Raw Failures")
        for row in failures[: max(0, args.top_regions)]:
            print(f"- {json.dumps(row, ensure_ascii=False, sort_keys=True)}")
    else:
        print("## Raw Failures")
        print("- none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
