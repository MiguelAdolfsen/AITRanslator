from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .text_filter import suspected_bad_translation


PROFILE_ENVS: dict[str, dict[str, str]] = {
    "quality": {
        "MANGA_QWEN_TRANSLATION_TEMPERATURE": "0.4",
        "MANGA_QWEN_TRANSLATION_TOP_P": "0.8",
        "MANGA_QWEN_REPAIR_TEMPERATURE": "0.6",
        "MANGA_QWEN_REPAIR_TOP_P": "0.95",
    },
    "strict": {
        "MANGA_QWEN_TRANSLATION_TEMPERATURE": "0.2",
        "MANGA_QWEN_TRANSLATION_TOP_P": "0.8",
        "MANGA_QWEN_REPAIR_TEMPERATURE": "0.4",
        "MANGA_QWEN_REPAIR_TOP_P": "0.9",
    },
    "official": {
        "MANGA_QWEN_TRANSLATION_TEMPERATURE": "0.7",
        "MANGA_QWEN_TRANSLATION_TOP_P": "0.8",
        "MANGA_QWEN_REPAIR_TEMPERATURE": "0.6",
        "MANGA_QWEN_REPAIR_TOP_P": "0.95",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m manga_local_translator.quality_eval",
        description="Run repeatable Qwen quality benchmarks and summarize debug OCR reports.",
    )
    parser.add_argument("input", type=Path, help="Input image folder or file.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("quality-runs"),
        help="Folder where per-profile translated outputs and summaries are written.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--profiles",
        default="quality",
        help=f"Comma-separated profiles to run. Available: {', '.join(PROFILE_ENVS)}",
    )
    parser.add_argument(
        "--translator",
        choices=["qwen", "cat"],
        default="qwen",
        help="Primary translator to benchmark. Qwen critic/fallback can still be used with CAT primary.",
    )
    parser.add_argument(
        "--qwen-model",
        type=Path,
        default=None,
        help="Optional Qwen text GGUF path for all profile runs.",
    )
    parser.add_argument(
        "--qwen-fallback-model",
        type=Path,
        default=None,
        help="Optional second Qwen text GGUF for rejected/repaired/suspicious blocks.",
    )
    parser.add_argument(
        "--qwen-critic-model",
        type=Path,
        default=None,
        help="Optional Qwen text GGUF used as a critic/verifier for suspicious accepted blocks.",
    )
    parser.add_argument(
        "--cat-model",
        default=None,
        help="Optional CAT GGUF path or Hugging Face model id. Defaults to local .models/CAT-Translate/*.gguf, then cyberagent/CAT-Translate-7b.",
    )
    parser.add_argument(
        "--qwen-mode",
        choices=["block", "page"],
        default="block",
        help="Qwen translation mode to benchmark.",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable vision repair in each profile run.",
    )
    parser.add_argument(
        "--vision-facts",
        action="store_true",
        help="Enable pre-translation vision facts in each profile run.",
    )
    parser.add_argument(
        "--compare-vision",
        action="store_true",
        help="Run paired text-only and vision-enabled outputs for each profile.",
    )
    parser.add_argument(
        "--vision-trigger",
        choices=["suspicious", "layout", "all"],
        default="suspicious",
        help="Vision repair trigger to pass through.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not run translation; summarize existing profile output folders.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume to the translator for batched hybrid runs.",
    )
    parser.add_argument(
        "--render-images",
        action="store_true",
        help="Render translated image files during benchmark runs. By default benchmarks write OCR JSON/review reports only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_name = args.name or datetime.now().strftime("qwen-quality-%Y%m%d-%H%M%S")
    profiles = [profile.strip() for profile in args.profiles.split(",") if profile.strip()]
    unknown = [profile for profile in profiles if profile not in PROFILE_ENVS]
    if unknown:
        raise SystemExit(f"Unknown profile(s): {', '.join(unknown)}")

    run_root = args.output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for profile, label, output_name, vision_enabled in benchmark_variants(
        profiles,
        vision=args.vision,
        compare_vision=args.compare_vision,
        vision_trigger=args.vision_trigger,
    ):
        output_dir = run_root / output_name
        if not args.summarize_only:
            run_profile(
                args.input,
                output_dir,
                profile,
                qwen_model=args.qwen_model,
                qwen_fallback_model=args.qwen_fallback_model,
                qwen_critic_model=args.qwen_critic_model,
                translator=args.translator,
                cat_model=args.cat_model,
                qwen_mode=args.qwen_mode,
                vision=vision_enabled,
                vision_facts=args.vision_facts,
                vision_trigger=args.vision_trigger,
                resume=args.resume,
                skip_render=not args.render_images,
            )
        rows = summarize_debug_reports(output_dir, profile=label)
        all_rows.extend(rows)
        write_summary_files(output_dir, rows)
        write_review_artifacts(output_dir)

    write_summary_files(run_root, all_rows)
    write_review_artifacts(run_root)
    print(f"Quality evaluation written to: {run_root}")
    return 0


def write_review_artifacts(input_dir: Path) -> None:
    from .review_report import common_root, iter_reports, rows_from_report, write_csv_report, write_html_report, write_markdown_report

    reports = sorted({path.resolve() for path in iter_reports(input_dir)}, key=lambda item: natural_report_sort_key(input_dir, item))
    if not reports:
        return
    root_hint = common_root(reports)
    review_rows = [
        row
        for report_path in reports
        for row in rows_from_report(report_path, root_hint=root_hint)
    ]
    html_path = input_dir / "translation_review.html"
    csv_path = input_dir / "translation_review.csv"
    markdown_path = input_dir / "translation_review.md"
    write_html_report(review_rows, html_path)
    write_csv_report(review_rows, csv_path)
    write_markdown_report(review_rows, markdown_path)


def benchmark_variants(
    profiles: list[str],
    *,
    vision: bool,
    compare_vision: bool,
    vision_trigger: str,
) -> list[tuple[str, str, str, bool]]:
    variants: list[tuple[str, str, str, bool]] = []
    for profile in profiles:
        if compare_vision:
            variants.append((profile, f"{profile}:text", f"{profile}-text", False))
            variants.append((profile, f"{profile}:vision-{vision_trigger}", f"{profile}-vision-{vision_trigger}", True))
        else:
            suffix = f"vision-{vision_trigger}" if vision else "text"
            variants.append((profile, f"{profile}:{suffix}", profile, vision))
    return variants


def run_profile(
    input_path: Path,
    output_dir: Path,
    profile: str,
    *,
    qwen_model: Path | None,
    qwen_fallback_model: Path | None,
    qwen_critic_model: Path | None,
    translator: str,
    cat_model: str | None,
    qwen_mode: str,
    vision: bool,
    vision_facts: bool,
    vision_trigger: str,
    resume: bool,
    skip_render: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PROFILE_ENVS[profile])
    command = [
        sys.executable,
        "-m",
        "manga_local_translator",
        str(input_path),
        str(output_dir),
        "--detector",
        "ctd",
        "--ocr-engine",
        "manga-ocr",
        "--translator",
        translator,
        "--qwen-mode",
        qwen_mode,
        "--debug",
        "--overwrite",
    ]
    if skip_render:
        command.append("--skip-render")
    if qwen_model is not None:
        command.extend(["--qwen-model", str(qwen_model)])
    if qwen_fallback_model is not None:
        command.extend(["--qwen-fallback-model", str(qwen_fallback_model)])
    if qwen_critic_model is not None:
        command.extend(["--qwen-critic-model", str(qwen_critic_model)])
    if cat_model is not None:
        command.extend(["--cat-model", str(cat_model)])
    if vision:
        command.extend(["--vision", "--vision-trigger", vision_trigger])
    if vision_facts:
        command.append("--vision-facts")
    if resume:
        command.append("--resume")
    completed = subprocess.run(command, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Quality profile failed: profile={profile} returncode={completed.returncode}")


def summarize_debug_reports(output_dir: Path, *, profile: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(output_dir.rglob("*.ocr.json"), key=lambda path: natural_report_sort_key(output_dir, path)):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        kept_blocks = report.get("kept_blocks", [])
        page_summary = report.get("page_summary", {})
        rows.append(
            {
                "profile": profile,
                "page": report_page_label(output_dir, report_path),
                "report": str(report_path),
                "kept_blocks": len(kept_blocks),
                "skipped_blocks": len(report.get("skipped_blocks", [])),
                "fallback_blocks": len(report.get("fallback_blocks", [])),
                "ellipsis_outputs": count_blocks(kept_blocks, lambda block: block.get("translated_text") == "..."),
                "cat_used": count_blocks(kept_blocks, lambda block: block.get("cat_used") is True),
                "cat_bypassed": count_blocks(kept_blocks, lambda block: block.get("cat_bypassed") is True),
                "cat_bypass_reasons": json.dumps(count_values(block.get("cat_bypass_reason") for block in kept_blocks if block.get("cat_bypassed") is True), ensure_ascii=False, sort_keys=True),
                "cat_rejected": count_blocks(kept_blocks, lambda block: block.get("cat_rejected") is True),
                "cat_reject_reasons": json.dumps(count_values(block.get("cat_reject_reason") for block in kept_blocks if block.get("cat_rejected") is True), ensure_ascii=False, sort_keys=True),
                "cat_chatter_rejected": count_blocks(kept_blocks, lambda block: block.get("cat_chatter_rejected") is True),
                "cat_retry_attempted": count_blocks(kept_blocks, lambda block: block.get("cat_retry_attempted") is True),
                "cat_retry_accepted": count_blocks(kept_blocks, lambda block: block.get("cat_retry_accepted") is True),
                "cat_retry_rejected": count_blocks(kept_blocks, lambda block: block.get("cat_retry_attempted") is True and block.get("cat_retry_accepted") is not True),
                "cat_retry_reject_reasons": json.dumps(count_values(block.get("cat_retry_reject_reason") for block in kept_blocks if block.get("cat_retry_attempted") is True and block.get("cat_retry_accepted") is not True), ensure_ascii=False, sort_keys=True),
                "cat_suspect_second_pass_attempted": count_blocks(kept_blocks, lambda block: block.get("cat_suspect_second_pass_attempted") is True),
                "cat_suspect_second_pass_accepted": count_blocks(kept_blocks, lambda block: block.get("cat_suspect_second_pass_accepted") is True),
                "cat_suspect_second_pass_rejected": count_blocks(kept_blocks, lambda block: block.get("cat_suspect_second_pass_attempted") is True and block.get("cat_suspect_second_pass_accepted") is not True),
                "cat_suspect_second_pass_reject_reasons": json.dumps(count_values(block.get("cat_suspect_second_pass_reject_reason") for block in kept_blocks if block.get("cat_suspect_second_pass_attempted") is True and block.get("cat_suspect_second_pass_accepted") is not True), ensure_ascii=False, sort_keys=True),
                "cat_q8_fallback_attempted": count_blocks(kept_blocks, lambda block: block.get("cat_q8_fallback_attempted") is True),
                "cat_q8_fallback_accepted": count_blocks(kept_blocks, lambda block: block.get("cat_q8_fallback_accepted") is True),
                "qwen_used": count_blocks(kept_blocks, lambda block: block.get("qwen_used") is True),
                "qwen_rejected": count_blocks(kept_blocks, lambda block: block.get("qwen_rejected") is True),
                "qwen_repairs_attempted": count_blocks(kept_blocks, lambda block: block.get("qwen_repair_used") is True),
                "qwen_repairs_accepted": count_blocks(kept_blocks, lambda block: block.get("qwen_repair_accepted") is True),
                "qwen_fallback_attempted": count_blocks(kept_blocks, lambda block: block.get("qwen_fallback_attempted") is True),
                "qwen_fallback_accepted": count_blocks(kept_blocks, lambda block: block.get("qwen_fallback_accepted") is True),
                "qwen_critic_attempted": count_blocks(kept_blocks, lambda block: block.get("qwen_critic_attempted") is True),
                "qwen_critic_flagged": count_blocks(kept_blocks, lambda block: block.get("qwen_critic_flagged") is True),
                "qwen_critic_issue_types": json.dumps(count_split_values(issue for block in kept_blocks for issue in block.get("qwen_critic_issues", [])), ensure_ascii=False, sort_keys=True),
                "qwen_critic_evidence_gate_reasons": json.dumps(count_values(block.get("qwen_critic_evidence_gate_reason") for block in kept_blocks if block.get("qwen_critic_attempted") is True), ensure_ascii=False, sort_keys=True),
                "evidence_risk_blocks": count_blocks(kept_blocks, lambda block: bool(block.get("evidence_risk_flags"))),
                "evidence_repair_reason_blocks": count_blocks(kept_blocks, lambda block: bool(block.get("evidence_repair_reasons"))),
                "evidence_risk_types": json.dumps(count_split_values(flag for block in kept_blocks for flag in block.get("evidence_risk_flags", [])), ensure_ascii=False, sort_keys=True),
                "evidence_repair_reason_types": json.dumps(count_split_values(reason for block in kept_blocks for reason in block.get("evidence_repair_reasons", [])), ensure_ascii=False, sort_keys=True),
                "qwen_fallback_reject_reasons": json.dumps(count_values(block.get("qwen_fallback_reject_reason") for block in kept_blocks if block.get("qwen_fallback_attempted") is True), ensure_ascii=False, sort_keys=True),
                "qwen_page_used": count_blocks(kept_blocks, lambda block: block.get("qwen_page_used") is True),
                "qwen_page_rejected": count_blocks(kept_blocks, lambda block: block.get("qwen_page_rejected") is True),
                "layout_warning_blocks": count_blocks(kept_blocks, lambda block: bool(block.get("layout_warnings"))),
                "layout_warning_types": json.dumps(count_split_values(warning for block in kept_blocks for warning in block.get("layout_warnings", [])), ensure_ascii=False, sort_keys=True),
                "vision_attempted": count_blocks(kept_blocks, lambda block: block.get("vision_attempted") is True),
                "vision_accepted": count_blocks(kept_blocks, lambda block: block.get("vision_accepted") is True),
                "vision_rejected": count_blocks(kept_blocks, lambda block: block.get("vision_attempted") is True and block.get("vision_accepted") is not True),
                "vision_json_repair_attempted": count_blocks(kept_blocks, lambda block: block.get("vision_json_repair_attempted") is True),
                "vision_json_repair_used": count_blocks(kept_blocks, lambda block: block.get("vision_json_repair_used") is True),
                "vision_reject_reasons": json.dumps(count_values(block.get("vision_reject_reason") for block in kept_blocks if block.get("vision_attempted") is True), ensure_ascii=False, sort_keys=True),
                "vision_facts_attempted": count_blocks(kept_blocks, lambda block: block.get("vision_facts_attempted") is True),
                "vision_facts_accepted": count_blocks(kept_blocks, lambda block: block.get("vision_facts_accepted") is True),
                "vision_facts_rejected": count_blocks(kept_blocks, lambda block: block.get("vision_facts_attempted") is True and block.get("vision_facts_accepted") is not True),
                "vision_facts_json_repair_attempted": count_blocks(kept_blocks, lambda block: block.get("vision_facts_json_repair_attempted") is True),
                "vision_facts_json_repair_used": count_blocks(kept_blocks, lambda block: block.get("vision_facts_json_repair_used") is True),
                "vision_facts_reject_reasons": json.dumps(count_values(block.get("vision_facts_reject_reason") for block in kept_blocks if block.get("vision_facts_attempted") is True), ensure_ascii=False, sort_keys=True),
                "suspected_bad_translations": count_blocks(kept_blocks, is_suspected_bad_block),
                "page_summary": page_summary,
            }
        )
    rows.append(build_total_row(rows, profile=profile, output_dir=output_dir))
    return rows


def natural_report_sort_key(output_dir: Path, report_path: Path) -> list[object]:
    try:
        label = report_path.relative_to(output_dir).as_posix().lower()
    except ValueError:
        label = report_path.as_posix().lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", label)]


def report_page_label(output_dir: Path, report_path: Path) -> str:
    try:
        relative = report_path.relative_to(output_dir)
    except ValueError:
        relative = report_path.name
    label = Path(relative).as_posix()
    return label.removesuffix(".ocr.json")


def count_blocks(blocks: list[dict[str, Any]], predicate) -> int:
    return sum(1 for block in blocks if predicate(block))


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_split_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if isinstance(value, str):
            parts = [part for part in value.split(",") if part]
        else:
            parts = [str(value)] if value else []
        for part in parts:
            counts[part] = counts.get(part, 0) + 1
    return counts


def is_suspected_bad_block(block: dict[str, Any]) -> bool:
    text = str(block.get("translated_text") or "").strip()
    if not text or text == "...":
        return True
    return suspected_bad_translation(text)


def build_total_row(rows: list[dict[str, Any]], *, profile: str, output_dir: Path) -> dict[str, Any]:
    numeric_keys = [
        "kept_blocks",
        "skipped_blocks",
        "fallback_blocks",
        "ellipsis_outputs",
        "cat_used",
        "cat_bypassed",
        "cat_rejected",
        "cat_chatter_rejected",
        "cat_retry_attempted",
        "cat_retry_accepted",
        "cat_retry_rejected",
        "cat_suspect_second_pass_attempted",
        "cat_suspect_second_pass_accepted",
        "cat_suspect_second_pass_rejected",
        "cat_q8_fallback_attempted",
        "cat_q8_fallback_accepted",
        "qwen_used",
        "qwen_rejected",
        "qwen_repairs_attempted",
        "qwen_repairs_accepted",
        "qwen_fallback_attempted",
        "qwen_fallback_accepted",
        "qwen_critic_attempted",
        "qwen_critic_flagged",
        "evidence_risk_blocks",
        "evidence_repair_reason_blocks",
        "qwen_page_used",
        "qwen_page_rejected",
        "layout_warning_blocks",
        "vision_attempted",
        "vision_accepted",
        "vision_rejected",
        "vision_json_repair_attempted",
        "vision_json_repair_used",
        "vision_facts_attempted",
        "vision_facts_accepted",
        "vision_facts_rejected",
        "vision_facts_json_repair_attempted",
        "vision_facts_json_repair_used",
        "suspected_bad_translations",
    ]
    total = {
        "profile": profile,
        "page": "__total__",
        "report": str(output_dir),
        "page_summary": {},
    }
    for key in numeric_keys:
        total[key] = sum(int(row.get(key, 0)) for row in rows)
    total["layout_warning_types"] = json.dumps(aggregate_json_counts(rows, "layout_warning_types"), ensure_ascii=False, sort_keys=True)
    total["cat_bypass_reasons"] = json.dumps(aggregate_json_counts(rows, "cat_bypass_reasons"), ensure_ascii=False, sort_keys=True)
    total["cat_reject_reasons"] = json.dumps(aggregate_json_counts(rows, "cat_reject_reasons"), ensure_ascii=False, sort_keys=True)
    total["cat_retry_reject_reasons"] = json.dumps(aggregate_json_counts(rows, "cat_retry_reject_reasons"), ensure_ascii=False, sort_keys=True)
    total["cat_suspect_second_pass_reject_reasons"] = json.dumps(aggregate_json_counts(rows, "cat_suspect_second_pass_reject_reasons"), ensure_ascii=False, sort_keys=True)
    total["qwen_critic_issue_types"] = json.dumps(aggregate_json_counts(rows, "qwen_critic_issue_types"), ensure_ascii=False, sort_keys=True)
    total["qwen_critic_evidence_gate_reasons"] = json.dumps(aggregate_json_counts(rows, "qwen_critic_evidence_gate_reasons"), ensure_ascii=False, sort_keys=True)
    total["evidence_risk_types"] = json.dumps(aggregate_json_counts(rows, "evidence_risk_types"), ensure_ascii=False, sort_keys=True)
    total["evidence_repair_reason_types"] = json.dumps(aggregate_json_counts(rows, "evidence_repair_reason_types"), ensure_ascii=False, sort_keys=True)
    total["qwen_fallback_reject_reasons"] = json.dumps(aggregate_json_counts(rows, "qwen_fallback_reject_reasons"), ensure_ascii=False, sort_keys=True)
    total["vision_reject_reasons"] = json.dumps(aggregate_json_counts(rows, "vision_reject_reasons"), ensure_ascii=False, sort_keys=True)
    total["vision_facts_reject_reasons"] = json.dumps(aggregate_json_counts(rows, "vision_facts_reject_reasons"), ensure_ascii=False, sort_keys=True)
    return total


def aggregate_json_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for item_key, item_value in payload.items():
            try:
                aggregate[str(item_key)] = aggregate.get(str(item_key), 0) + int(item_value)
            except (TypeError, ValueError):
                continue
    return aggregate


def write_summary_files(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quality_summary.json"
    csv_path = output_dir / "quality_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "profile",
        "page",
        "report",
        "kept_blocks",
        "skipped_blocks",
        "fallback_blocks",
        "ellipsis_outputs",
        "cat_used",
        "cat_bypassed",
        "cat_bypass_reasons",
        "cat_rejected",
        "cat_reject_reasons",
        "cat_chatter_rejected",
        "cat_retry_attempted",
        "cat_retry_accepted",
        "cat_retry_rejected",
        "cat_retry_reject_reasons",
        "cat_suspect_second_pass_attempted",
        "cat_suspect_second_pass_accepted",
        "cat_suspect_second_pass_rejected",
        "cat_suspect_second_pass_reject_reasons",
        "cat_q8_fallback_attempted",
        "cat_q8_fallback_accepted",
        "qwen_used",
        "qwen_rejected",
        "qwen_repairs_attempted",
        "qwen_repairs_accepted",
        "qwen_fallback_attempted",
        "qwen_fallback_accepted",
        "qwen_critic_attempted",
        "qwen_critic_flagged",
        "qwen_critic_issue_types",
        "qwen_critic_evidence_gate_reasons",
        "evidence_risk_blocks",
        "evidence_repair_reason_blocks",
        "evidence_risk_types",
        "evidence_repair_reason_types",
        "qwen_fallback_reject_reasons",
        "qwen_page_used",
        "qwen_page_rejected",
        "layout_warning_blocks",
        "layout_warning_types",
        "vision_attempted",
        "vision_accepted",
        "vision_rejected",
        "vision_json_repair_attempted",
        "vision_json_repair_used",
        "vision_reject_reasons",
        "vision_facts_attempted",
        "vision_facts_accepted",
        "vision_facts_rejected",
        "vision_facts_json_repair_attempted",
        "vision_facts_json_repair_used",
        "vision_facts_reject_reasons",
        "suspected_bad_translations",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
