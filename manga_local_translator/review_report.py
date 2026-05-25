from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .text_filter import count_japanese_chars, suspected_bad_translation


@dataclass(frozen=True)
class ReviewRow:
    run: str
    page: str
    line_id: str
    page_order: int
    source_text: str
    source_hash: str
    translated_text: str
    issues: tuple[str, ...]
    primary_translator: str = ""
    cat_raw_translation: str = ""
    cat_cleaned_translation: str = ""
    cat_final: str = ""
    cat_reject_reason: str = ""
    cat_bypass_reason: str = ""
    cat_retry_raw_translation: str = ""
    cat_retry_cleaned_translation: str = ""
    cat_retry_final: str = ""
    cat_retry_reject_reason: str = ""
    qwen_baseline: str = ""
    qwen_candidate: str = ""
    qwen_final: str = ""
    qwen_verify_choice: str = ""
    qwen_verify_reason: str = ""
    qwen_repair_candidate: str = ""
    qwen_repair_reject_reason: str = ""
    qwen_fallback_translation: str = ""
    qwen_fallback_reject_reason: str = ""
    qwen_critic_severity: str = ""
    qwen_critic_issues: tuple[str, ...] = ()
    qwen_critic_reason: str = ""
    qwen_critic_source_evidence: tuple[str, ...] = ()
    qwen_critic_translation_evidence: tuple[str, ...] = ()
    evidence_risk_flags: tuple[str, ...] = ()
    evidence_repair_reasons: tuple[str, ...] = ()
    context_before: str = ""
    context_after: str = ""
    visual_facts: tuple[str, ...] = ()
    layout_warnings: tuple[str, ...] = ()
    report_path: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m manga_local_translator.review_report",
        description="Build an HTML/CSV review report from manga translator .ocr.json debug files.",
    )
    parser.add_argument("input", type=Path, nargs="+", help="One or more .ocr.json files or folders containing .ocr.json reports.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path. Defaults to translation_review.html beside the input folder.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults to the HTML path with .csv suffix.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Human-readable Markdown output path. Defaults to the HTML path with .md suffix.",
    )
    parser.add_argument(
        "--suspicious-only",
        action="store_true",
        help="Only include rows with review issues.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = sorted({path.resolve() for input_path in args.input for path in iter_reports(input_path)}, key=lambda item: natural_sort_key(str(item)))
    if not reports:
        raise SystemExit(f"No .ocr.json reports found in: {', '.join(str(path) for path in args.input)}")

    rows: list[ReviewRow] = []
    for report_path in reports:
        rows.extend(rows_from_report(report_path, root_hint=common_root(reports)))
    if args.suspicious_only:
        rows = [row for row in rows if row.issues]

    output_path = args.output or default_output_path(args.input)
    csv_path = args.csv or output_path.with_suffix(".csv")
    markdown_path = args.markdown or output_path.with_suffix(".md")
    write_html_report(rows, output_path)
    write_csv_report(rows, csv_path)
    write_markdown_report(rows, markdown_path)
    print(f"Review report written: {output_path}")
    print(f"Review CSV written: {csv_path}")
    print(f"Review Markdown written: {markdown_path}")
    return 0


def iter_reports(path: Path):
    if path.is_file():
        if path.name.endswith(".ocr.json"):
            yield path
        return
    yield from sorted(path.rglob("*.ocr.json"), key=lambda item: natural_sort_key(str(item)))


def default_output_path(input_paths: list[Path]) -> Path:
    if len(input_paths) > 1:
        return Path("translation_review.html")
    input_path = input_paths[0]
    if input_path.is_file():
        return input_path.with_name("translation_review.html")
    return input_path / "translation_review.html"


def rows_from_report(report_path: Path, *, root_hint: Path | None = None) -> list[ReviewRow]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    page = report_path.stem.removesuffix(".ocr")
    run = report_group(report_path, root_hint=root_hint)
    rows = [
        row_from_block(run, page, block, report_path)
        for block in report.get("kept_blocks", [])
        if isinstance(block, dict)
    ]
    return sorted(rows, key=lambda row: (natural_sort_key(row.run), natural_sort_key(row.page), row.page_order))


def row_from_block(run: str, page: str, block: dict[str, Any], report_path: Path) -> ReviewRow:
    source = text_value(block.get("source_text"))
    translated = text_value(block.get("translated_text"))
    layout_warnings = tuple(str(value) for value in block.get("layout_warnings", []) if str(value).strip())
    visual_facts = tuple(str(value) for value in block.get("qwen_visual_facts_used") or block.get("visual_facts") or [] if str(value).strip())
    issues = detect_review_issues(block, source, translated, layout_warnings)
    return ReviewRow(
        run=run,
        page=page,
        line_id=text_value(block.get("line_id")),
        page_order=int(block.get("page_order") or 0),
        source_text=source,
        source_hash=text_value(block.get("source_hash")),
        translated_text=translated,
        issues=issues,
        primary_translator=text_value(block.get("primary_translator")),
        cat_raw_translation=text_value(block.get("cat_raw_translation")),
        cat_cleaned_translation=text_value(block.get("cat_cleaned_translation") or block.get("cat_candidate")),
        cat_final=text_value(block.get("cat_final")),
        cat_reject_reason=text_value(block.get("cat_reject_reason")),
        cat_bypass_reason=text_value(block.get("cat_bypass_reason")),
        cat_retry_raw_translation=text_value(block.get("cat_retry_raw_translation")),
        cat_retry_cleaned_translation=text_value(block.get("cat_retry_cleaned_translation")),
        cat_retry_final=text_value(block.get("cat_retry_final")),
        cat_retry_reject_reason=text_value(block.get("cat_retry_reject_reason")),
        qwen_baseline=text_value(block.get("qwen_baseline")),
        qwen_candidate=text_value(block.get("qwen_candidate")),
        qwen_final=text_value(block.get("qwen_final")),
        qwen_verify_choice=text_value(block.get("qwen_verify_choice")),
        qwen_verify_reason=text_value(block.get("qwen_verify_reason")),
        qwen_repair_candidate=text_value(block.get("qwen_repair_candidate")),
        qwen_repair_reject_reason=text_value(block.get("qwen_repair_reject_reason")),
        qwen_fallback_translation=text_value(block.get("qwen_fallback_translation")),
        qwen_fallback_reject_reason=text_value(block.get("qwen_fallback_reject_reason")),
        qwen_critic_severity=text_value(block.get("qwen_critic_severity")),
        qwen_critic_issues=tuple(str(value) for value in block.get("qwen_critic_issues") or [] if str(value).strip()),
        qwen_critic_reason=text_value(block.get("qwen_critic_reason")),
        qwen_critic_source_evidence=tuple(str(value) for value in block.get("qwen_critic_source_evidence") or [] if str(value).strip()),
        qwen_critic_translation_evidence=tuple(str(value) for value in block.get("qwen_critic_translation_evidence") or [] if str(value).strip()),
        evidence_risk_flags=tuple(str(value) for value in block.get("evidence_risk_flags") or [] if str(value).strip()),
        evidence_repair_reasons=tuple(str(value) for value in block.get("evidence_repair_reasons") or [] if str(value).strip()),
        context_before=text_value(block.get("context_before")),
        context_after=text_value(block.get("context_after")),
        visual_facts=visual_facts,
        layout_warnings=layout_warnings,
        report_path=str(report_path),
    )


def detect_review_issues(
    block: dict[str, Any],
    source: str,
    translated: str,
    layout_warnings: tuple[str, ...],
) -> tuple[str, ...]:
    issues: list[str] = []
    if not translated.strip() or translated.strip() == "...":
        issues.append("empty_or_ellipsis")
    if suspected_bad_translation(translated):
        issues.append("suspected_bad_translation")
    if block.get("cat_rejected") is True:
        issues.append("cat_rejected")
    if block.get("cat_bypassed") is True:
        issues.append("cat_bypassed")
    if block.get("cat_chatter_rejected") is True:
        issues.append("cat_chatter")
    if block.get("cat_reject_reason") == "cat_untranslated_japanese":
        issues.append("cat_untranslated")
    if block.get("cat_retry_attempted") is True:
        issues.append("cat_retry_attempted")
    if block.get("cat_retry_accepted") is True:
        issues.append("cat_retry_accepted")
    elif block.get("cat_retry_attempted") is True:
        issues.append("cat_retry_failed")
    if text_value(block.get("cat_final")) and len(text_value(block.get("cat_final"))) > 180:
        issues.append("cat_verbose")
    if block.get("cat_q8_fallback_accepted") is True:
        issues.append("cat_q8_changed")
    if count_japanese_chars(translated) > 0:
        issues.append("japanese_in_translation")
    if block.get("qwen_rejected") is True:
        issues.append("qwen_rejected")
    if block.get("qwen_repair_used") is True:
        issues.append("qwen_repair_used")
    if block.get("qwen_fallback_attempted") is True:
        issues.append("qwen_fallback_attempted")
    if block.get("qwen_critic_flagged") is True:
        severity = block.get("qwen_critic_severity") or "flagged"
        issues.append(f"qwen_critic:{severity}")
    if block.get("evidence_risk_flags"):
        issues.append("evidence_risk")
    if block.get("evidence_repair_reasons"):
        issues.append("evidence_repair_reason")
    if block.get("vision_accepted") is True:
        issues.append("vision_changed_text")
    if block.get("vision_facts_accepted") is True:
        issues.append("vision_facts_used")
    if block.get("vision_facts_reject_reason"):
        issues.append(f"vision_facts_rejected:{block.get('vision_facts_reject_reason')}")
    if layout_warnings:
        issues.extend(f"layout:{warning}" for warning in layout_warnings)
    issues.extend(pattern_issues(source, translated))
    return tuple(dict.fromkeys(issues))


def pattern_issues(source: str, translated: str) -> list[str]:
    issues: list[str] = []
    code_terms = [
        term
        for term in re.findall(r"\u3008([^>\u3009]{1,20})\u3009", source)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", term) and (re.search(r"\d", term) or term.upper() == term)
    ]
    if code_terms and not any(term.lower() in translated.lower() for term in code_terms):
        issues.append("possibly_dropped_bracket_term")
    if re.search(r"\b[A-Za-z]+(?:-[A-Za-z]+){4,}\b", translated):
        issues.append("malformed_hyphen_chain")
    return issues


def write_csv_report(rows: list[ReviewRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_row(rows[0]).keys()) if rows else default_csv_fields())
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row))


def csv_row(row: ReviewRow) -> dict[str, str | int]:
    return {
        "run": row.run,
        "page": row.page,
        "line_id": row.line_id,
        "page_order": row.page_order,
        "issues": "; ".join(row.issues),
        "source_hash": row.source_hash,
        "source_text": row.source_text,
        "translated_text": row.translated_text,
        "primary_translator": row.primary_translator,
        "cat_raw_translation": row.cat_raw_translation,
        "cat_cleaned_translation": row.cat_cleaned_translation,
        "cat_final": row.cat_final,
        "cat_reject_reason": row.cat_reject_reason,
        "cat_bypass_reason": row.cat_bypass_reason,
        "cat_retry_raw_translation": row.cat_retry_raw_translation,
        "cat_retry_cleaned_translation": row.cat_retry_cleaned_translation,
        "cat_retry_final": row.cat_retry_final,
        "cat_retry_reject_reason": row.cat_retry_reject_reason,
        "qwen_baseline": row.qwen_baseline,
        "qwen_candidate": row.qwen_candidate,
        "qwen_final": row.qwen_final,
        "qwen_verify_choice": row.qwen_verify_choice,
        "qwen_verify_reason": row.qwen_verify_reason,
        "qwen_repair_candidate": row.qwen_repair_candidate,
        "qwen_repair_reject_reason": row.qwen_repair_reject_reason,
        "qwen_fallback_translation": row.qwen_fallback_translation,
        "qwen_fallback_reject_reason": row.qwen_fallback_reject_reason,
        "qwen_critic_severity": row.qwen_critic_severity,
        "qwen_critic_issues": "; ".join(row.qwen_critic_issues),
        "qwen_critic_reason": row.qwen_critic_reason,
        "qwen_critic_source_evidence": " | ".join(row.qwen_critic_source_evidence),
        "qwen_critic_translation_evidence": " | ".join(row.qwen_critic_translation_evidence),
        "evidence_risk_flags": "; ".join(row.evidence_risk_flags),
        "evidence_repair_reasons": "; ".join(row.evidence_repair_reasons),
        "context_before": row.context_before,
        "context_after": row.context_after,
        "visual_facts": " | ".join(row.visual_facts),
        "layout_warnings": "; ".join(row.layout_warnings),
        "report_path": row.report_path,
    }


def default_csv_fields() -> list[str]:
    return list(csv_row(ReviewRow("", "", "", 0, "", "", "", ())).keys())


def write_html_report(rows: list[ReviewRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    issue_counts: dict[str, int] = {}
    for row in rows:
        for issue in row.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    html_text = render_html(rows, issue_counts)
    path.write_text(html_text, encoding="utf-8")


def write_markdown_report(rows: list[ReviewRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(rows), encoding="utf-8")


def render_markdown(rows: list[ReviewRow]) -> str:
    issue_counts = Counter(issue for row in rows for issue in row.issues)
    run_counts: dict[str, list[int]] = {}
    page_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        total, flagged = run_counts.setdefault(row.run, [0, 0])
        total += 1
        if row.issues:
            flagged += 1
            page_counts[(row.run, row.page)] += 1
        run_counts[row.run] = [total, flagged]

    suspicious = [row for row in rows if row.issues]
    cat_reject_counts = Counter(row.cat_reject_reason for row in rows if row.cat_reject_reason)
    cat_retry_rows = [
        row
        for row in rows
        if row.cat_retry_raw_translation or row.cat_retry_cleaned_translation or row.cat_retry_final or row.cat_retry_reject_reason
    ]
    cat_retry_reject_counts = Counter(row.cat_retry_reject_reason for row in cat_retry_rows if row.cat_retry_reject_reason)
    cat_q8_changed = [row for row in rows if "cat_q8_changed" in row.issues]

    lines: list[str] = [
        "# Manga Translation Human Review",
        "",
        f"- Total reviewed lines: **{len(rows)}**",
        f"- Lines with review flags: **{len(suspicious)}**",
        f"- CAT rejected lines: **{sum(1 for row in rows if row.cat_reject_reason)}**",
        f"- CAT retries: **{len(cat_retry_rows)}**",
        f"- CAT retries still rejected: **{sum(1 for row in cat_retry_rows if row.cat_retry_reject_reason)}**",
        f"- CAT -> Q8 accepted changes: **{len(cat_q8_changed)}**",
        "",
        "## Top Review Flags",
        "",
    ]
    if issue_counts:
        for issue, count in issue_counts.most_common(20):
            lines.append(f"- `{issue}`: {count}")
    else:
        lines.append("- No review flags found.")

    lines.extend(["", "## Run / Chapter Summary", "", "| Run / Chapter | Lines | Flagged lines |", "|---|---:|---:|"])
    for run, (total, flagged) in sorted(run_counts.items(), key=lambda item: natural_sort_key(item[0])):
        lines.append(f"| {markdown_cell(run, 120)} | {total} | {flagged} |")

    lines.extend(["", "## Pages With Most Flags", ""])
    if page_counts:
        lines.extend(["| Page | Flagged lines |", "|---|---:|"])
        for (run, page), count in page_counts.most_common(20):
            lines.append(f"| {markdown_cell(run, 80)} / `{markdown_cell(page, 40)}` | {count} |")
    else:
        lines.append("- No flagged pages.")

    lines.extend(["", "## CAT Rejections", ""])
    if cat_reject_counts:
        for reason, count in cat_reject_counts.most_common():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- No CAT rejections.")

    if cat_retry_rows:
        lines.extend(["", "## CAT Retries", ""])
        lines.extend(["| Page | Order | Source | Retry cleaned/final | Retry reject |", "|---|---:|---|---|---|"])
        for row in cat_retry_rows[:80]:
            lines.append(
                "| {page} | {order} | {source} | {retry} | {reject} |".format(
                    page=f"{markdown_cell(row.run, 60)} / `{markdown_cell(row.page, 20)}`",
                    order=markdown_cell(str(row.page_order), 8),
                    source=markdown_cell(row.source_text, 120),
                    retry=markdown_cell(row.cat_retry_cleaned_translation or row.cat_retry_final or row.cat_retry_raw_translation, 160),
                    reject=markdown_cell(row.cat_retry_reject_reason or "accepted", 80),
                )
            )
        if len(cat_retry_rows) > 80:
            lines.append(f"- ... {len(cat_retry_rows) - 80} more CAT retry rows omitted from this section.")
        if cat_retry_reject_counts:
            lines.append("")
            lines.append(
                "Retry reject reasons: "
                + ", ".join(f"`{reason}`={count}" for reason, count in cat_retry_reject_counts.most_common())
            )

    lines.extend(["", "## Q8 Rescues After CAT", ""])
    if cat_q8_changed:
        lines.extend(["| Page | Order | Source | CAT cleaned/final | Q8 replacement |", "|---|---:|---|---|---|"])
        for row in cat_q8_changed[:40]:
            lines.append(
                "| {page} | {order} | {source} | {cat} | {q8} |".format(
                    page=f"{markdown_cell(row.run, 60)} / `{markdown_cell(row.page, 20)}`",
                    order=markdown_cell(str(row.page_order), 8),
                    source=markdown_cell(row.source_text, 120),
                    cat=markdown_cell(row.cat_cleaned_translation or row.cat_final, 140),
                    q8=markdown_cell(row.qwen_fallback_translation or row.translated_text, 140),
                )
            )
    else:
        lines.append("- No accepted Q8 rescues after CAT.")

    lines.extend(["", "## Rows Needing Human Review", ""])
    if suspicious:
        grouped: dict[tuple[str, str], list[ReviewRow]] = defaultdict(list)
        for row in suspicious:
            grouped[(row.run, row.page)].append(row)
        for (run, page), group in sorted(grouped.items(), key=lambda item: (natural_sort_key(item[0][0]), natural_sort_key(item[0][1]))):
            lines.extend(
                [
                    f"### {markdown_cell(run, 120)} / {markdown_cell(page, 80)}",
                    "",
                    "| Order | Flags | Source | Final English | CAT | Critic / Q8 notes |",
                    "|---:|---|---|---|---|---|",
                ]
            )
            for row in group:
                lines.append(
                    "| {order} | {issues} | {source} | {final} | {cat} | {notes} |".format(
                        order=markdown_cell(str(row.page_order), 8),
                        issues="<br>".join(f"`{markdown_cell(issue, 60)}`" for issue in row.issues),
                        source=markdown_cell(row.source_text, 140),
                        final=markdown_cell(row.translated_text, 140),
                        cat=markdown_cell(markdown_cat_summary(row), 160),
                        notes=markdown_cell(markdown_notes_summary(row), 180),
                    )
                )
            lines.append("")
    else:
        lines.append("- No flagged rows.")

    lines.extend(
        [
            "",
            "## Suggested Manual Review Order",
            "",
            "1. Check `empty_or_ellipsis`, `suspected_bad_translation`, and final `...` rows first.",
            "2. Review `cat_rejected`, `cat_chatter`, and `cat_q8_changed` rows to judge whether Q8 rescue helped.",
            "3. Review critic-flagged evidence rows before making prompt or validation changes.",
            "4. Treat layout-only rows separately from translation-quality rows.",
            "5. Avoid global phrasebook/prompt changes from one chapter-specific example.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_cat_summary(row: ReviewRow) -> str:
    parts: list[str] = []
    if row.cat_reject_reason:
        parts.append(f"reject: {row.cat_reject_reason}")
    if row.cat_bypass_reason:
        parts.append(f"bypass: {row.cat_bypass_reason}")
    if row.cat_retry_final or row.cat_retry_cleaned_translation or row.cat_retry_raw_translation or row.cat_retry_reject_reason:
        retry_text = row.cat_retry_cleaned_translation or row.cat_retry_final or row.cat_retry_raw_translation
        retry_status = row.cat_retry_reject_reason or "accepted"
        parts.append(f"retry {retry_status}: {retry_text}")
    if row.cat_cleaned_translation:
        parts.append(row.cat_cleaned_translation)
    elif row.cat_final:
        parts.append(row.cat_final)
    elif row.cat_raw_translation:
        parts.append(row.cat_raw_translation)
    return " / ".join(parts)


def markdown_notes_summary(row: ReviewRow) -> str:
    parts: list[str] = []
    if row.qwen_critic_issues:
        parts.append("critic: " + "; ".join(row.qwen_critic_issues))
    if row.qwen_critic_reason:
        parts.append(row.qwen_critic_reason)
    if row.qwen_fallback_translation and "cat_q8_changed" in row.issues:
        parts.append("Q8 accepted: " + row.qwen_fallback_translation)
    if row.qwen_fallback_reject_reason:
        parts.append("Q8 reject: " + row.qwen_fallback_reject_reason)
    return " / ".join(parts)


def markdown_cell(value: str, limit: int) -> str:
    text = repair_common_mojibake(str(value or "").replace("\r\n", "\n")).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "?"
    return text.replace("|", "\\|")


def repair_common_mojibake(text: str) -> str:
    if not text or not any(marker in text for marker in ("ã", "æ", "é", "ï", "â")):
        return text
    original_score = japanese_char_count(text)
    best = text
    best_score = original_score
    for encoding in ("cp1252", "latin1"):
        try:
            fixed = text.encode(encoding, errors="strict").decode("utf-8", errors="strict")
        except UnicodeError:
            continue
        score = japanese_char_count(fixed)
        if score > best_score:
            best = fixed
            best_score = score
    return best


def japanese_char_count(text: str) -> int:
    return sum(
        0x3040 <= ord(char) <= 0x30FF
        or 0x31F0 <= ord(char) <= 0x31FF
        or 0x3400 <= ord(char) <= 0x4DBF
        or 0x4E00 <= ord(char) <= 0x9FFF
        for char in text
    )


def render_html(rows: list[ReviewRow], issue_counts: dict[str, int]) -> str:
    suspicious = sum(1 for row in rows if row.issues)
    run_summary = render_run_summary(rows)
    issue_summary = "".join(
        f"<li><code>{escape(issue)}</code>: {count}</li>"
        for issue, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    row_html = "\n".join(render_row(row) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Manga Translation Review</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f7f7f5; color: #181818; }}
h1 {{ margin-bottom: 4px; }}
.summary {{ display: flex; gap: 24px; flex-wrap: wrap; margin: 16px 0; }}
.panel {{ background: white; border: 1px solid #d7d7d2; border-radius: 6px; padding: 12px 14px; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #d7d7d2; padding: 8px; vertical-align: top; font-size: 13px; }}
th {{ position: sticky; top: 0; background: #ecece8; z-index: 1; }}
tr.issue {{ background: #fff9e8; }}
code {{ background: #ecece8; padding: 1px 4px; border-radius: 3px; }}
.muted {{ color: #666; }}
.text {{ max-width: 340px; white-space: pre-wrap; }}
.issues {{ max-width: 220px; }}
</style>
</head>
<body>
<h1>Manga Translation Review</h1>
<div class="muted">Generated from debug <code>.ocr.json</code> files.</div>
<div class="summary">
  <div class="panel"><strong>Total rows:</strong> {len(rows)}<br><strong>Suspicious rows:</strong> {suspicious}</div>
  <div class="panel"><strong>Issue counts</strong><ul>{issue_summary or "<li>None</li>"}</ul></div>
  <div class="panel"><strong>Runs</strong>{run_summary}</div>
</div>
<table>
<thead>
<tr>
  <th>Run</th><th>Page</th><th>Line</th><th>Order</th><th>Issues</th><th>Source OCR</th><th>Final</th>
  <th>CAT</th><th>Baseline / Candidate / Repair</th><th>Verifier</th><th>Context</th><th>Visual Facts</th>
</tr>
</thead>
<tbody>
{row_html}
</tbody>
</table>
</body>
</html>
"""


def render_row(row: ReviewRow) -> str:
    klass = "issue" if row.issues else ""
    baseline_bits = [
        label_value("Baseline", row.qwen_baseline),
        label_value("Candidate", row.qwen_candidate),
        label_value("Repair", row.qwen_repair_candidate),
        label_value("Q8 fallback", row.qwen_fallback_translation),
        label_value("Critic severity", row.qwen_critic_severity),
        label_value("Critic issues", "; ".join(row.qwen_critic_issues)),
        label_value("Evidence risks", "; ".join(row.evidence_risk_flags)),
        label_value("Evidence repair", "; ".join(row.evidence_repair_reasons)),
    ]
    cat_bits = [
        label_value("Primary", row.primary_translator),
        label_value("Raw", row.cat_raw_translation),
        label_value("Cleaned", row.cat_cleaned_translation),
        label_value("Final", row.cat_final),
        label_value("Reject", row.cat_reject_reason),
        label_value("Bypass", row.cat_bypass_reason),
        label_value("Retry raw", row.cat_retry_raw_translation),
        label_value("Retry cleaned", row.cat_retry_cleaned_translation),
        label_value("Retry final", row.cat_retry_final),
        label_value("Retry reject", row.cat_retry_reject_reason),
    ]
    verifier_bits = [
        label_value("Choice", row.qwen_verify_choice),
        label_value("Reason", row.qwen_verify_reason),
        label_value("Repair reject", row.qwen_repair_reject_reason),
        label_value("Fallback reject", row.qwen_fallback_reject_reason),
        label_value("Critic reason", row.qwen_critic_reason),
        label_value("Critic source evidence", " | ".join(row.qwen_critic_source_evidence)),
        label_value("Critic translation evidence", " | ".join(row.qwen_critic_translation_evidence)),
    ]
    context_bits = [
        label_value("Before", row.context_before),
        label_value("After", row.context_after),
    ]
    return f"""<tr class="{klass}">
<td>{escape(row.run)}</td>
<td>{escape(row.page)}</td>
<td><code>{escape(row.line_id)}</code></td>
<td>{row.page_order}</td>
<td class="issues">{render_issue_list(row.issues)}</td>
<td class="text">{escape(row.source_text)}</td>
<td class="text">{escape(row.translated_text)}</td>
<td class="text">{''.join(cat_bits)}</td>
<td class="text">{''.join(baseline_bits)}</td>
<td class="text">{''.join(verifier_bits)}</td>
<td class="text">{''.join(context_bits)}</td>
<td class="text">{escape(' | '.join(row.visual_facts))}</td>
</tr>"""


def render_run_summary(rows: list[ReviewRow]) -> str:
    counts: dict[str, list[int]] = {}
    for row in rows:
        total, suspicious = counts.setdefault(row.run, [0, 0])
        total += 1
        if row.issues:
            suspicious += 1
        counts[row.run] = [total, suspicious]
    items = "".join(
        f"<li><code>{escape(run)}</code>: {total} rows, {suspicious} suspicious</li>"
        for run, (total, suspicious) in sorted(counts.items(), key=lambda item: natural_sort_key(item[0]))
    )
    return f"<ul>{items}</ul>" if items else "<ul><li>None</li></ul>"


def label_value(label: str, value: str) -> str:
    if not value:
        return ""
    return f"<strong>{escape(label)}:</strong> {escape(value)}<br>"


def render_issue_list(issues: tuple[str, ...]) -> str:
    if not issues:
        return ""
    return "<br>".join(f"<code>{escape(issue)}</code>" for issue in issues)


def text_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def natural_sort_key(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def common_root(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    try:
        import os

        return Path(os.path.commonpath([str(path.parent.resolve()) for path in paths]))
    except (OSError, ValueError):
        return None


def report_group(report_path: Path, *, root_hint: Path | None) -> str:
    parent = report_path.parent.resolve()
    root = root_hint.resolve() if root_hint is not None else None
    if root is not None:
        try:
            relative = parent.relative_to(root)
        except ValueError:
            relative = parent
        text = str(relative).replace("\\", "/")
        return text if text and text != "." else parent.name
    return str(parent).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
