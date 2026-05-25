from __future__ import annotations

import argparse
import csv
import html
import json
import re
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
    write_html_report(rows, output_path)
    write_csv_report(rows, csv_path)
    print(f"Review report written: {output_path}")
    print(f"Review CSV written: {csv_path}")
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
    if block.get("cat_chatter_rejected") is True:
        issues.append("cat_chatter")
    if block.get("cat_reject_reason") == "cat_untranslated_japanese":
        issues.append("cat_untranslated")
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
