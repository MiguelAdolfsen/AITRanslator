from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_quality_autoresearch.common.hashing import sha256_text
from translation_quality_autoresearch.common.io_utils import write_jsonl

DEFAULT_FORBIDDEN_PATTERNS = ["I cannot", "please provide", "as an AI", "Translation:", "Japanese:", "English:"]
DEFAULT_SELECTION_TARGETS = {
    "sfx": 6,
    "reaction": 6,
    "ocr_noisy_but_salvageable": 8,
    "names_and_honorifics": 8,
    "ambiguous_subject": 5,
    "agent_rejected_candidate": 5,
    "ellipsis": 8,
    "short_fragment": 8,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine candidate benchmark cases from quality-run OCR debug reports.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-source-chars", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--case-prefix", default="generated_quality")
    parser.add_argument("--allow-duplicate-source", action="store_true")
    parser.add_argument("--round-robin-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --overwrite")
    rows = mine_cases(
        args.input,
        min_source_chars=args.min_source_chars,
        max_cases=args.max_cases,
        include_context=args.include_context,
        case_prefix=args.case_prefix,
        dedupe_source=not args.allow_duplicate_source,
        diverse=not args.round_robin_only,
    )
    if args.dry_run:
        print(f"would write {len(rows)} generated cases to {args.output}")
        return 0
    if "translation_quality_autoresearch" not in args.output.as_posix() or "/benchmark/" not in args.output.as_posix().replace("\\", "/"):
        raise SystemExit("output must stay inside translation_quality_autoresearch/benchmark/")
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} generated cases to {args.output}")
    return 0


def mine_cases(
    root: Path,
    *,
    min_source_chars: int,
    max_cases: int,
    include_context: bool,
    case_prefix: str = "generated_quality",
    dedupe_source: bool = True,
    diverse: bool = True,
) -> list[dict[str, Any]]:
    candidates_by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_hashes: set[str] = set()
    for report_path in sorted(root.rglob("*.ocr.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        blocks = report.get("kept_blocks") or []
        for index, block in enumerate(blocks):
            source = str(block.get("source_text") or block.get("text") or "").strip()
            if len(source) < min_source_chars:
                continue
            source_hash = sha256_text(source)
            if dedupe_source and source_hash in seen_hashes:
                continue
            seen_hashes.add(source_hash)
            row = build_case_row(
                report_path=report_path,
                report=report,
                block=block,
                block_index=index,
                source=source,
                source_hash=source_hash,
                include_context=include_context,
            )
            key = str(report.get("source_image") or report_path)
            candidates_by_report[key].append(row)

    ranked_rows = select_round_robin(candidates_by_report, max_cases=10_000)
    rows = select_diverse(ranked_rows, max_cases=max_cases) if diverse else ranked_rows[:max_cases]
    for index, row in enumerate(rows, start=1):
        row["case_id"] = f"{case_prefix}_{index:05d}"
    return rows


def build_case_row(
    *,
    report_path: Path,
    report: dict[str, Any],
    block: dict[str, Any],
    block_index: int,
    source: str,
    source_hash: str,
    include_context: bool,
) -> dict[str, Any]:
    metadata = dict(block.get("metadata") or {})
    source_image = str(report.get("source_image") or "")
    page_id = Path(source_image).stem if source_image else report_path.stem.replace(".ocr", "")
    tags = infer_tags(source, block=block, report=report)
    row = {
        "case_id": "generated_quality_PENDING",
        "page_id": page_id,
        "group_id": str(block.get("translation_group_id") or block.get("line_id") or block_index + 1),
        "source_text": source,
        "normalized_source": source,
        "source_hash": source_hash,
        "context_before": [],
        "context_after": [],
        "speaker_hint": "unknown",
        "ocr_risk": infer_ocr_risk(source, block),
        "source_type": infer_source_type(source),
        "orientation": infer_orientation(block),
        "box": block.get("box") or [0, 0, 0, 0],
        "page_order": block.get("page_order") or block_index + 1,
        "glossary_terms": [],
        "must_preserve": [],
        "forbidden_patterns": DEFAULT_FORBIDDEN_PATTERNS,
        "style_notes": style_notes_for(source, tags),
        "tags": tags,
        "metadata": {
            "generated_from_quality_run": True,
            "needs_human_review": True,
            "source_report": str(report_path),
            "source_image": source_image,
            "output_image": str(report.get("output_image") or ""),
            "detector": str(report.get("detector") or ""),
            "ocr_engine": str(report.get("ocr_engine") or ""),
            "translator": str(report.get("translator") or ""),
            "qwen_mode": str(report.get("qwen_mode") or ""),
            "block_id": str(block.get("block_id") or ""),
            "line_id": str(block.get("line_id") or ""),
            "original_source_hash": str(block.get("source_hash") or ""),
            "confidence": block.get("confidence"),
            "draft_machine_translation": str(block.get("translated_text") or ""),
            "primary_translator": str(block.get("primary_translator") or ""),
            "translation_mode": str(block.get("translation_mode") or ""),
            "cat_rejected": bool(block.get("cat_rejected")),
            "cat_reject_reason": str(block.get("cat_reject_reason") or ""),
            "qwen_rejected": bool(block.get("qwen_rejected")),
            "qwen_reject_reason": str(block.get("qwen_reject_reason") or ""),
            "source_features": block.get("source_features") or {},
            "raw_metadata": metadata,
        },
    }
    if include_context:
        row["context_before"] = context_values(block, "context_before", "context_before_window")
        row["context_after"] = context_values(block, "context_after", "context_after_window")
    if source_mentions_honorific(source):
        row["style_notes"].append("preserve names and honorifics unless a project glossary says otherwise")
    return row


def select_round_robin(candidates_by_report: dict[str, list[dict[str, Any]]], *, max_cases: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(candidates_by_report)
    depth = 0
    while len(rows) < max_cases:
        added = False
        for key in keys:
            candidates = candidates_by_report[key]
            if depth < len(candidates):
                rows.append(candidates[depth])
                added = True
                if len(rows) >= max_cases:
                    break
        if not added:
            break
        depth += 1
    return rows


def select_diverse(rows: list[dict[str, Any]], *, max_cases: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    for tag, target in DEFAULT_SELECTION_TARGETS.items():
        for row in rows:
            if len(selected) >= max_cases:
                return selected
            if tag not in row.get("tags", []):
                continue
            if row["source_hash"] in selected_hashes:
                continue
            selected.append(row)
            selected_hashes.add(row["source_hash"])
            if sum(1 for item in selected if tag in item.get("tags", [])) >= target:
                break
    for row in rows:
        if len(selected) >= max_cases:
            break
        if row["source_hash"] in selected_hashes:
            continue
        selected.append(row)
        selected_hashes.add(row["source_hash"])
    return selected


def context_values(block: dict[str, Any], scalar_key: str, window_key: str) -> list[str]:
    values: list[str] = []
    window = block.get(window_key)
    if isinstance(window, list):
        values.extend(str(value).strip() for value in window if str(value).strip())
    scalar = block.get(scalar_key)
    if scalar and str(scalar).strip() and str(scalar).strip() not in values:
        values.append(str(scalar).strip())
    return values[:3]


def infer_orientation(block: dict[str, Any]) -> str:
    metadata = block.get("metadata") or {}
    if block.get("vertical") is True or metadata.get("vertical") is True:
        return "vertical"
    if block.get("vertical") is False or metadata.get("vertical") is False:
        return "horizontal"
    box = block.get("box") or []
    if len(box) == 4:
        width = abs(float(box[2]) - float(box[0]))
        height = abs(float(box[3]) - float(box[1]))
        if height > width * 1.4:
            return "vertical"
        if width > height * 1.4:
            return "horizontal"
    return "unknown"


def infer_ocr_risk(source: str, block: dict[str, Any]) -> str:
    confidence = block.get("confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.75:
        return "noisy_salvageable"
    features = block.get("source_features") or {}
    if features.get("ocr_risk") or features.get("has_suspicious_spacing"):
        return "noisy_salvageable"
    if " " in source or "�" in source:
        return "noisy_salvageable"
    return "clean"


def infer_source_type(source: str) -> str:
    compact = source.strip(" 　！？!?…ー～・.。")
    katakana = sum(1 for char in compact if "\u30a1" <= char <= "\u30fa")
    kana = sum(1 for char in compact if ("\u3040" <= char <= "\u309f") or ("\u30a1" <= char <= "\u30fa") or char == "ー")
    sfx_chars = set("ドゴガバザダパビピグギズンッァォャュョー")
    katakana_like = all(("\u30a1" <= char <= "\u30fa") or char == "ー" for char in compact)
    if (
        compact
        and len(compact) <= 8
        and katakana_like
        and any(char in sfx_chars for char in compact)
        and any(mark in source for mark in "！？!?")
    ):
        return "sfx"
    if compact and len(compact) <= 5 and kana == len(compact):
        return "reaction"
    return "dialogue"


def infer_tags(source: str, *, block: dict[str, Any], report: dict[str, Any]) -> list[str]:
    tags = ["generated", "needs_human_review"]
    source_type = infer_source_type(source)
    if source_type == "sfx":
        tags.extend(["sfx", "reaction"])
    elif source_type == "reaction":
        tags.append("reaction")
    if len(source) <= 8:
        tags.append("short_fragment")
    if any(mark in source for mark in ["…", "．．．", "..."]):
        tags.append("ellipsis")
    if block.get("context_before") or block.get("context_after") or block.get("context_before_window") or block.get("context_after_window"):
        tags.append("page_context_available")
    if source_mentions_honorific(source):
        tags.append("names_and_honorifics")
    if any(term in source for term in ["あの人", "あの子", "彼", "彼女", "こいつ", "あいつ"]):
        tags.append("ambiguous_subject")
    if infer_ocr_risk(source, block) != "clean":
        tags.append("ocr_noisy_but_salvageable")
    if block.get("cat_rejected") or block.get("qwen_rejected"):
        tags.append("agent_rejected_candidate")
    summary = report.get("page_summary") or {}
    if summary.get("vision_attempted") or summary.get("vision_facts_attempted"):
        tags.append("vision_context_available")
    return list(dict.fromkeys(tags))


def source_mentions_honorific(source: str) -> bool:
    return any(term in source for term in ["さん", "君", "ちゃん", "様", "先輩", "先生", "殿"])


def style_notes_for(source: str, tags: list[str]) -> list[str]:
    notes = ["generated from quality run; needs human reference before scoring"]
    if "short_fragment" in tags:
        notes.append("keep short manga fragments concise")
    if "sfx" in tags:
        notes.append("sound effect or reaction text")
    if "page_context_available" in tags:
        notes.append("context is available for ambiguity, but do not invent facts")
    if len(source) > 30:
        notes.append("avoid over-explaining long dialogue")
    return notes


if __name__ == "__main__":
    raise SystemExit(main())
