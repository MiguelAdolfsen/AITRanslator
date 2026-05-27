from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_REFERENCE_FIELDS = {
    "page_id",
    "line_id",
    "page_order",
    "source_text",
    "source_box",
    "block_id",
    "source_hash",
    "reference_text",
    "alignment_confidence",
    "reference_role",
    "skip_reference",
    "skip_reason",
    "notes",
}
ALLOWED_ALIGNMENT = {"low", "medium", "high"}
ALLOWED_ROLES = {"speech", "thought", "narration", "sfx", "metadata", "sign"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(payload)
    return rows


def load_label_files(benchmark: Path) -> list[dict[str, Any]]:
    labels_dir = benchmark / "labels"
    pages: list[dict[str, Any]] = []
    for path in sorted(labels_dir.glob("*.references.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected object")
        payload["_label_path"] = str(path)
        pages.append(payload)
    return pages


def labels_from_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for page in pages:
        page_labels = page.get("labels")
        if not isinstance(page_labels, list):
            raise ValueError(f"{page.get('_label_path', page.get('page_id'))}: expected labels list")
        for label in page_labels:
            if not isinstance(label, dict):
                raise ValueError(f"{page.get('_label_path')}: expected label object")
            labels.append(label)
    return labels


def validate_benchmark(benchmark: Path) -> list[str]:
    benchmark = benchmark.resolve()
    errors: list[str] = []
    manifest_path = benchmark / "manifest.json"
    references_path = benchmark / "references.jsonl"
    labels_dir = benchmark / "labels"
    if not manifest_path.exists():
        return [f"missing manifest: {manifest_path}"]
    if not references_path.exists():
        errors.append(f"missing references: {references_path}")
    if not labels_dir.exists():
        errors.append(f"missing labels dir: {labels_dir}")
    if errors:
        return errors

    try:
        manifest = read_json(manifest_path)
        references = read_jsonl(references_path)
        pages = load_label_files(benchmark)
        label_rows = labels_from_pages(pages)
    except Exception as exc:
        return [str(exc)]

    expected_pages = list(manifest.get("pages") or [])
    if not expected_pages:
        errors.append("manifest.pages must list benchmark page IDs")
    page_ids = [str(page.get("page_id", "")) for page in pages]
    if expected_pages and page_ids != expected_pages:
        errors.append(f"label page order does not match manifest.pages: {page_ids} != {expected_pages}")

    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        errors.append("manifest.splits is required")
    else:
        main_pages = set(str(value) for value in splits.get("main", []) or [])
        holdout_pages = set(str(value) for value in splits.get("holdout", []) or [])
        unknown = sorted((main_pages | holdout_pages) - set(expected_pages))
        overlap = sorted(main_pages & holdout_pages)
        if unknown:
            errors.append(f"manifest.splits references unknown pages: {unknown}")
        if overlap:
            errors.append(f"manifest.splits main/holdout overlap: {overlap}")
        if not main_pages or not holdout_pages:
            errors.append("manifest.splits must include non-empty main and holdout lists")

    reference_counter = Counter(canonical_json(row) for row in references)
    label_counter = Counter(canonical_json(row) for row in label_rows)
    if reference_counter != label_counter:
        errors.append("references.jsonl does not exactly match combined labels[].")

    seen_line_ids: set[tuple[str, str]] = set()
    for index, row in enumerate(references, start=1):
        errors.extend(validate_reference_row(row, index=index, seen_line_ids=seen_line_ids))

    for page in pages:
        errors.extend(validate_label_page(page))
        cache_path = Path(str(page.get("prepared_cache", "")))
        if cache_path.exists():
            errors.extend(validate_prepared_cache(page, cache_path))

    return errors


def validate_reference_row(row: dict[str, Any], *, index: int, seen_line_ids: set[tuple[str, str]]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_REFERENCE_FIELDS - set(row)
    if missing:
        errors.append(f"references.jsonl:{index}: missing fields: {sorted(missing)}")
    key = (str(row.get("page_id", "")), str(row.get("line_id", "")))
    if key in seen_line_ids:
        errors.append(f"references.jsonl:{index}: duplicate page_id/line_id: {key}")
    seen_line_ids.add(key)
    box = row.get("source_box")
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(value, int) for value in box)):
        errors.append(f"references.jsonl:{index}: invalid source_box for {key}: {box!r}")
    if row.get("alignment_confidence") not in ALLOWED_ALIGNMENT:
        errors.append(f"references.jsonl:{index}: invalid alignment_confidence for {key}")
    if row.get("reference_role") not in ALLOWED_ROLES:
        errors.append(f"references.jsonl:{index}: invalid reference_role for {key}")
    if row.get("skip_reference") is True and not str(row.get("skip_reason", "")).strip():
        errors.append(f"references.jsonl:{index}: skipped row missing skip_reason for {key}")
    if row.get("skip_reference") is False and not str(row.get("reference_text", "")).strip():
        errors.append(f"references.jsonl:{index}: scored row missing reference_text for {key}")
    return errors


def validate_label_page(page: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label_path = str(page.get("_label_path", page.get("page_id", "<unknown>")))
    for key in ("japanese_image", "english_image", "prepared_cache"):
        value = str(page.get(key, "") or "")
        if not value:
            errors.append(f"{label_path}: missing {key}")
        elif not Path(value).exists():
            errors.append(f"{label_path}: {key} path does not exist: {value}")
    labels = page.get("labels") if isinstance(page.get("labels"), list) else []
    expected_page_id = str(page.get("page_id", ""))
    for label in labels:
        if str(label.get("page_id", "")) != expected_page_id:
            errors.append(f"{label_path}: label page_id mismatch for {label.get('line_id')}")
    return errors


def validate_prepared_cache(page: dict[str, Any], cache_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        cache = read_json(cache_path)
    except Exception as exc:
        return [f"{cache_path}: invalid prepared cache JSON: {exc}"]
    by_line = {
        str(item.get("line_id", "")): item
        for item in cache.get("page_order_report", []) or []
        if item.get("line_id")
    }
    for label in page.get("labels", []) or []:
        line_id = str(label.get("line_id", ""))
        item = by_line.get(line_id)
        if item is None:
            errors.append(f"{cache_path}: missing line_id from page_order_report: {line_id}")
            continue
        if str(item.get("source_text", "")) != str(label.get("source_text", "")):
            errors.append(f"{cache_path}: source_text mismatch for {line_id}")
        if list(item.get("box") or []) != list(label.get("source_box") or []):
            errors.append(f"{cache_path}: source_box mismatch for {line_id}")
        if str(item.get("source_hash", "")) != str(label.get("source_hash", "")):
            errors.append(f"{cache_path}: source_hash mismatch for {line_id}")
    return errors


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a translation quality benchmark.")
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args(argv)
    errors = validate_benchmark(args.benchmark)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"benchmark ok: {args.benchmark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

