from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASE_REQUIRED = {"schema_version", "case_id", "source_text", "source_type", "risk_labels"}
REFERENCE_REQUIRED = {
    "schema_version",
    "case_id",
    "expected_decision",
    "forbidden_patterns",
    "allowed_japanese_output",
    "max_chars",
    "max_words",
}
VALID_DECISIONS = {"accept", "reject"}
VALID_SOURCE_TYPES = {
    "ellipsis_fragment",
    "short_fragment",
    "punctuation_fragment",
    "normal_dialogue",
    "long_dialogue",
    "interrupted_speech",
    "honorific_name",
    "name_or_term",
    "sfx",
    "ocr_noise",
    "credits_or_metadata",
    "metadata_or_noise",
    "narration",
}


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows, [f"missing file: {path}"]
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"{path}:{line_no}: expected object")
                continue
            rows.append(payload)
    return rows, errors


def collect_ids(rows: list[dict[str, Any]], path: Path) -> tuple[set[str], list[str]]:
    seen: set[str] = set()
    errors: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id", "") or "")
        if not case_id:
            errors.append(f"{path}: missing case_id")
            continue
        if case_id in seen:
            errors.append(f"{path}: duplicate case_id: {case_id}")
        seen.add(case_id)
    return seen, errors


def validate_benchmark(benchmark: Path) -> list[str]:
    errors: list[str] = []
    cases_path = benchmark / "cases.jsonl"
    refs_path = benchmark / "references.jsonl"
    fake_path = benchmark / "fake_outputs.jsonl"
    cases, case_errors = read_jsonl(cases_path)
    refs, ref_errors = read_jsonl(refs_path)
    errors.extend(case_errors + ref_errors)
    if errors:
        return errors

    case_ids, case_id_errors = collect_ids(cases, cases_path)
    ref_ids, ref_id_errors = collect_ids(refs, refs_path)
    errors.extend(case_id_errors + ref_id_errors)
    if case_ids != ref_ids:
        errors.append(f"case_id mismatch: cases vs references: {sorted(case_ids ^ ref_ids)}")

    for row in cases:
        cid = row.get("case_id", "<unknown>")
        missing = sorted(CASE_REQUIRED - set(row))
        if missing:
            errors.append(f"{cases_path}: {cid}: missing {missing}")
        if not isinstance(row.get("risk_labels"), list):
            errors.append(f"{cases_path}: {cid}: risk_labels must be a list")
        source_type = str(row.get("source_type", "") or "")
        if source_type not in VALID_SOURCE_TYPES:
            errors.append(f"{cases_path}: {cid}: unknown source_type {source_type!r}")
        if not str(row.get("source_text", "")).strip():
            errors.append(f"{cases_path}: {cid}: source_text must not be empty")

    for row in refs:
        cid = row.get("case_id", "<unknown>")
        missing = sorted(REFERENCE_REQUIRED - set(row))
        if missing:
            errors.append(f"{refs_path}: {cid}: missing {missing}")
        decision = str(row.get("expected_decision", "") or "")
        if decision not in VALID_DECISIONS:
            errors.append(f"{refs_path}: {cid}: expected_decision must be accept or reject")
        for key in ("forbidden_patterns", "acceptable_outputs", "must_preserve_terms"):
            if key in row and not isinstance(row.get(key), list):
                errors.append(f"{refs_path}: {cid}: {key} must be a list")
        for key in ("max_chars", "max_words"):
            try:
                value = int(row.get(key))
            except (TypeError, ValueError):
                errors.append(f"{refs_path}: {cid}: {key} must be an integer")
                continue
            if value < 0:
                errors.append(f"{refs_path}: {cid}: {key} must be >= 0")

    if fake_path.exists():
        fake_rows, fake_errors = read_jsonl(fake_path)
        errors.extend(fake_errors)
        fake_ids, fake_id_errors = collect_ids(fake_rows, fake_path)
        errors.extend(fake_id_errors)
        unknown = sorted(fake_ids - case_ids)
        if unknown:
            errors.append(f"{fake_path}: unknown fake output case_ids {unknown}")
        for row in fake_rows:
            if "raw_output" not in row:
                errors.append(f"{fake_path}: {row.get('case_id', '<unknown>')}: missing raw_output")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args(argv)
    errors = validate_benchmark(args.benchmark)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Fixture validation passed: {args.benchmark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
