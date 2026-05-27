from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_quality_autoresearch.common.io_utils import load_jsonl
from translation_quality_autoresearch.common.io_utils import load_json
from translation_quality_autoresearch.common.schemas import TranslationCase


CASE_FILES = ["cases.jsonl", "adversarial_cases.jsonl", "page_context_cases.jsonl", "seed_cases.jsonl"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate translation-quality benchmark files.")
    parser.add_argument("--benchmark", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors = validate_benchmark(args.benchmark)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"Benchmark OK: {args.benchmark}")
    return 0


def validate_benchmark(benchmark: Path) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_version(benchmark / "VERSION.json"))
    known_cases: dict[str, TranslationCase] = {}
    for name in CASE_FILES:
        path = benchmark / name
        try:
            rows = load_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for payload in rows:
            try:
                case = TranslationCase.from_dict(payload)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
            if case.case_id in known_cases:
                errors.append(f"{path}: duplicate case_id {case.case_id}")
            known_cases[case.case_id] = case
            errors.extend(f"{path}: {error}" for error in case.validate())
    errors.extend(validate_references(benchmark / "references.jsonl", known_cases))
    errors.extend(validate_frozen_outputs(benchmark / "frozen_agent_outputs.jsonl", known_cases))
    errors.extend(validate_human_gold(benchmark / "human_gold.jsonl", known_cases))
    for name in ["glossary.json"]:
        if not (benchmark / name).exists():
            errors.append(f"{benchmark / name}: missing")
    return errors


def validate_version(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = load_json(path, default={})
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"{path}: VERSION.json must be an object"]
    for key in ["benchmark_id", "reference_policy", "case_count", "frozen_output_policy"]:
        if key not in payload:
            errors.append(f"{path}: missing {key}")
    try:
        case_count = int(payload.get("case_count"))
    except (TypeError, ValueError):
        errors.append(f"{path}: case_count must be an integer")
    else:
        if case_count < 0:
            errors.append(f"{path}: case_count must be non-negative")
    return errors


def validate_references(path: Path, known_cases: dict[str, TranslationCase]) -> list[str]:
    errors: list[str] = []
    for row in load_jsonl(path):
        case_id = str(row.get("case_id") or "")
        if case_id not in known_cases:
            errors.append(f"{path}: reference uses unknown case_id {case_id}")
        if not isinstance(row.get("reference_translations"), list) or not row.get("reference_translations"):
            errors.append(f"{path}: {case_id}: reference_translations must be a non-empty list")
    return errors


def validate_frozen_outputs(path: Path, known_cases: dict[str, TranslationCase]) -> list[str]:
    errors: list[str] = []
    for row in load_jsonl(path):
        case_id = str(row.get("case_id") or "")
        case = known_cases.get(case_id)
        if case is None:
            errors.append(f"{path}: frozen output uses unknown case_id {case_id}")
            continue
        if str(row.get("source_hash") or "") != case.source_hash:
            errors.append(f"{path}: {case_id}: source_hash mismatch")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"{path}: {case_id}: candidates must be a non-empty list")
            continue
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                errors.append(f"{path}: {case_id}: candidate must be an object")
                continue
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                errors.append(f"{path}: {case_id}: candidate missing candidate_id")
            if candidate_id in seen:
                errors.append(f"{path}: {case_id}: duplicate candidate_id {candidate_id}")
            seen.add(candidate_id)
            for key in ["agent", "text"]:
                if key not in candidate:
                    errors.append(f"{path}: {case_id}: {candidate_id}: missing {key}")
    return errors


def validate_human_gold(path: Path, known_cases: dict[str, TranslationCase]) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return errors
    for row in load_jsonl(path):
        case_id = str(row.get("case_id") or "")
        if case_id not in known_cases and not row.get("source_text"):
            errors.append(f"{path}: human_gold references unknown case_id without full source fields: {case_id}")
        if not isinstance(row.get("gold_rating"), dict):
            errors.append(f"{path}: {case_id}: gold_rating must be an object")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
