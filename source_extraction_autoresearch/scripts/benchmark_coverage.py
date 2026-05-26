from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from validate_fixtures import load_json, validate_benchmark


def counter_text(counter: Counter[str]) -> str:
    if not counter:
        return "(none)"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def is_extractable(region: dict[str, Any]) -> bool:
    return bool(region.get("should_extract"))


def label_paths(benchmark: Path) -> list[Path]:
    return sorted((benchmark / "labels").glob("*.labels.json"))


def summarize_benchmark(benchmark: Path) -> dict[str, Any]:
    pages = 0
    regions = 0
    extractable = 0
    non_extractable = 0
    ignore_regions = 0
    config_files = 0
    empty_ignore_files = 0
    empty_config_files = 0
    kind_counts: Counter[str] = Counter()
    extractable_kind_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    extractable_orientation_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    group_size_counts: Counter[str] = Counter()
    pages_with_non_extractable = 0
    pages_with_ignore = 0

    for label_path in label_paths(benchmark):
        label = load_json(label_path)
        pages += 1
        page_has_non_extractable = False
        for region in label.get("text_regions", []):
            regions += 1
            kind = str(region.get("kind", "unknown"))
            orientation = str(region.get("orientation", "unknown"))
            kind_counts[kind] += 1
            orientation_counts[orientation] += 1
            for tag in region.get("difficulty_tags", []) or []:
                difficulty_counts[str(tag)] += 1
            if is_extractable(region):
                extractable += 1
                extractable_kind_counts[kind] += 1
                extractable_orientation_counts[orientation] += 1
            else:
                non_extractable += 1
                page_has_non_extractable = True
        if page_has_non_extractable:
            pages_with_non_extractable += 1

        for group in label.get("groups", []):
            size = len(group.get("member_region_ids", []) or [])
            group_size_counts[str(size)] += 1

        page_id = str(label.get("page_id"))
        ignore_path = benchmark / "ignore_regions" / f"{page_id}.ignore.json"
        if ignore_path.exists():
            ignore = load_json(ignore_path)
            items = ignore.get("ignore_regions", []) or []
            ignore_regions += len(items)
            pages_with_ignore += 1 if items else 0
            empty_ignore_files += 1 if not items else 0
        config_path = benchmark / "configs" / f"{page_id}.config.json"
        if config_path.exists():
            config_files += 1
            config = load_json(config_path)
            empty_config_files += 1 if sorted(config.keys()) == ["page_id", "schema_version"] else 0

    return {
        "schema_version": 1,
        "benchmark": str(benchmark),
        "pages": pages,
        "regions": regions,
        "extractable_regions": extractable,
        "non_extractable_regions": non_extractable,
        "extractable_ratio": round(extractable / max(1, regions), 6),
        "pages_with_non_extractable": pages_with_non_extractable,
        "ignore_regions": ignore_regions,
        "pages_with_ignore": pages_with_ignore,
        "empty_ignore_files": empty_ignore_files,
        "config_files": config_files,
        "empty_config_files": empty_config_files,
        "kinds": dict(sorted(kind_counts.items())),
        "extractable_kinds": dict(sorted(extractable_kind_counts.items())),
        "orientations": dict(sorted(orientation_counts.items())),
        "extractable_orientations": dict(sorted(extractable_orientation_counts.items())),
        "difficulty_tags": dict(sorted(difficulty_counts.items())),
        "group_sizes": dict(sorted(group_size_counts.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 0)),
    }


def print_text(summary: dict[str, Any]) -> None:
    print(f"benchmark: {summary['benchmark']}")
    print(
        "pages={pages} regions={regions} extractable={extractable_regions} "
        "non_extractable={non_extractable_regions} extractable_ratio={extractable_ratio}".format(**summary)
    )
    print(f"pages_with_non_extractable={summary['pages_with_non_extractable']}")
    print(
        "ignore_regions={ignore_regions} pages_with_ignore={pages_with_ignore} "
        "empty_ignore_files={empty_ignore_files}".format(**summary)
    )
    print(f"config_files={summary['config_files']} empty_config_files={summary['empty_config_files']}")
    print(f"kinds: {counter_text(Counter(summary['kinds']))}")
    print(f"extractable_kinds: {counter_text(Counter(summary['extractable_kinds']))}")
    print(f"orientations: {counter_text(Counter(summary['orientations']))}")
    print(f"extractable_orientations: {counter_text(Counter(summary['extractable_orientations']))}")
    print(f"difficulty_tags: {counter_text(Counter(summary['difficulty_tags']))}")
    print(f"group_sizes: {counter_text(Counter(summary['group_sizes']))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report label coverage for a source-extraction benchmark.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_validation:
        errors = validate_benchmark(args.benchmark)
        if errors:
            for error in errors:
                print(error)
            return 1

    summary = summarize_benchmark(args.benchmark)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
