from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


ALLOWED_ORIENTATIONS = {"vertical", "horizontal", "mixed", "unknown"}
ALLOWED_KINDS = {"dialogue", "narration", "sfx", "sign", "thought", "metadata", "credit", "page_number", "noise", "unknown"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_box(box: Any, *, width: int | None = None, height: int | None = None) -> str | None:
    if not isinstance(box, list) or len(box) != 4:
        return "box must be a four-item list"
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError):
        return "box coordinates must be numeric"
    if x2 <= x1 or y2 <= y1:
        return "box must have positive width and height"
    if width is not None and height is not None:
        margin = 8
        if x1 < -margin or y1 < -margin or x2 > width + margin or y2 > height + margin:
            return f"box is outside image bounds: {box} image={width}x{height}"
    return None


def validate_label(label_path: Path, benchmark: Path) -> list[str]:
    errors: list[str] = []
    try:
        label = load_json(label_path)
    except Exception as exc:
        return [f"{label_path}: invalid JSON: {exc}"]

    page_id = label_path.name.removesuffix(".labels.json")
    if label.get("page_id") != page_id:
        errors.append(f"{label_path}: page_id must match filename ({page_id})")
    for key in ("schema_version", "page_id", "image", "text_regions", "groups"):
        if key not in label:
            errors.append(f"{label_path}: missing required field {key}")
    image_rel = str(label.get("image", ""))
    image_path = benchmark / image_rel
    width = height = None
    if not image_path.exists():
        errors.append(f"{label_path}: image path does not exist: {image_rel}")
    else:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            errors.append(f"{label_path}: cannot read image: {exc}")

    regions = label.get("text_regions", [])
    if not isinstance(regions, list):
        errors.append(f"{label_path}: text_regions must be a list")
        regions = []
    region_ids: set[str] = set()
    region_by_id: dict[str, dict[str, Any]] = {}
    extractable_group_ids: set[str] = set()
    for index, region in enumerate(regions):
        prefix = f"{label_path}: text_regions[{index}]"
        if not isinstance(region, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        for key in ("region_id", "box", "source_text", "orientation", "kind", "should_extract"):
            if key not in region:
                errors.append(f"{prefix}: missing required field {key}")
        region_id = str(region.get("region_id", ""))
        if region_id in region_ids:
            errors.append(f"{prefix}: duplicate region_id {region_id}")
        region_ids.add(region_id)
        region_by_id[region_id] = region
        box_error = validate_box(region.get("box"), width=width, height=height) if width is not None else validate_box(region.get("box"))
        if box_error:
            errors.append(f"{prefix}: {box_error}")
        if region.get("orientation") not in ALLOWED_ORIENTATIONS:
            errors.append(f"{prefix}: invalid orientation {region.get('orientation')}")
        if region.get("kind") not in ALLOWED_KINDS:
            errors.append(f"{prefix}: invalid kind {region.get('kind')}")
        if bool(region.get("should_extract")) and (region.get("group_id") is None or region.get("page_order") is None):
            errors.append(f"{prefix}: extractable regions require group_id and page_order")
        if bool(region.get("should_extract")) and region.get("group_id") is not None:
            extractable_group_ids.add(str(region.get("group_id")))

    groups = label.get("groups", [])
    if not isinstance(groups, list):
        errors.append(f"{label_path}: groups must be a list")
        groups = []
    group_ids: set[str] = set()
    group_orders: dict[int, str] = {}
    for index, group in enumerate(groups):
        prefix = f"{label_path}: groups[{index}]"
        if not isinstance(group, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        group_id = str(group.get("group_id", ""))
        if not group_id:
            errors.append(f"{prefix}: missing group_id")
        if group_id in group_ids:
            errors.append(f"{prefix}: duplicate group_id {group_id}")
        group_ids.add(group_id)
        page_order = group.get("page_order")
        if page_order is not None:
            try:
                order_key = int(page_order)
            except (TypeError, ValueError):
                errors.append(f"{prefix}: page_order must be an integer")
            else:
                if order_key in group_orders:
                    errors.append(f"{prefix}: duplicate group page_order {order_key} also used by {group_orders[order_key]}")
                group_orders[order_key] = group_id
        combined_parts: list[str] = []
        for member in group.get("member_region_ids", []):
            member_id = str(member)
            if member_id not in region_ids:
                errors.append(f"{prefix}: unknown member_region_id {member}")
                continue
            member_region = region_by_id[member_id]
            if not bool(member_region.get("should_extract")):
                errors.append(f"{prefix}: non-extractable member_region_id {member}")
            if str(member_region.get("group_id")) != group_id:
                errors.append(f"{prefix}: member_region_id {member} has group_id {member_region.get('group_id')}")
            combined_parts.append(str(member_region.get("source_text") or ""))
        if group.get("combined_source_text") is not None:
            combined = "".join(combined_parts)
            if str(group.get("combined_source_text") or "") != combined:
                errors.append(f"{prefix}: combined_source_text must equal concatenated member source_text")
    if group_ids != extractable_group_ids:
        missing = sorted(extractable_group_ids - group_ids)
        extra = sorted(group_ids - extractable_group_ids)
        if missing:
            errors.append(f"{label_path}: missing group objects for extractable group_id(s): {missing}")
        if extra:
            errors.append(f"{label_path}: group objects without extractable regions: {extra}")

    ignore_path = benchmark / "ignore_regions" / f"{page_id}.ignore.json"
    if ignore_path.exists():
        try:
            ignore = load_json(ignore_path)
            if ignore.get("page_id") != page_id:
                errors.append(f"{ignore_path}: page_id must match {page_id}")
            for index, item in enumerate(ignore.get("ignore_regions", [])):
                box_error = validate_box(item.get("box"), width=width, height=height) if width is not None else validate_box(item.get("box"))
                if box_error:
                    errors.append(f"{ignore_path}: ignore_regions[{index}]: {box_error}")
        except Exception as exc:
            errors.append(f"{ignore_path}: invalid JSON: {exc}")

    config_path = benchmark / "configs" / f"{page_id}.config.json"
    if config_path.exists():
        try:
            config = load_json(config_path)
            if config.get("page_id") != page_id:
                errors.append(f"{config_path}: page_id must match {page_id}")
            if "schema_version" not in config:
                errors.append(f"{config_path}: missing schema_version")
        except Exception as exc:
            errors.append(f"{config_path}: invalid JSON: {exc}")
    return errors


def validate_benchmark(benchmark: Path) -> list[str]:
    errors: list[str] = []
    if not benchmark.exists():
        return [f"benchmark folder does not exist: {benchmark}"]
    for name in ("pages", "labels"):
        if not (benchmark / name).is_dir():
            errors.append(f"missing required folder: {benchmark / name}")
    if errors:
        return errors
    label_paths = sorted((benchmark / "labels").glob("*.labels.json"))
    if not label_paths:
        errors.append(f"no label files found in {benchmark / 'labels'}")
    for label_path in label_paths:
        errors.extend(validate_label(label_path, benchmark))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate source-extraction benchmark fixtures.")
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args(argv)
    errors = validate_benchmark(args.benchmark)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"OK: {args.benchmark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
