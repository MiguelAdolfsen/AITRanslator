from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def numeric_stem_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem.split(".")[0]), path.name)
    except ValueError:
        return (10**9, path.name)


def orientation_from_block(block: dict[str, Any]) -> str:
    metadata = block.get("metadata") or {}
    if metadata.get("vertical") is True:
        return "vertical"
    if metadata.get("vertical") is False:
        return "horizontal"
    box = block.get("box") or [0, 0, 0, 0]
    width = max(1, int(box[2]) - int(box[0]))
    height = max(1, int(box[3]) - int(box[1]))
    return "vertical" if height > width * 1.25 else "horizontal"


def source_kind_from_block(block: dict[str, Any]) -> str:
    text = str(block.get("source_text") or "")
    if len(text) <= 4 and any(ch in text for ch in "っッー〜～…・・・・！！!?！？"):
        return "sfx"
    return "dialogue"


def source_image_for_report(report: dict[str, Any], source_image_dir: Path | None) -> Path:
    source = report.get("source_image")
    if source:
        path = Path(source)
        if path.exists():
            return path
    if source_image_dir:
        for suffix in (".png", ".webp", ".jpg", ".jpeg"):
            candidate = source_image_dir / (Path(str(report.get("source_image") or "")).stem + suffix)
            if candidate.exists():
                return candidate
    output = report.get("output_image")
    if output and Path(output).exists():
        return Path(output)
    raise FileNotFoundError("Could not locate source image from OCR JSON. Pass --source-image-dir if needed.")


def region_from_member(
    *,
    page_number: int,
    region_number: int,
    group_id: str,
    page_order: int,
    member: dict[str, Any],
    fallback_orientation: str,
    fallback_kind: str,
    extra_tags: list[str],
) -> dict[str, Any]:
    metadata = member.get("metadata") or {}
    orientation = "vertical" if metadata.get("vertical") is True else "horizontal" if metadata.get("vertical") is False else fallback_orientation
    text = str(member.get("source_text") or "")
    return {
        "region_id": f"r{page_number:03d}_{region_number:03d}",
        "box": [int(v) for v in member.get("box", [0, 0, 0, 0])],
        "source_text": text,
        "normalized_source_text": text,
        "orientation": orientation,
        "group_id": group_id,
        "page_order": page_order,
        "kind": fallback_kind,
        "should_extract": True,
        "difficulty_tags": ["draft_from_ocr_json", "manual_review_required", orientation, *extra_tags],
    }


def label_from_ocr_report(report: dict[str, Any], page_number: int, image_name: str, image_size: tuple[int, int]) -> dict[str, Any]:
    regions: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    region_counter = 1
    kept_blocks = sorted(report.get("kept_blocks") or [], key=lambda block: int(block.get("page_order") or 9999))
    for group_number, block in enumerate(kept_blocks, start=1):
        group_id = f"g{page_number:03d}_{group_number:03d}"
        page_order = int(block.get("page_order") or group_number)
        kind = source_kind_from_block(block)
        orientation = orientation_from_block(block)
        metadata = block.get("metadata") or {}
        members = metadata.get("grouped_from") if isinstance(metadata.get("grouped_from"), list) else None
        member_region_ids: list[str] = []
        if members:
            for member in members:
                region = region_from_member(
                    page_number=page_number,
                    region_number=region_counter,
                    group_id=group_id,
                    page_order=page_order,
                    member=member,
                    fallback_orientation=orientation,
                    fallback_kind=kind,
                    extra_tags=["split_group", "grouped_from_debug"],
                )
                regions.append(region)
                member_region_ids.append(region["region_id"])
                region_counter += 1
        else:
            region = region_from_member(
                page_number=page_number,
                region_number=region_counter,
                group_id=group_id,
                page_order=page_order,
                member=block,
                fallback_orientation=orientation,
                fallback_kind=kind,
                extra_tags=[],
            )
            regions.append(region)
            member_region_ids.append(region["region_id"])
            region_counter += 1
        combined = str(block.get("source_text") or "")
        groups.append(
            {
                "group_id": group_id,
                "member_region_ids": member_region_ids,
                "combined_source_text": combined,
                "page_order": page_order,
                "kind": kind,
            }
        )
    return {
        "schema_version": 1,
        "page_id": f"page_{page_number:03d}",
        "image": f"pages/{image_name}",
        "image_size": list(image_size),
        "text_regions": regions,
        "groups": groups,
        "notes": (
            "DRAFT labels scaffolded from OCR debug output. They are useful for manual review "
            "but must be corrected by a human before this case is treated as frozen ground truth."
        ),
    }


def scaffold_from_ocr_json(args: argparse.Namespace) -> int:
    target = args.output / "cases" / args.case_name
    if target.exists() and args.force:
        shutil.rmtree(target)
    for name in ("pages", "labels", "ignore_regions", "configs"):
        (target / name).mkdir(parents=True, exist_ok=True)

    reports = sorted(args.ocr_json_dir.glob("*.ocr.json"), key=numeric_stem_key)
    if args.limit:
        reports = reports[: args.limit]
    if not reports:
        raise SystemExit(f"No .ocr.json files found in {args.ocr_json_dir}")

    for page_number, report_path in enumerate(reports, start=1):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source_image = source_image_for_report(report, args.source_image_dir)
        image_name = f"page_{page_number:03d}{source_image.suffix.lower()}"
        destination = target / "pages" / image_name
        shutil.copy2(source_image, destination)
        with Image.open(destination) as image:
            image_size = image.size
        label = label_from_ocr_report(report, page_number, image_name, image_size)
        write_json(target / "labels" / f"page_{page_number:03d}.labels.json", label)
        write_json(
            target / "ignore_regions" / f"page_{page_number:03d}.ignore.json",
            {"schema_version": 1, "page_id": f"page_{page_number:03d}", "ignore_regions": []},
        )
        write_json(
            target / "configs" / f"page_{page_number:03d}.config.json",
            {
                "schema_version": 1,
                "page_id": f"page_{page_number:03d}",
                "detector": "ctd",
                "ocr_engine": "manga-ocr",
                "notes": "Draft real-page case. Use project defaults unless explicitly testing detector variants.",
            },
        )

    (target / "README.md").write_text(
        f"# {args.case_name}\n\n"
        "Local real-page source extraction case set.\n\n"
        "Status: DRAFT, manual review required before using as frozen ground truth.\n\n"
        "These pages were copied from local input images and labels were scaffolded from existing `.ocr.json` debug output. "
        "The labels are intentionally visible and editable for manual correction, but optimization agents must not tune "
        "against this case until the labels have been reviewed and frozen.\n\n"
        f"Pages: {len(reports)}\n",
        encoding="utf-8",
    )
    print(f"Scaffolded {len(reports)} draft real page(s) in {target}")
    return 0


def scaffold_empty_from_debug_png(args: argparse.Namespace) -> int:
    if not args.debug_dir.exists():
        raise SystemExit(f"debug dir does not exist: {args.debug_dir}")
    target = args.output / "cases" / args.case_name
    for name in ("pages", "labels", "ignore_regions", "configs"):
        (target / name).mkdir(parents=True, exist_ok=True)
    copied = 0
    for image in sorted(args.debug_dir.rglob("*.png")):
        page_id = f"page_{copied + 1:03d}"
        shutil.copy2(image, target / "pages" / f"{page_id}.png")
        label = {
            "schema_version": 1,
            "page_id": page_id,
            "image": f"pages/{page_id}.png",
            "text_regions": [],
            "groups": [],
            "notes": f"Draft label scaffolded from {image}. Human review required before use as benchmark ground truth.",
        }
        write_json(target / "labels" / f"{page_id}.labels.json", label)
        copied += 1
    (target / "README.md").write_text(
        f"# {args.case_name}\n\nDraft benchmark scaffold. Human review required before scoring.\n",
        encoding="utf-8",
    )
    print(f"Scaffolded {copied} draft page(s) in {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scaffold a draft real benchmark case from compact debug artifacts.")
    parser.add_argument("--debug-dir", type=Path, help="Folder of debug PNGs. Creates empty draft labels.")
    parser.add_argument("--ocr-json-dir", type=Path, help="Folder of .ocr.json files. Creates draft labels from kept blocks.")
    parser.add_argument("--source-image-dir", type=Path, help="Optional source image folder if .ocr.json source paths are unavailable.")
    parser.add_argument("--output", type=Path, required=True, help="Benchmark root, usually source_extraction_autoresearch/benchmarks.")
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.ocr_json_dir:
        return scaffold_from_ocr_json(args)
    if args.debug_dir:
        return scaffold_empty_from_debug_png(args)
    raise SystemExit("Pass either --ocr-json-dir or --debug-dir.")


if __name__ == "__main__":
    raise SystemExit(main())
