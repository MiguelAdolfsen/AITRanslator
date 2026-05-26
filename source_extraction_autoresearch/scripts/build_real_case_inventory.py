from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        gray = image.convert("L")
        thumb = gray.copy()
        thumb.thumbnail((320, 320))
        stat = ImageStat.Stat(thumb)
        edge = thumb.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edge)
        histogram = thumb.histogram()
        pixels = max(1, thumb.width * thumb.height)
        dark_pixels = sum(histogram[:96])
        return {
            "width": image.width,
            "height": image.height,
            "mean_luma": round(float(stat.mean[0]), 3),
            "dark_ink_ratio": round(dark_pixels / pixels, 6),
            "edge_density": round(float(edge_stat.mean[0]) / 255.0, 6),
        }


def candidate_notes(metrics: dict[str, Any], index: int, total: int) -> list[str]:
    notes: list[str] = []
    if index <= 2:
        notes.append("early_chapter_page")
    if index >= max(1, total - 1):
        notes.append("late_chapter_page")
    if metrics["edge_density"] >= 0.13:
        notes.append("dense_line_art")
    if metrics["dark_ink_ratio"] >= 0.30:
        notes.append("high_ink_or_dark_panels")
    if metrics["mean_luma"] <= 185:
        notes.append("low_luma")
    if metrics["height"] / max(1, metrics["width"]) < 1.25:
        notes.append("wide_or_nonstandard_aspect")
    return notes or ["standard_candidate"]


def scan_sources(source_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for folder in sorted([item for item in source_root.iterdir() if item.is_dir()], key=lambda item: item.name.lower()):
        images = sorted(
            [item for item in folder.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES],
            key=lambda item: item.name.lower(),
        )
        for index, image in enumerate(images, start=1):
            metrics = image_metrics(image)
            records.append(
                {
                    "source_path": str(image.as_posix()),
                    "source_folder": folder.name,
                    "source_index": index,
                    "source_folder_page_count": len(images),
                    "metrics": metrics,
                    "candidate_notes": candidate_notes(metrics, index, len(images)),
                    "ocr_region_count": None,
                }
            )
    return records


def select_diverse(records: list[dict[str, Any]], per_folder: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    folders = sorted({record["source_folder"] for record in records})
    for folder in folders:
        folder_records = [record for record in records if record["source_folder"] == folder]
        if len(folder_records) <= per_folder:
            selected.extend(folder_records)
            continue
        by_density = sorted(
            folder_records,
            key=lambda record: (
                float(record["metrics"]["edge_density"]) + float(record["metrics"]["dark_ink_ratio"]),
                record["source_index"],
            ),
            reverse=True,
        )
        anchors = [
            folder_records[0],
            folder_records[len(folder_records) // 3],
            folder_records[(2 * len(folder_records)) // 3],
            folder_records[-1],
        ]
        picked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in [*by_density, *anchors]:
            key = str(record["source_path"])
            if key in seen:
                continue
            picked.append(record)
            seen.add(key)
            if len(picked) >= per_folder:
                break
        selected.extend(sorted(picked, key=lambda record: (record["source_folder"], record["source_index"])))
    return selected


def empty_label(page_id: str, image_name: str, size: tuple[int, int], source_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "page_id": page_id,
        "image": f"pages/{image_name}",
        "image_size": [size[0], size[1]],
        "text_regions": [],
        "groups": [],
        "notes": (
            "DRAFT local real-page v2 candidate. Manual transcription and policy review are required "
            "before this page can be copied into local_real_diverse_v2_frozen."
        ),
        "source_record": {
            "source_path": source_record["source_path"],
            "source_folder": source_record["source_folder"],
            "candidate_notes": source_record["candidate_notes"],
        },
    }


def build_draft_case(selected: list[dict[str, Any]], output_root: Path, case_name: str, force: bool) -> Path:
    target = output_root / "cases" / case_name
    if target.exists() and force:
        shutil.rmtree(target)
    for name in ("pages", "labels", "ignore_regions", "configs"):
        (target / name).mkdir(parents=True, exist_ok=True)
    for page_number, record in enumerate(selected, start=1):
        source = Path(record["source_path"])
        page_id = f"page_{page_number:03d}"
        image_name = f"{page_id}{source.suffix.lower()}"
        destination = target / "pages" / image_name
        shutil.copy2(source, destination)
        with Image.open(destination) as image:
            size = image.size
        write_json(target / "labels" / f"{page_id}.labels.json", empty_label(page_id, image_name, size, record))
        write_json(
            target / "ignore_regions" / f"{page_id}.ignore.json",
            {"schema_version": 1, "page_id": page_id, "ignore_regions": []},
        )
        write_json(
            target / "configs" / f"{page_id}.config.json",
            {
                "schema_version": 1,
                "page_id": page_id,
                "detector": "ctd",
                "ocr_engine": "manga-ocr",
                "notes": "Draft v2 real-page candidate. Do not score until manually labeled and frozen.",
                "original_source_image": record["source_path"],
                "candidate_notes": record["candidate_notes"],
                "metrics": record["metrics"],
            },
        )
    write_json(target / "inventory.json", selected)
    (target / "README.md").write_text(
        f"# {case_name}\n\n"
        "Local real-page source extraction v2 draft set.\n\n"
        "Status: DRAFT. These pages were selected from `mangafolder` for manual labeling. "
        "The labels are intentionally empty scaffolds and must not drive autoresearch keep/revert decisions.\n\n"
        "Freeze criteria:\n"
        "- every extractable region has reviewed source text, box, kind, orientation, group_id, and page_order\n"
        "- non-extractable traps are labeled or ignored according to the core extraction policy\n"
        "- fixture validation passes\n"
        "- the reviewed pages are copied to `local_real_diverse_v2_frozen`\n\n"
        f"Pages: {len(selected)}\n",
        encoding="utf-8",
    )
    return target


def make_frozen_placeholder(output_root: Path, case_name: str) -> Path:
    target = output_root / "cases" / case_name
    for name in ("pages", "labels", "ignore_regions", "configs"):
        (target / name).mkdir(parents=True, exist_ok=True)
    (target / "README.md").write_text(
        f"# {case_name}\n\n"
        "Local real-page source extraction v2 frozen set.\n\n"
        "Status: NOT READY. Populate this folder only with manually reviewed pages from "
        "`local_real_diverse_v2_draft`. Do not run autoresearch decisions against this case "
        "until labels are complete and fixture validation passes.\n",
        encoding="utf-8",
    )
    return target


def make_montage(draft: Path) -> Path:
    pages = sorted((draft / "pages").glob("*"))
    thumbs: list[Image.Image] = []
    for image_path in pages:
        with Image.open(image_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((220, 320))
            canvas = Image.new("RGB", (240, 350), "white")
            canvas.paste(thumb, ((240 - thumb.width) // 2, 24))
            thumbs.append(canvas)
    if not thumbs:
        raise ValueError(f"No page images found in {draft / 'pages'}")
    columns = 6
    rows = (len(thumbs) + columns - 1) // columns
    montage = Image.new("RGB", (columns * 240, rows * 350), "white")
    for index, thumb in enumerate(thumbs):
        montage.paste(thumb, ((index % columns) * 240, (index // columns) * 350))
    output = draft / "page_montage.jpg"
    montage.save(output, quality=90)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory local manga pages and scaffold a v2 real-page draft case.")
    parser.add_argument("--source-root", type=Path, default=Path("mangafolder"))
    parser.add_argument("--output", type=Path, default=Path("source_extraction_autoresearch/benchmarks"))
    parser.add_argument("--draft-case-name", default="local_real_diverse_v2_draft")
    parser.add_argument("--frozen-case-name", default="local_real_diverse_v2_frozen")
    parser.add_argument("--per-folder", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    records = scan_sources(args.source_root)
    selected = select_diverse(records, args.per_folder)
    draft = build_draft_case(selected, args.output, args.draft_case_name, args.force)
    frozen = make_frozen_placeholder(args.output, args.frozen_case_name)
    montage = make_montage(draft)
    print(f"Inventoried {len(records)} source page(s)")
    print(f"Scaffolded {len(selected)} draft page(s): {draft}")
    print(f"Wrote draft montage: {montage}")
    print(f"Created frozen placeholder: {frozen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
