from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


TEXT_SAMPLES = [
    "おはよう",
    "何してるの",
    "まって",
    "大丈夫",
    "行こう",
    "それは違う",
    "ありがとう",
    "ごめんね",
    "見つけた",
    "本当だ",
    "しずかに",
    "ドキ",
    "バン",
    "第3話",
    "あとでね",
]


def find_font(size: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, str | None]:
    candidates = [
        Path("C:/Windows/Fonts/meiryo.ttc"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size), str(path)
    return ImageFont.load_default(), None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def vertical_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str = "black") -> None:
    x, y = xy
    step = max(18, int(getattr(font, "size", 24) * 1.05))
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        y += step


def add_region(
    draw: ImageDraw.ImageDraw,
    *,
    page_index: int,
    region_index: int,
    text: str,
    box: list[int],
    orientation: str,
    kind: str,
    should_extract: bool,
    group_id: str | None,
    page_order: int | None,
    font: ImageFont.ImageFont,
    tags: list[str],
    text_fill: str = "black",
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    if should_extract and kind in {"dialogue", "thought"}:
        draw.ellipse((x1 - 8, y1 - 8, x2 + 8, y2 + 8), fill="white", outline="black", width=2)
    elif should_extract and kind == "narration":
        draw.rectangle((x1 - 8, y1 - 6, x2 + 8, y2 + 6), fill="#f7f2d0", outline="black", width=2)
    elif should_extract and kind == "sign":
        draw.rectangle((x1 - 6, y1 - 6, x2 + 6, y2 + 6), fill="#eeeeee", outline="#666666", width=1)

    if orientation == "vertical":
        vertical_text(draw, (x1 + 8, y1 + 6), text, font, fill=text_fill)
    else:
        draw.text((x1 + 6, y1 + 6), text, font=font, fill=text_fill)

    region_id = f"r{page_index:03d}_{region_index:02d}"
    return {
        "region_id": region_id,
        "box": box,
        "source_text": text,
        "normalized_source_text": text,
        "orientation": orientation,
        "group_id": group_id,
        "page_order": page_order,
        "kind": kind,
        "should_extract": should_extract,
        "difficulty_tags": tags,
    }


def make_page(page_index: int, output: Path, font: ImageFont.ImageFont) -> None:
    rng = random.Random(1000 + page_index)
    width, height = 900, 1300
    image = Image.new("RGB", (width, height), "#f8f8f3")
    draw = ImageDraw.Draw(image)
    for _ in range(80):
        x = rng.randrange(0, width)
        y = rng.randrange(0, height)
        shade = rng.randrange(218, 246)
        draw.point((x, y), fill=(shade, shade, shade))
    draw.rectangle((0, 0, width - 1, height - 1), outline="#c8c8c8")

    regions: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    order_positions = [
        (680, 90, 760, 310, "vertical", "dialogue", ["vertical", "speech_bubble"]),
        (520, 160, 610, 390, "vertical", "dialogue", ["multi_column", "speech_bubble"]),
        (330, 120, 470, 190, "horizontal", "narration", ["horizontal", "narration_box"]),
        (620, 560, 735, 650, "horizontal", "sign", ["sign"]),
        (160, 640, 240, 875, "vertical", "thought", ["nearby_bubbles"]),
    ]
    for idx, (x1, y1, x2, y2, orientation, kind, tags) in enumerate(order_positions, start=1):
        text = TEXT_SAMPLES[(page_index + idx) % len(TEXT_SAMPLES)]
        if page_index % 4 == 0 and idx == 2:
            text = "しず\nかに".replace("\n", "")
            tags = tags + ["split_group"]
        if page_index % 5 == 0 and idx == 5:
            text = "……"
            kind = "sfx"
            tags = tags + ["punctuation_only"]
        group_id = f"g{page_index:03d}_{idx:02d}"
        region = add_region(
            draw,
            page_index=page_index,
            region_index=idx,
            text=text,
            box=[x1, y1, x2, y2],
            orientation=orientation,
            kind=kind,
            should_extract=True,
            group_id=group_id,
            page_order=idx,
            font=font,
            tags=tags,
        )
        regions.append(region)
        groups.append(
            {
                "group_id": group_id,
                "member_region_ids": [region["region_id"]],
                "combined_source_text": text,
                "page_order": idx,
                "kind": kind,
            }
        )

    # One page includes a group split across two small regions.
    if page_index == 6:
        first = add_region(
            draw,
            page_index=page_index,
            region_index=6,
            text="大",
            box=[720, 900, 765, 965],
            orientation="vertical",
            kind="sfx",
            should_extract=True,
            group_id=f"g{page_index:03d}_06",
            page_order=6,
            font=font,
            tags=["small_text", "split_group"],
        )
        second = add_region(
            draw,
            page_index=page_index,
            region_index=7,
            text="丈",
            box=[665, 902, 710, 965],
            orientation="vertical",
            kind="sfx",
            should_extract=True,
            group_id=f"g{page_index:03d}_06",
            page_order=6,
            font=font,
            tags=["small_text", "split_group"],
        )
        regions.extend([first, second])
        groups.append(
            {
                "group_id": f"g{page_index:03d}_06",
                "member_region_ids": [first["region_id"], second["region_id"]],
                "combined_source_text": "大丈",
                "page_order": 6,
                "kind": "sfx",
            }
        )

    def append_single(
        region_index: int,
        page_order: int,
        *,
        text: str,
        box: list[int],
        orientation: str,
        kind: str,
        tags: list[str],
        should_extract: bool = True,
        text_fill: str = "black",
        draw_font: ImageFont.ImageFont | None = None,
    ) -> None:
        group_id = f"g{page_index:03d}_{region_index:02d}" if should_extract else None
        region = add_region(
            draw,
            page_index=page_index,
            region_index=region_index,
            text=text,
            box=box,
            orientation=orientation,
            kind=kind,
            should_extract=should_extract,
            group_id=group_id,
            page_order=page_order if should_extract else None,
            font=draw_font or font,
            tags=tags,
            text_fill=text_fill,
        )
        regions.append(region)
        if should_extract:
            groups.append(
                {
                    "group_id": group_id,
                    "member_region_ids": [region["region_id"]],
                    "combined_source_text": text,
                    "page_order": page_order,
                    "kind": kind,
                }
            )

    if page_index == 2:
        append_single(20, 6, text="\u305d\u3063\u3061\u306f\uff1f", box=[735, 700, 805, 900], orientation="vertical", kind="dialogue", tags=["close_separate_bubbles", "vertical", "speech_bubble"])
        append_single(21, 7, text="\u3053\u3063\u3061\u3060\u3088", box=[645, 710, 715, 900], orientation="vertical", kind="dialogue", tags=["close_separate_bubbles", "vertical", "speech_bubble"])

    if page_index == 3:
        group_id = f"g{page_index:03d}_20"
        first = add_region(draw, page_index=page_index, region_index=20, text="\u305d\u306e\u6642", box=[110, 980, 235, 1030], orientation="horizontal", kind="narration", should_extract=True, group_id=group_id, page_order=6, font=font, tags=["split_group", "horizontal", "narration_box"])
        second = add_region(draw, page_index=page_index, region_index=21, text="\u6c17\u3065\u3044\u305f", box=[245, 980, 405, 1030], orientation="horizontal", kind="narration", should_extract=True, group_id=group_id, page_order=6, font=font, tags=["split_group", "horizontal", "narration_box"])
        regions.extend([first, second])
        groups.append({"group_id": group_id, "member_region_ids": [first["region_id"], second["region_id"]], "combined_source_text": "\u305d\u306e\u6642\u6c17\u3065\u3044\u305f", "page_order": 6, "kind": "narration"})

    if page_index == 4:
        append_single(20, 6, text="\u898b\u3048\u306a\u3044", box=[90, 980, 190, 1070], orientation="horizontal", kind="dialogue", tags=["low_contrast", "speech_bubble"], text_fill="#9a9a9a")
        append_single(93, 0, text="RAW 004", box=[365, 760, 510, 800], orientation="horizontal", kind="noise", should_extract=False, tags=["watermark_noise", "latin_noise"], text_fill="#777777")

    if page_index == 5:
        append_single(20, 6, text="\u5c0f\u3055\u3044\u58f0", box=[780, 1120, 850, 1260], orientation="vertical", kind="dialogue", tags=["page_edge", "vertical", "speech_bubble"])

    if page_index == 7:
        append_single(20, 6, text="\u30c9\u30aa\u30f3", box=[300, 850, 520, 930], orientation="horizontal", kind="sfx", tags=["large_sfx", "sound_effect"])
        append_single(21, 7, text="\u30b6\u30ef\u30b6\u30ef", box=[520, 875, 720, 930], orientation="horizontal", kind="sfx", tags=["large_sfx", "nearby_sfx"])

    if page_index == 8:
        append_single(20, 6, text="\u9375A-12", box=[90, 1010, 230, 1060], orientation="horizontal", kind="sign", tags=["sign", "mixed_japanese_latin", "code_text"])
        append_single(21, 7, text="\u7acb\u5165\u7981\u6b62", box=[255, 1000, 425, 1065], orientation="horizontal", kind="sign", tags=["sign", "kanji_sign"])

    if page_index == 9:
        small_font, _ = find_font(14)
        append_single(20, 6, text="\u3064\u3070\u3081", box=[730, 720, 780, 835], orientation="vertical", kind="dialogue", tags=["small_text", "furigana_like", "vertical"], draw_font=small_font)
        append_single(21, 7, text="\u5148\u8f29", box=[660, 720, 720, 835], orientation="vertical", kind="dialogue", tags=["small_text", "honorific", "vertical"], draw_font=small_font)

    if page_index == 10:
        append_single(20, 6, text="\u53f3\u306e\u5439\u304d\u51fa\u3057", box=[705, 850, 815, 1045], orientation="vertical", kind="dialogue", tags=["overlap_risk", "speech_bubble"])
        append_single(21, 7, text="\u5de6\u306e\u5439\u304d\u51fa\u3057", box=[585, 865, 695, 1060], orientation="vertical", kind="dialogue", tags=["overlap_risk", "speech_bubble"])

    if page_index == 11:
        append_single(20, 6, text="\u6700\u5f8c\u306b\u8aad\u3080", box=[70, 115, 160, 260], orientation="vertical", kind="thought", tags=["reading_order_stress", "left_side"])
        append_single(21, 7, text="\u5148\u306b\u3053\u3063\u3061", box=[725, 1000, 825, 1160], orientation="vertical", kind="dialogue", tags=["reading_order_stress", "lower_right"])

    if page_index == 12:
        for extra_index, (x, y, text) in enumerate([(790, 705, "\u3060\u3081"), (695, 735, "\u307e\u3060"), (600, 765, "\u3053\u3053"), (505, 795, "\u884c\u3053\u3046")], start=20):
            append_single(extra_index, extra_index - 14, text=text, box=[x, y, x + 70, y + 120], orientation="vertical", kind="dialogue", tags=["dense_page", "vertical", "speech_bubble"])

    metadata = add_region(
        draw,
        page_index=page_index,
        region_index=90,
        text=f"第{page_index}話",
        box=[55, 40, 150, 80],
        orientation="horizontal",
        kind="metadata",
        should_extract=False,
        group_id=None,
        page_order=None,
        font=font,
        tags=["chapter_label"],
    )
    credit = add_region(
        draw,
        page_index=page_index,
        region_index=91,
        text="SCAN TEAM",
        box=[620, 1210, 820, 1250],
        orientation="horizontal",
        kind="credit",
        should_extract=False,
        group_id=None,
        page_order=None,
        font=font,
        tags=["credit", "latin_noise"],
    )
    page_number = add_region(
        draw,
        page_index=page_index,
        region_index=92,
        text=str(page_index),
        box=[430, 1240, 470, 1275],
        orientation="horizontal",
        kind="page_number",
        should_extract=False,
        group_id=None,
        page_order=None,
        font=font,
        tags=["page_number"],
    )
    regions.extend([metadata, credit, page_number])

    page_id = f"page_{page_index:03d}"
    image_path = output / "pages" / f"{page_id}.png"
    image.save(image_path)
    label = {
        "schema_version": 1,
        "page_id": page_id,
        "image": f"pages/{page_id}.png",
        "image_size": [width, height],
        "text_regions": regions,
        "groups": groups,
        "notes": "Deterministic synthetic source-extraction case.",
    }
    write_json(output / "labels" / f"{page_id}.labels.json", label)
    write_json(
        output / "ignore_regions" / f"{page_id}.ignore.json",
        {
            "schema_version": 1,
            "page_id": page_id,
            "ignore_regions": [
                {"box": [0, 0, width, 28], "reason": "scan margin"},
                {"box": [0, height - 24, width, height], "reason": "scan footer margin"},
            ],
        },
    )
    write_json(
        output / "configs" / f"{page_id}.config.json",
        {
            "schema_version": 1,
            "page_id": page_id,
            "detector": "ctd",
            "ocr_engine": "manga-ocr",
            "allow_tesseract_fallback": True,
            "notes": "Synthetic default.",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic source-extraction benchmark pages.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output = args.output
    if output.exists() and args.force:
        shutil.rmtree(output)
    for name in ("pages", "labels", "ignore_regions", "configs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    (output / ".gitkeep").write_text("", encoding="utf-8")

    font, font_path = find_font(24)
    for page_index in range(1, 13):
        make_page(page_index, output, font)

    warning = "" if font_path else "\nWarning: no Japanese-capable font was found; labels remain valid but rendered glyph coverage may be limited.\n"
    (output / "README.md").write_text(
        "# Synthetic Source Extraction Benchmark\n\n"
        "Deterministic generated pages for harness smoke testing and regression checks.\n"
        "These fixtures are not a substitute for real hand-labeled manga pages.\n"
        "\n"
        "Coverage includes vertical and horizontal dialogue, narration boxes, signs, "
        "thought bubbles, split text groups, close but separate bubbles, low-contrast text, "
        "page-edge regions, large and nearby SFX, mixed Japanese/Latin signs, small "
        "furigana-like text, honorific fragments, overlapping bubble risk, reading-order "
        "stress, dense pages, and ignored metadata/credit/page-number/noise regions.\n"
        "\n"
        "The benchmark is intentionally deterministic. Add new generic manga/source "
        "extraction failure patterns here before tuning the extraction code, but avoid "
        "case-specific rules that only match these generated labels.\n"
        "\n"
        f"Font: {font_path or 'PIL default'}\n"
        f"{warning}",
        encoding="utf-8",
    )
    print(f"Wrote synthetic benchmark to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
