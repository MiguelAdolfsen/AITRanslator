from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def draw_overlay(image_path: Path, label: dict[str, Any], prediction: dict[str, Any], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for region in label.get("text_regions", []):
        box = [int(value) for value in region["box"]]
        color = "green" if region.get("should_extract") else "gray"
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], max(0, box[1] - 12)), str(region.get("region_id", "")), fill=color)
    for pred in prediction.get("regions", []):
        box = [int(value) for value in pred["box"]]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[3] + 2), str(pred.get("pred_region_id", "")), fill="red")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a simple source-extraction overlay.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    label = json.loads(args.label.read_text(encoding="utf-8-sig"))
    prediction = json.loads(args.prediction.read_text(encoding="utf-8-sig"))
    draw_overlay(args.image, label, prediction, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

