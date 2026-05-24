from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(path: Path, *, recursive: bool) -> list[Path]:
    if path.is_file():
        supported = path.suffix.lower() in IMAGE_EXTENSIONS
        logger.debug("Single input file support check: path=%s supported=%s", path, supported)
        return [path] if supported else []

    iterator = path.rglob("*") if recursive else path.glob("*")
    images = sorted(
        (
            candidate
            for candidate in iterator
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_path_sort_key,
    )
    logger.debug("Image discovery complete: root=%s recursive=%s count=%d", path, recursive, len(images))
    return images


def natural_path_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.as_posix().lower())
    return [int(part) if part.isdigit() else part for part in parts]


def resolve_output_path(input_root: Path, output_root: Path, image_path: Path) -> Path:
    if input_root.is_file():
        if output_root.suffix:
            return output_root
        return output_root / input_root.name

    return output_root / image_path.relative_to(input_root)


def write_image_with_fallback(image_bgr, output_path: Path) -> Path:
    import cv2

    logger.info("Writing translated image: %s", output_path)
    if cv2.imwrite(str(output_path), image_bgr):
        return output_path

    fallback_path = output_path.with_name(
        f"{output_path.stem}.translated-{datetime.now().strftime('%Y%m%d-%H%M%S')}{output_path.suffix}"
    )
    logger.warning("Could not write %s; trying fallback path %s", output_path, fallback_path)
    if cv2.imwrite(str(fallback_path), image_bgr):
        return fallback_path

    logger.error("OpenCV could not write image to target or fallback: %s", output_path)
    raise RuntimeError(f"Could not write image: {output_path}")
