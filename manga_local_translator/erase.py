from __future__ import annotations

import logging

import numpy as np

from .detect_ocr import TextBlock

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CAT-only environments
    cv2 = None

logger = logging.getLogger(__name__)


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for erasing text. Install opencv-python.")


def erase_text(
    image_bgr: np.ndarray,
    blocks: list[TextBlock],
    *,
    mode: str,
    padding: int,
) -> np.ndarray:
    _require_cv2()
    if not blocks:
        logger.info("No text blocks to erase")
        return image_bgr.copy()

    logger.info("Erasing %d text block(s) with mode=%s padding=%d", len(blocks), mode, padding)
    if mode == "white":
        cleaned = image_bgr.copy()
        height, width = cleaned.shape[:2]
        for block in blocks:
            x1, y1, x2, y2 = pad_box(block.box, width, height, padding)
            logger.debug("White ink erase: original=%s padded=%s", block.box, (x1, y1, x2, y2))
            if uses_dark_background(cleaned, block.box):
                logger.debug("Using rectangular white erase for dark-background text: box=%s", block.box)
                cv2.rectangle(cleaned, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
            else:
                erase_dark_ink(cleaned, block.box, (x1, y1, x2, y2))
        return cleaned

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    height, width = mask.shape[:2]
    for block in blocks:
        x1, y1, x2, y2 = pad_box(block.box, width, height, padding)
        logger.debug("Inpaint mask box: original=%s padded=%s", block.box, (x1, y1, x2, y2))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    logger.debug("Running OpenCV Telea inpainting")
    return cv2.inpaint(image_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def erase_dark_ink(
    image_bgr: np.ndarray,
    source_box: tuple[int, int, int, int],
    erase_box: tuple[int, int, int, int],
) -> None:
    _require_cv2()
    x1, y1, x2, y2 = erase_box
    if x2 <= x1 or y2 <= y1:
        return

    roi = image_bgr[y1:y2 + 1, x1:x2 + 1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark = (gray < 190).astype(np.uint8)
    if dark.max() == 0:
        return

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros_like(dark)
    local_source = (
        max(0, source_box[0] - x1),
        max(0, source_box[1] - y1),
        min(dark.shape[1] - 1, source_box[2] - x1),
        min(dark.shape[0] - 1, source_box[3] - y1),
    )
    source_area = max(1, (local_source[2] - local_source[0]) * (local_source[3] - local_source[1]))

    for index in range(1, component_count):
        cx, cy, component_width, component_height, area = stats[index]
        component_box = (int(cx), int(cy), int(cx + component_width), int(cy + component_height))
        if not should_erase_component(component_box, int(area), local_source, source_area, dark.shape[1], dark.shape[0]):
            continue
        keep[labels == index] = 255

    if keep.max() == 0:
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    keep = cv2.dilate(keep, kernel, iterations=1)
    roi[keep > 0] = (255, 255, 255)


def uses_dark_background(image_bgr: np.ndarray, source_box: tuple[int, int, int, int]) -> bool:
    _require_cv2()
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = pad_box(source_box, width, height, 1)
    if x2 <= x1 or y2 <= y1:
        return False
    gray = cv2.cvtColor(image_bgr[y1:y2 + 1, x1:x2 + 1], cv2.COLOR_BGR2GRAY)
    dark_ratio = float((gray < 95).mean())
    bright_ratio = float((gray > 210).mean())
    median = float(np.median(gray))
    return median < 150 and dark_ratio > 0.25 and bright_ratio > 0.08


def should_erase_component(
    component_box: tuple[int, int, int, int],
    area: int,
    source_box: tuple[int, int, int, int],
    source_area: int,
    roi_width: int,
    roi_height: int,
) -> bool:
    overlap = intersection_area(component_box, source_box)
    if overlap <= 0:
        return False

    x1, y1, x2, y2 = component_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    component_area = max(1, width * height)
    overlap_ratio = overlap / component_area

    if area > source_area * 0.75 and overlap_ratio < 0.65:
        return False
    if width > max(28, (source_box[2] - source_box[0]) * 1.6) and overlap_ratio < 0.75:
        return False
    if height > max(28, (source_box[3] - source_box[1]) * 1.6) and overlap_ratio < 0.75:
        return False
    if touches_roi_edge(component_box, roi_width, roi_height) and overlap_ratio < 0.85:
        return False
    return True


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def touches_roi_edge(box: tuple[int, int, int, int], roi_width: int, roi_height: int) -> bool:
    return box[0] <= 0 or box[1] <= 0 or box[2] >= roi_width - 1 or box[3] >= roi_height - 1


def pad_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_width - 1, x2 + padding),
        min(image_height - 1, y2 + padding),
    )
