from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .detect_types import TextBlock
from .grouping import page_order_for_block
from .line_identity import block_id_for_block, line_id_for_block, lookup_translation, source_hash, translation_key_for_block
from .vision_types import VisionArtifact, VisionNumberMapEntry

logger = logging.getLogger(__name__)


def create_vision_artifact(
    image_bgr: np.ndarray,
    blocks: list[TextBlock],
    translations: dict[str, str],
    page_order_report: list[dict[str, object]],
    output_path: Path,
    *,
    mode: str,
    artifact_label: str = "vision",
) -> VisionArtifact:
    number_map = build_number_map(blocks, translations, page_order_report, output_path)
    artifact_path = output_path.with_name(f"{output_path.stem}.{artifact_label}.png")
    warnings: list[str] = []
    if mode == "page_image":
        artifact = image_bgr.copy()
    else:
        artifact = render_numbered_page(image_bgr, number_map, warnings)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(artifact_path), artifact):
        raise RuntimeError(f"Could not write vision artifact: {artifact_path}")
    logger.info("Vision artifact written: %s", artifact_path)
    return VisionArtifact(
        mode=mode,
        image_path=artifact_path,
        number_map=tuple(number_map),
        preprocessing_warnings=tuple(warnings),
    )


def build_number_map(
    blocks: list[TextBlock],
    translations: dict[str, str],
    page_order_report: list[dict[str, object]],
    output_path: Path,
) -> list[VisionNumberMapEntry]:
    entries: list[VisionNumberMapEntry] = []
    for fallback_index, block in enumerate(blocks, start=1):
        order_index = page_order_for_block(block, page_order_report) or fallback_index
        entries.append(
            VisionNumberMapEntry(
                number=order_index,
                line_id=line_id_for_block(block) or f"{output_path.stem}_{order_index:03d}",
                order_index=order_index,
                source_text=block.text,
                translated_text=lookup_translation(translations, block, ""),
                box=tuple(int(value) for value in block.box),
                state_key=translation_key_for_block(block),
                block_id=block_id_for_block(block),
                source_hash=source_hash(block.text),
            )
        )
    return sorted(entries, key=lambda item: item.order_index)


def render_numbered_page(
    image_bgr: np.ndarray,
    number_map: list[VisionNumberMapEntry],
    warnings: list[str],
) -> np.ndarray:
    artifact = image_bgr.copy()
    height, width = artifact.shape[:2]
    label_boxes: list[tuple[int, int, int, int]] = []
    for entry in number_map:
        x1, y1, x2, y2 = clamp_box(entry.box, width, height)
        if x2 <= x1 or y2 <= y1:
            warnings.append(f"invalid_box:{entry.line_id}")
            continue
        roi = artifact[y1:y2, x1:x2]
        if roi.size:
            fill = dominant_light_fill(roi)
            cv2.rectangle(artifact, (x1, y1), (x2, y2), fill, thickness=-1)
        label_box, leader = label_plan(
            entry.number,
            (x1, y1, x2, y2),
            width,
            height,
            label_boxes,
            warnings,
            line_id=entry.line_id,
        )
        if leader is not None:
            cv2.line(artifact, leader[0], leader[1], (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        draw_number_label(artifact, str(entry.number), label_box)
        label_boxes.append(label_box)
    return artifact


def dominant_light_fill(roi) -> tuple[int, int, int]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = gray >= 180
    if mask.any():
        pixels = roi[mask]
        median = np.median(pixels, axis=0)
        return tuple(int(value) for value in median)
    return (245, 245, 245)


def label_position(
    number: int,
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    label_box, _leader = label_plan(number, box, image_width, image_height, [], [], line_id="")
    return label_box


def label_plan(
    number: int,
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    existing_labels: list[tuple[int, int, int, int]],
    warnings: list[str],
    *,
    line_id: str,
) -> tuple[tuple[int, int, int, int], tuple[tuple[int, int], tuple[int, int]] | None]:
    _ = number
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    size = max(26, min(56, int(max(width, height) * 0.7)))
    if width < 34 or height < 34:
        warnings.append(f"tiny_label_region:{line_id}")
        size = max(size, 30)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    centered = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
    if width >= 34 and height >= 34:
        return warn_if_clamped(centered, image_width, image_height, warnings, line_id=line_id), None

    gap = 5
    candidates = [
        (x2 + gap, cy - size // 2, x2 + gap + size, cy + size // 2),
        (x1 - gap - size, cy - size // 2, x1 - gap, cy + size // 2),
        (cx - size // 2, y1 - gap - size, cx + size // 2, y1 - gap),
        (cx - size // 2, y2 + gap, cx + size // 2, y2 + gap + size),
        centered,
    ]
    for candidate in candidates:
        clamped = clamp_box(candidate, image_width, image_height)
        if clamped != candidate:
            continue
        if any(boxes_overlap(clamped, existing) for existing in existing_labels):
            continue
        return clamped, leader_line(box, clamped)

    clamped = warn_if_clamped(candidates[0], image_width, image_height, warnings, line_id=line_id)
    if any(boxes_overlap(clamped, existing) for existing in existing_labels):
        warnings.append(f"overlapping_label:{line_id}")
    return clamped, leader_line(box, clamped)


def draw_number_label(image_bgr, label: str, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(1.6, (y2 - y1) / 34))
    thickness = max(2, int(round(scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    tx = x1 + max(0, (x2 - x1 - text_width) // 2)
    ty = y1 + max(text_height, (y2 - y1 + text_height) // 2 - baseline // 2)
    cv2.putText(image_bgr, label, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def warn_if_clamped(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    warnings: list[str],
    *,
    line_id: str,
) -> tuple[int, int, int, int]:
    clamped = clamp_box(box, image_width, image_height)
    if clamped != box:
        warnings.append(f"label_clamped:{line_id}")
    return clamped


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def leader_line(
    source_box: tuple[int, int, int, int],
    label_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    sx = (source_box[0] + source_box[2]) // 2
    sy = (source_box[1] + source_box[3]) // 2
    lx = (label_box[0] + label_box[2]) // 2
    ly = (label_box[1] + label_box[3]) // 2
    return (sx, sy), (lx, ly)


def clamp_box(box: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(image_width - 1, int(x1))),
        max(0, min(image_height - 1, int(y1))),
        max(0, min(image_width - 1, int(x2))),
        max(0, min(image_height - 1, int(y2))),
    )


def vision_artifact_to_debug_dict(artifact: VisionArtifact | None) -> dict[str, object] | None:
    if artifact is None:
        return None
    return {
        "mode": artifact.mode,
        "image_path": str(artifact.image_path),
        "number_map": [
            {
                "number": entry.number,
                "line_id": entry.line_id,
                "order_index": entry.order_index,
                "source_text": entry.source_text,
                "translated_text": entry.translated_text,
                "box": entry.box,
                "state_key": entry.state_key,
                "block_id": entry.block_id,
                "source_hash": entry.source_hash,
            }
            for entry in artifact.number_map
        ],
        "preprocessing_warnings": list(artifact.preprocessing_warnings),
    }
