from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detect_ocr import TextBlock
from .line_identity import lookup_translation
from .logging_utils import shorten

logger = logging.getLogger(__name__)


_FONT_REGISTRY: dict[tuple[str, int], ImageFont.ImageFont] = {}
_TEXT_FIT_CACHE: dict[tuple[str, tuple[int, int, int, int], str | None, int, tuple[int, int, int, int] | None], TextFit] = {}
_WRAP_TEXT_CACHE: dict[tuple[tuple[str, int], str, int], tuple[str, ...]] = {}


@dataclass(frozen=True)
class RenderLayout:
    source_box: tuple[int, int, int, int]
    render_box: tuple[int, int, int, int]
    bubble_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class TextFit:
    font_size: int
    line_count: int
    fit_status: str
    clipped: bool
    widest_line: int
    total_height: int
    usable_width: int
    usable_height: int
    attempted_font_size: int
    line_height: int
    overflow_width: int = 0
    overflow_height: int = 0
    lines: tuple[str, ...] = ()
    wrap_score: float = 0.0
    wrap_warnings: tuple[str, ...] = ()
    split_word_count: int = 0
    orphan_line_count: int = 0
    candidate_font_sizes: tuple[int, ...] = ()
    compacted_for_render: bool = False


def plan_render_layouts(
    image_bgr: np.ndarray,
    blocks: list[TextBlock],
    *,
    render_expand: float,
) -> list[RenderLayout]:
    image_height, image_width = image_bgr.shape[:2]
    layouts = [
        choose_render_layout(image_bgr, render_anchor_box(block), image_width, image_height, render_expand)
        for block in blocks
    ]
    layouts = avoid_render_collisions(layouts)
    logger.info("Planned %d render layout(s)", len(layouts))
    return layouts


def avoid_render_collisions(layouts: list[RenderLayout]) -> list[RenderLayout]:
    adjusted: list[RenderLayout] = []
    for index, layout in enumerate(layouts):
        render_box = layout.render_box
        for other_index, other_layout in enumerate(layouts):
            if other_index == index:
                continue
            render_box = trim_render_box_away_from_source(render_box, layout.source_box, other_layout.source_box)
        adjusted.append(RenderLayout(source_box=layout.source_box, render_box=render_box, bubble_box=layout.bubble_box))
    adjusted = balance_overlapping_bubble_rows(adjusted)
    return trim_residual_render_overlaps(adjusted)


def balance_overlapping_bubble_rows(layouts: list[RenderLayout]) -> list[RenderLayout]:
    adjusted = list(layouts)
    indexed = sorted(
        [(index, layout) for index, layout in enumerate(adjusted) if layout.bubble_box is not None],
        key=lambda item: box_center(item[1].source_box)[0],
    )
    used: set[int] = set()
    for start, (index, layout) in enumerate(indexed):
        if index in used:
            continue
        group = [(index, layout)]
        _, cy = box_center(layout.source_box)
        for other_index, other_layout in indexed[start + 1 :]:
            if other_index in used:
                continue
            _ox, oy = box_center(other_layout.source_box)
            source_height = max(layout.source_box[3] - layout.source_box[1], other_layout.source_box[3] - other_layout.source_box[1])
            if abs(oy - cy) > max(22, int(source_height * 0.55)):
                continue
            if any(boxes_intersect(other_layout.bubble_box, member.bubble_box) for _member_index, member in group):
                group.append((other_index, other_layout))
        if len(group) < 3:
            continue
        balanced = balance_bubble_row_group(group)
        if balanced is None:
            continue
        for balanced_index, balanced_layout in balanced:
            adjusted[balanced_index] = balanced_layout
            used.add(balanced_index)
    return adjusted


def balance_bubble_row_group(group: list[tuple[int, RenderLayout]]) -> list[tuple[int, RenderLayout]] | None:
    ordered = sorted(group, key=lambda item: box_center(item[1].source_box)[0])
    bubble_boxes = [layout.bubble_box for _index, layout in ordered if layout.bubble_box is not None]
    if len(bubble_boxes) != len(ordered):
        return None

    union_x1 = min(box[0] for box in bubble_boxes)
    union_x2 = max(box[2] for box in bubble_boxes)
    slot_width = (union_x2 - union_x1) / len(ordered)
    if slot_width < 90:
        return None

    balanced: list[tuple[int, RenderLayout]] = []
    for position, (index, layout) in enumerate(ordered):
        bubble = layout.bubble_box
        assert bubble is not None
        slot_x1 = int(round(union_x1 + slot_width * position))
        slot_x2 = int(round(union_x1 + slot_width * (position + 1)))
        x1 = max(bubble[0], slot_x1)
        x2 = min(bubble[2], slot_x2)
        candidate = (x1, layout.render_box[1], x2, layout.render_box[3])
        if box_area(candidate) < box_area(layout.source_box) * 1.2:
            return None
        if not candidate_keeps_source_anchor(candidate, layout.source_box):
            return None
        balanced.append((index, RenderLayout(source_box=layout.source_box, render_box=candidate, bubble_box=layout.bubble_box)))
    return balanced


def trim_residual_render_overlaps(layouts: list[RenderLayout]) -> list[RenderLayout]:
    adjusted = list(layouts)
    for index, layout in enumerate(adjusted):
        render_box = layout.render_box
        for other_index, other_layout in enumerate(adjusted):
            if other_index == index or not boxes_intersect(render_box, other_layout.render_box):
                continue
            if boxes_intersect(layout.source_box, other_layout.source_box):
                continue
            render_box = trim_small_render_overlap(render_box, layout.source_box, other_layout.render_box)
        if render_box != layout.render_box:
            adjusted[index] = RenderLayout(source_box=layout.source_box, render_box=render_box, bubble_box=layout.bubble_box)
    return adjusted


def trim_small_render_overlap(
    render_box: tuple[int, int, int, int],
    source_box: tuple[int, int, int, int],
    other_box: tuple[int, int, int, int],
    *,
    margin: int = 1,
) -> tuple[int, int, int, int]:
    ix1 = max(render_box[0], other_box[0])
    iy1 = max(render_box[1], other_box[1])
    ix2 = min(render_box[2], other_box[2])
    iy2 = min(render_box[3], other_box[3])
    overlap_width = ix2 - ix1
    overlap_height = iy2 - iy1
    if overlap_width <= 0 or overlap_height <= 0:
        return render_box

    x1, y1, x2, y2 = render_box
    sx, sy = box_center(source_box)
    ox, oy = box_center(other_box)
    candidates: list[tuple[int, int, int, int]] = []
    if overlap_width <= overlap_height:
        if ox >= sx:
            candidates.append((x1, y1, min(x2, other_box[0] - margin), y2))
        else:
            candidates.append((max(x1, other_box[2] + margin), y1, x2, y2))
    else:
        if oy >= sy:
            candidates.append((x1, y1, x2, min(y2, other_box[1] - margin)))
        else:
            candidates.append((x1, max(y1, other_box[3] + margin), x2, y2))

    valid = [
        candidate
        for candidate in candidates
        if box_area(candidate) >= box_area(source_box) and candidate_keeps_source_anchor(candidate, source_box)
    ]
    if not valid:
        return render_box
    return max(valid, key=box_area)


def trim_render_box_away_from_source(
    render_box: tuple[int, int, int, int],
    source_box: tuple[int, int, int, int],
    other_source: tuple[int, int, int, int],
    *,
    margin: int = 5,
) -> tuple[int, int, int, int]:
    if not boxes_intersect(render_box, other_source):
        return render_box
    if boxes_intersect(source_box, other_source):
        return render_box

    x1, y1, x2, y2 = render_box
    sx, sy = box_center(source_box)
    ox, oy = box_center(other_source)
    candidates: list[tuple[int, int, int, int]] = []
    if oy > sy:
        candidates.append((x1, y1, x2, min(y2, other_source[1] - margin)))
    elif oy < sy:
        candidates.append((x1, max(y1, other_source[3] + margin), x2, y2))
    if ox > sx:
        candidates.append((x1, y1, min(x2, other_source[0] - margin), y2))
    elif ox < sx:
        candidates.append((max(x1, other_source[2] + margin), y1, x2, y2))

    valid = [candidate for candidate in candidates if box_area(candidate) >= box_area(source_box) * 1.2 and candidate_keeps_source_anchor(candidate, source_box)]
    if not valid:
        return render_box
    return max(valid, key=box_area)


def plan_text_fits(
    image_bgr: np.ndarray,
    blocks: list[TextBlock],
    translations: dict[str, str],
    *,
    font_path: Path | None,
    base_font_size: int,
    render_expand: float,
    render_layouts: list[RenderLayout] | None = None,
) -> list[TextFit]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    if render_layouts is None:
        render_layouts = plan_render_layouts(image_bgr, blocks, render_expand=render_expand)
    fits = [
        measure_fitted_text(
            draw,
            lookup_translation(translations, block, block.text).strip(),
            layout.render_box,
            font_path=font_path,
            base_font_size=base_font_size,
            source_box=layout.source_box,
        )
        for block, layout in zip(blocks, render_layouts)
    ]
    logger.info("Planned %d render text fit(s)", len(fits))
    return fits


def render_translations(
    image_bgr: np.ndarray,
    blocks: list[TextBlock],
    translations: dict[str, str],
    *,
    font_path: Path | None,
    base_font_size: int,
    render_expand: float,
    render_layouts: list[RenderLayout] | None = None,
) -> np.ndarray:
    logger.info("Rendering %d translated text block(s)", len(blocks))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    image_width, image_height = pil_image.size
    if render_layouts is None:
        render_layouts = plan_render_layouts(image_bgr, blocks, render_expand=render_expand)

    for block, layout in zip(blocks, render_layouts):
        translated = lookup_translation(translations, block, block.text).strip()
        if not translated:
            logger.debug("Skipping empty translation for source=%s", shorten(block.text))
            continue

        box = layout.render_box
        logger.debug(
            "Rendering block: source=%s translated=%s original_box=%s render_box=%s",
            shorten(block.text),
            shorten(translated),
            block.box,
            box,
        )
        anchor_box = layout.source_box
        draw_fitted_text(
            draw,
            translated,
            box,
            font_path=font_path,
            base_font_size=base_font_size,
            source_box=anchor_box,
        )

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def expand_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    factor: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    x_factor = factor
    y_factor = 1.25
    if height > width * 1.6:
        x_factor = max(factor, 4.6)
        y_factor = 1.18
    elif width > height * 3.0:
        y_factor = max(y_factor, 4.0)

    new_width = int(width * x_factor)
    if height > width * 1.6:
        new_width = min(new_width, 125)
    else:
        new_width = min(new_width, 240)
    new_width = min(image_width, max(width, new_width))
    new_height = int(height * y_factor)
    if width > height * 3.0:
        new_height = max(new_height, min(48, height + 34))
    new_height = min(image_height, max(height, new_height))
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    nx1 = max(0, center_x - new_width // 2)
    ny1 = max(0, center_y - new_height // 2)
    nx2 = min(image_width - 1, nx1 + new_width)
    ny2 = min(image_height - 1, ny1 + new_height)

    if nx2 == image_width - 1:
        nx1 = max(0, nx2 - new_width)
    if ny2 == image_height - 1:
        ny1 = max(0, ny2 - new_height)

    return nx1, ny1, nx2, ny2


def render_anchor_box(block: TextBlock) -> tuple[int, int, int, int]:
    grouped_from = block.metadata.get("grouped_from") if getattr(block, "metadata", None) else None
    if not isinstance(grouped_from, list) or len(grouped_from) < 2:
        return block.box

    member_boxes = []
    for member in grouped_from:
        if not isinstance(member, dict):
            continue
        box = member.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                member_boxes.append(tuple(int(value) for value in box))
            except (TypeError, ValueError):
                continue
    if len(member_boxes) < 2:
        return block.box

    primary = max(member_boxes, key=box_area)
    primary_height = max(1, primary[3] - primary[1])
    primary_width = max(1, primary[2] - primary[0])
    kept = [primary]
    for box in member_boxes:
        if box == primary:
            continue
        width = max(1, box[2] - box[0])
        height = max(1, box[3] - box[1])
        vertical_overlap = max(0, min(primary[3], box[3]) - max(primary[1], box[1]))
        horizontal_overlap = max(0, min(primary[2], box[2]) - max(primary[0], box[0]))
        vertical_overlap_ratio = vertical_overlap / max(1, min(primary_height, height))
        horizontal_overlap_ratio = horizontal_overlap / max(1, min(primary_width, width))
        horizontal_gap = max(0, max(primary[0], box[0]) - min(primary[2], box[2]))
        vertical_gap = max(0, max(primary[1], box[1]) - min(primary[3], box[3]))
        if vertical_overlap_ratio >= 0.55 and horizontal_gap <= max(20, int((primary_width + width) * 1.1)):
            kept.append(box)
        elif horizontal_overlap_ratio >= 0.55 and vertical_gap <= max(18, int((primary_height + height) * 0.8)):
            kept.append(box)

    if len(kept) == len(member_boxes):
        return block.box
    anchor = (
        min(box[0] for box in kept),
        min(box[1] for box in kept),
        max(box[2] for box in kept),
        max(box[3] for box in kept),
    )
    logger.debug("Using render anchor box for grouped block: group_box=%s anchor=%s members=%d kept=%d", block.box, anchor, len(member_boxes), len(kept))
    return anchor


def choose_render_box(
    image_bgr: np.ndarray,
    source_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    factor: float,
) -> tuple[int, int, int, int]:
    expanded = expand_box(source_box, image_width, image_height, factor)
    if box_area(source_box) < 700:
        return expanded

    bubble_box = find_white_region_box(image_bgr, source_box, expanded)
    if bubble_box is None:
        return expanded

    source_area = box_area(source_box)
    bubble_area = box_area(bubble_box)
    if not is_usable_bubble_box(source_box, expanded, bubble_box, image_width, image_height):
        logger.debug("Ignoring unusable bubble candidate: source=%s expanded=%s bubble=%s", source_box, expanded, bubble_box)
        return expanded

    if bubble_area < source_area * 1.2:
        logger.debug("Ignoring tiny bubble candidate: source=%s bubble=%s", source_box, bubble_box)
        return expanded
    if bubble_area < box_area(expanded) * 0.32:
        logger.debug("Ignoring cramped bubble candidate: source=%s expanded=%s bubble=%s", source_box, expanded, bubble_box)
        return expanded

    logger.debug("Using white-region render box: source=%s expanded=%s bubble=%s", source_box, expanded, bubble_box)
    return bubble_box


def choose_render_layout(
    image_bgr: np.ndarray,
    source_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    factor: float,
) -> RenderLayout:
    expanded = expand_box(source_box, image_width, image_height, factor)
    expanded = tighten_large_text_box(source_box, expanded)
    if box_area(source_box) < 700:
        return RenderLayout(source_box=source_box, render_box=expanded)

    source_area = box_area(source_box)
    bubble_box = find_usable_bubble_box(image_bgr, source_box, expanded, image_width, image_height)
    if bubble_box is None:
        return RenderLayout(source_box=source_box, render_box=fallback_render_box(source_box, expanded))
    bubble_area = box_area(bubble_box)

    if bubble_area < source_area * 1.2:
        logger.debug("Ignoring tiny bubble candidate: source=%s bubble=%s", source_box, bubble_box)
        return RenderLayout(source_box=source_box, render_box=fallback_render_box(source_box, expanded))
    if bubble_area < box_area(expanded) * 0.32:
        logger.debug("Ignoring cramped bubble candidate: source=%s expanded=%s bubble=%s", source_box, expanded, bubble_box)
        return RenderLayout(source_box=source_box, render_box=fallback_render_box(source_box, expanded))

    safe_bubble_box = inset_text_box(bubble_box)
    if is_outline_bubble_candidate(source_box, bubble_box, image_width, image_height):
        render_box = safe_bubble_box
    else:
        render_box = clamp_box_to_container(tighten_large_text_box(source_box, safe_bubble_box), safe_bubble_box)
    logger.debug(
        "Using white-region render box: source=%s expanded=%s bubble=%s safe_bubble=%s render=%s",
        source_box,
        expanded,
        bubble_box,
        safe_bubble_box,
        render_box,
    )
    return RenderLayout(source_box=source_box, render_box=render_box, bubble_box=bubble_box)


def find_usable_bubble_box(
    image_bgr: np.ndarray,
    source_box: tuple[int, int, int, int],
    expanded: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    bubble_box = find_white_region_box(image_bgr, source_box, expanded)
    if bubble_box is not None and is_usable_bubble_box(source_box, expanded, bubble_box, image_width, image_height):
        return bubble_box
    if bubble_box is not None:
        logger.debug("Ignoring unusable white bubble candidate: source=%s expanded=%s bubble=%s", source_box, expanded, bubble_box)

    outline_box = find_dark_outline_bubble_box(image_bgr, source_box, expanded)
    if outline_box is not None and is_usable_outline_bubble_box(source_box, outline_box, image_width, image_height):
        return outline_box
    if outline_box is not None:
        logger.debug("Ignoring unusable outline bubble candidate: source=%s expanded=%s bubble=%s", source_box, expanded, outline_box)
    return None


def fallback_render_box(
    source_box: tuple[int, int, int, int],
    expanded_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    source_width = max(1, source_box[2] - source_box[0])
    source_height = max(1, source_box[3] - source_box[1])
    source_area = box_area(source_box)
    if source_area >= 15000 and source_width >= 58 and source_height >= 45:
        logger.debug("Using source-box fallback render area: source=%s expanded=%s", source_box, expanded_box)
        return source_box
    return expanded_box


def inset_text_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])
    inset = max(4, min(10, min(width, height) // 12))
    if width <= inset * 3 or height <= inset * 3:
        return box
    return box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset


def tighten_large_text_box(
    source_box: tuple[int, int, int, int],
    render_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    source_width = max(1, source_box[2] - source_box[0])
    source_height = max(1, source_box[3] - source_box[1])
    render_width = max(1, render_box[2] - render_box[0])
    render_height = max(1, render_box[3] - render_box[1])
    source_area = box_area(source_box)
    render_area = box_area(render_box)
    if source_area < 2500 and render_area < source_area * 6:
        return render_box

    min_width = 125 if source_width <= 35 and source_height >= source_width * 4 else 95
    max_width = int(min(render_width, max(source_width * 1.35, min_width)))
    max_height = int(min(render_height, max(source_height * 1.05, 58)))
    if max_width >= render_width and max_height >= render_height:
        return render_box

    center_x, center_y = box_center(source_box)
    nx1 = int(round(center_x - max_width / 2))
    ny1 = int(round(center_y - max_height / 2))
    tightened = (nx1, ny1, nx1 + max_width, ny1 + max_height)
    return clamp_box_to_container(tightened, render_box)


def clamp_box_to_container(
    box: tuple[int, int, int, int],
    container: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])
    cx1, cy1, cx2, cy2 = container
    nx1 = min(max(box[0], cx1), max(cx1, cx2 - width))
    ny1 = min(max(box[1], cy1), max(cy1, cy2 - height))
    nx2 = min(cx2, nx1 + width)
    ny2 = min(cy2, ny1 + height)
    return nx1, ny1, nx2, ny2


def is_usable_bubble_box(
    source_box: tuple[int, int, int, int],
    expanded_box: tuple[int, int, int, int],
    bubble_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> bool:
    source_area = max(1, box_area(source_box))
    bubble_area = max(1, box_area(bubble_box))
    expanded_area = max(1, box_area(expanded_box))
    image_area = max(1, image_width * image_height)
    bubble_width = max(1, bubble_box[2] - bubble_box[0])
    bubble_height = max(1, bubble_box[3] - bubble_box[1])
    aspect = bubble_width / bubble_height

    if bubble_area / image_area > 0.09:
        return False
    if bubble_area > expanded_area * 1.65:
        return False
    if bubble_area > source_area * 18:
        return False
    if aspect > 4.5 or aspect < 0.22:
        return False
    if touches_panel_edge(bubble_box, image_width, image_height) and bubble_area > source_area * 5:
        return False
    return True


def is_usable_outline_bubble_box(
    source_box: tuple[int, int, int, int],
    bubble_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> bool:
    source_area = max(1, box_area(source_box))
    bubble_area = max(1, box_area(bubble_box))
    image_area = max(1, image_width * image_height)
    bubble_width = max(1, bubble_box[2] - bubble_box[0])
    bubble_height = max(1, bubble_box[3] - bubble_box[1])
    aspect = bubble_width / bubble_height

    if not point_in_box(box_center(source_box), bubble_box):
        return False
    if bubble_area / image_area > 0.18:
        return False
    if bubble_area < source_area * 2.0 or bubble_area > source_area * 10.0:
        return False
    if aspect > 3.2 or aspect < 0.31:
        return False
    return True


def is_outline_bubble_candidate(
    source_box: tuple[int, int, int, int],
    bubble_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> bool:
    return is_usable_outline_bubble_box(source_box, bubble_box, image_width, image_height)


def touches_panel_edge(box: tuple[int, int, int, int], image_width: int, image_height: int) -> bool:
    margin = 3
    return box[0] <= margin or box[1] <= margin or box[2] >= image_width - 1 - margin or box[3] >= image_height - 1 - margin


def find_white_region_box(
    image_bgr: np.ndarray,
    source_box: tuple[int, int, int, int],
    expanded_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    height, width = image_bgr.shape[:2]
    search_box = pad_box(expanded_box, width, height, 24)
    sx1, sy1, sx2, sy2 = search_box
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    gray = cv2.cvtColor(image_bgr[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
    white_mask = (gray >= 245).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(white_mask, 8)

    best_box = None
    best_score = 0.0
    source_local = (source_box[0] - sx1, source_box[1] - sy1, source_box[2] - sx1, source_box[3] - sy1)
    source_center = box_center(source_local)
    search_area = max(1, (sx2 - sx1) * (sy2 - sy1))

    for index in range(1, component_count):
        x, y, component_width, component_height, area = stats[index]
        if area < max(25, box_area(source_local) * 0.5):
            continue

        local_box = (int(x), int(y), int(x + component_width), int(y + component_height))
        if area / search_area > 0.82:
            continue
        if not point_in_box(source_center, local_box) and box_iou(source_local, local_box) < 0.12:
            continue

        score = box_iou(source_local, local_box) + min(1.0, area / max(1, box_area(source_local)) / 12)
        if score > best_score:
            best_score = score
            best_box = local_box

    if best_box is None:
        return None

    gx1, gy1, gx2, gy2 = best_box[0] + sx1, best_box[1] + sy1, best_box[2] + sx1, best_box[3] + sy1
    inset = 5
    if gx2 - gx1 > inset * 3 and gy2 - gy1 > inset * 3:
        gx1 += inset
        gy1 += inset
        gx2 -= inset
        gy2 -= inset
    return clamp_box((gx1, gy1, gx2, gy2), width, height)


def find_dark_outline_bubble_box(
    image_bgr: np.ndarray,
    source_box: tuple[int, int, int, int],
    expanded_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    source_width = max(1, source_box[2] - source_box[0])
    source_height = max(1, source_box[3] - source_box[1])
    if source_width < source_height * 0.8:
        return None

    height, width = image_bgr.shape[:2]
    search_padding = max(36, min(72, max(source_width, source_height)))
    search_box = pad_box(expanded_box, width, height, search_padding)
    sx1, sy1, sx2, sy2 = search_box
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    gray = cv2.cvtColor(image_bgr[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
    dark_mask = (gray <= 120).astype("uint8")
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_mask, 8)
    source_local = (source_box[0] - sx1, source_box[1] - sy1, source_box[2] - sx1, source_box[3] - sy1)
    source_center = box_center(source_local)
    source_area = max(1, box_area(source_local))

    best_box = None
    best_score = 0.0
    for index in range(1, component_count):
        x, y, component_width, component_height, area = stats[index]
        if area < 80:
            continue
        local_box = (int(x), int(y), int(x + component_width), int(y + component_height))
        local_area = box_area(local_box)
        if local_area < source_area * 2.0:
            continue
        if not point_in_box(source_center, local_box):
            continue
        fill_ratio = area / max(1, local_area)
        if fill_ratio > 0.35:
            continue
        aspect = max(1, component_width) / max(1, component_height)
        if aspect > 3.2 or aspect < 0.31:
            continue
        score = local_area / source_area - fill_ratio
        if score > best_score:
            best_score = score
            best_box = local_box

    if best_box is None:
        return None

    return clamp_box((best_box[0] + sx1, best_box[1] + sy1, best_box[2] + sx1, best_box[3] + sy1), width, height)


def pad_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return clamp_box((x1 - padding, y1 - padding, x2 + padding, y2 + padding), image_width, image_height)


def clamp_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(image_width - 1, x1)),
        max(0, min(image_height - 1, y1)),
        max(0, min(image_width - 1, x2)),
        max(0, min(image_height - 1, y2)),
    )


def box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_in_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    return intersection / max(1, box_area(a) + box_area(b) - intersection)


def boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def candidate_keeps_source_anchor(candidate: tuple[int, int, int, int], source_box: tuple[int, int, int, int]) -> bool:
    if point_in_box(box_center(source_box), candidate):
        return True
    ix1 = max(candidate[0], source_box[0])
    iy1 = max(candidate[1], source_box[1])
    ix2 = min(candidate[2], source_box[2])
    iy2 = min(candidate[3], source_box[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    overlap_area = (ix2 - ix1) * (iy2 - iy1)
    return overlap_area / max(1, box_area(source_box)) >= 0.35


def draw_fitted_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    font_path: Path | None,
    base_font_size: int,
    source_box: tuple[int, int, int, int] | None = None,
) -> TextFit:
    return fit_text(
        draw,
        text,
        box,
        font_path=font_path,
        base_font_size=base_font_size,
        source_box=source_box,
        draw_text=True,
    )


def measure_fitted_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    font_path: Path | None,
    base_font_size: int,
    source_box: tuple[int, int, int, int] | None = None,
) -> TextFit:
    return fit_text(
        draw,
        text,
        box,
        font_path=font_path,
        base_font_size=base_font_size,
        source_box=source_box,
        draw_text=False,
    )


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    font_path: Path | None,
    base_font_size: int,
    source_box: tuple[int, int, int, int] | None,
    draw_text: bool,
) -> TextFit:
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    min_dimension = min(box_width, box_height)
    padding = max(4, min_dimension // 24)
    if min_dimension < 52:
        padding = max(padding, 6)
    usable_width = max(1, box_width - padding * 2)
    usable_height = max(1, box_height - padding * 2)

    start_font_size = min(base_font_size, estimate_start_font_size(text, box, source_box, base_font_size))
    min_font_size = 5
    if len(text.split()) <= 2 and should_cap_single_line_start(draw, text, font_path, start_font_size, usable_width, usable_height):
        start_font_size = cap_single_line_start_font_size(
            draw,
            text,
            font_path,
            start_font_size,
            min_font_size,
            usable_width,
            usable_height,
        )
    if 3 <= len(text.split()) <= 6 and should_cap_two_line_start(draw, text, font_path, start_font_size, usable_width, usable_height):
        start_font_size = cap_two_line_start_font_size(draw, font_path, start_font_size, min_font_size, usable_height)
    if 3 <= len(text.split()) <= 6:
        start_font_size = cap_two_line_width_start_font_size(
            draw,
            text,
            font_path,
            start_font_size,
            min_font_size,
            usable_width,
            usable_height,
        )
    cache_key = text_fit_cache_key(text, box, font_path, base_font_size, source_box)
    cached_fit = _TEXT_FIT_CACHE.get(cache_key)
    if draw_text and cached_fit is not None:
        draw_cached_text_fit(draw, cached_fit, box, font_path=font_path, padding=padding)
        return cached_fit

    selected: tuple[int, ImageFont.ImageFont, list[str], int, int, int, float, tuple[str, ...], int, int] | None = None
    candidate_font_sizes: list[int] = []

    for font_size in range(start_font_size, min_font_size - 1, -1):
        if selected is not None and selected[0] > 8 and font_size <= 8:
            break
        font = load_font(font_path, font_size)
        candidate_font_sizes.append(font_size)
        if font_size > min_font_size and has_overwide_word(draw, text, font, usable_width):
            continue
        lines = wrap_text(draw, text, font, usable_width)
        if not lines:
            continue

        line_height = text_height(draw, "Ag", font) + max(1, font_size // 10)
        total_height = line_height * len(lines)
        widest = max(text_width(draw, line, font) for line in lines)
        if total_height <= usable_height and widest <= usable_width:
            warnings, split_count, orphan_count = wrap_metrics_for_lines(text, lines)
            score = render_fit_score(lines, font_size, start_font_size, warnings=warnings, split_word_count=split_count, orphan_line_count=orphan_count)
            if selected is None or is_better_fit_candidate(font_size, score, selected[0], selected[6]):
                selected = (font_size, font, lines, line_height, total_height, widest, score, warnings, split_count, orphan_count)
                if can_accept_fit_early(font_size, lines, score, warnings, split_count, orphan_count):
                    break

    if selected is not None:
        font_size, font, lines, line_height, total_height, widest, score, warnings, split_count, orphan_count = selected
        logger.debug(
            "Text fit: font_size=%d lines=%d score=%.2f warnings=%s box=%s text=%s",
            font_size,
            len(lines),
            score,
            ",".join(warnings),
            box,
            shorten(text),
        )
        y = y1 + padding + max(0, (usable_height - total_height) // 2)
        if draw_text:
            for line in lines:
                line_width = text_width(draw, line, font)
                x = x1 + padding + max(0, (usable_width - line_width) // 2)
                draw.text((x, y), line, font=font, fill=(0, 0, 0))
                y += line_height
        status = "fit" if font_size > min_font_size else "fit_min_font"
        fit = TextFit(
            font_size=font_size,
            line_count=len(lines),
            fit_status=status,
            clipped=False,
            widest_line=widest,
            total_height=total_height,
            usable_width=usable_width,
            usable_height=usable_height,
            attempted_font_size=start_font_size,
            line_height=line_height,
            lines=tuple(lines),
            wrap_score=round(score, 3),
            wrap_warnings=warnings,
            split_word_count=split_count,
            orphan_line_count=orphan_count,
            candidate_font_sizes=tuple(candidate_font_sizes),
        )
        _TEXT_FIT_CACHE[cache_key] = fit
        return fit

    logger.warning("Text did not fit cleanly; drawing clipped fallback at font size %d: box=%s text=%s", min_font_size, box, shorten(text))
    font = load_font(font_path, min_font_size)
    line_height = text_height(draw, "Ag", font) + 1
    max_lines = max(1, usable_height // max(1, line_height))
    lines = wrap_text(draw, text, font, usable_width)[:max_lines]
    total_height = line_height * len(lines)
    widest = max((text_width(draw, line, font) for line in lines), default=0)
    warnings, split_count, orphan_count = wrap_metrics_for_lines(text, lines)
    score = render_fit_score(lines, min_font_size, start_font_size, warnings=warnings, split_word_count=split_count, orphan_line_count=orphan_count)
    y = y1 + padding + max(0, (usable_height - total_height) // 2)
    if draw_text:
        for line in lines:
            line_width = text_width(draw, line, font)
            x = x1 + padding + max(0, (usable_width - line_width) // 2)
            if y + line_height > y2 - padding + 1:
                break
            draw.text((x, y), line, font=font, fill=(0, 0, 0))
            y += line_height
    fit = TextFit(
        font_size=min_font_size,
        line_count=len(lines),
        fit_status="clipped",
        clipped=True,
        widest_line=widest,
        total_height=total_height,
        usable_width=usable_width,
        usable_height=usable_height,
        attempted_font_size=start_font_size,
        line_height=line_height,
        overflow_width=max(0, widest - usable_width),
        overflow_height=max(0, total_height - usable_height),
        lines=tuple(lines),
        wrap_score=round(score, 3),
        wrap_warnings=warnings,
        split_word_count=split_count,
        orphan_line_count=orphan_count,
        candidate_font_sizes=tuple(candidate_font_sizes),
    )
    _TEXT_FIT_CACHE[cache_key] = fit
    return fit


def should_cap_single_line_start(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path | None,
    start_font_size: int,
    usable_width: int,
    usable_height: int,
) -> bool:
    font = load_font(font_path, start_font_size)
    word_count = len(text.split())
    if word_count == 1 and text_width(draw, text, font) > usable_width:
        return True
    if word_count == 2 and usable_height >= 50 and text_width(draw, text, font) > usable_width:
        return True
    if usable_height <= 24:
        return True
    if usable_height > 44:
        return False
    line_height = text_height(draw, "Ag", font) + max(1, start_font_size // 10)
    if line_height * 2 <= usable_height:
        return False
    return text_width(draw, text, font) <= usable_width * 1.35


def cap_single_line_start_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path | None,
    start_font_size: int,
    min_font_size: int,
    usable_width: int,
    usable_height: int,
) -> int:
    for font_size in range(start_font_size, min_font_size - 1, -1):
        font = load_font(font_path, font_size)
        line_height = text_height(draw, "Ag", font) + max(1, font_size // 10)
        if line_height <= usable_height and text_width(draw, text, font) <= usable_width:
            return font_size
    return start_font_size


def should_cap_two_line_start(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path | None,
    start_font_size: int,
    usable_width: int,
    usable_height: int,
) -> bool:
    if usable_height > 60:
        return False
    font = load_font(font_path, start_font_size)
    if text_width(draw, text, font) <= usable_width:
        return False
    if any(text_width(draw, word, font) > usable_width for word in text.split()):
        return False
    line_height = text_height(draw, "Ag", font) + max(1, start_font_size // 10)
    return line_height * 2 > usable_height


def cap_two_line_start_font_size(
    draw: ImageDraw.ImageDraw,
    font_path: Path | None,
    start_font_size: int,
    min_font_size: int,
    usable_height: int,
) -> int:
    for font_size in range(start_font_size, min_font_size - 1, -1):
        font = load_font(font_path, font_size)
        line_height = text_height(draw, "Ag", font) + max(1, font_size // 10)
        if line_height * 2 <= usable_height:
            return font_size
    return start_font_size


def cap_two_line_width_start_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path | None,
    start_font_size: int,
    min_font_size: int,
    usable_width: int,
    usable_height: int,
) -> int:
    three_line_fit: int | None = None
    for font_size in range(start_font_size, min_font_size - 1, -1):
        font = load_font(font_path, font_size)
        lines = wrap_text(draw, text, font, usable_width)
        if not lines or len(lines) > 3:
            continue
        line_height = text_height(draw, "Ag", font) + max(1, font_size // 10)
        if line_height * len(lines) > usable_height or any(text_width(draw, line, font) > usable_width for line in lines):
            continue
        if len(lines) <= 2:
            if font_size <= 8 and three_line_fit is not None:
                return three_line_fit
            return font_size
        if font_size > 8 and three_line_fit is None:
            three_line_fit = font_size
    return start_font_size


def is_better_fit_candidate(
    font_size: int,
    score: float,
    selected_font_size: int,
    selected_score: float,
) -> bool:
    if font_size > 8 and selected_font_size <= 8:
        return True
    if font_size <= 8 and selected_font_size > 8:
        return False
    return score < selected_score


def can_accept_fit_early(
    font_size: int,
    lines: list[str],
    score: float,
    warnings: tuple[str, ...],
    split_word_count: int,
    orphan_line_count: int,
) -> bool:
    if font_size <= 8 or len(lines) > 4 or score > 80:
        return False
    if split_word_count > 0 or orphan_line_count > 0:
        return False
    return not warnings


def text_fit_cache_key(
    text: str,
    box: tuple[int, int, int, int],
    font_path: Path | None,
    base_font_size: int,
    source_box: tuple[int, int, int, int] | None,
) -> tuple[str, tuple[int, int, int, int], str | None, int, tuple[int, int, int, int] | None]:
    return text, box, str(font_path) if font_path is not None else None, base_font_size, source_box


def draw_cached_text_fit(
    draw: ImageDraw.ImageDraw,
    fit: TextFit,
    box: tuple[int, int, int, int],
    *,
    font_path: Path | None,
    padding: int,
) -> None:
    x1, y1, _x2, y2 = box
    font = load_font(font_path, fit.font_size)
    y = y1 + padding + max(0, (fit.usable_height - fit.total_height) // 2)
    for line in fit.lines:
        line_width = text_width(draw, line, font)
        x = x1 + padding + max(0, (fit.usable_width - line_width) // 2)
        if fit.clipped and y + fit.line_height > y2 - padding + 1:
            break
        draw.text((x, y), line, font=font, fill=(0, 0, 0))
        y += fit.line_height


def render_fit_score(
    lines: list[str],
    font_size: int,
    start_font_size: int,
    *,
    warnings: tuple[str, ...] | None = None,
    split_word_count: int = 0,
    orphan_line_count: int = 0,
) -> float:
    warnings = warnings or ()
    score = (start_font_size - font_size) * 5
    score += max(0, len(lines) - 1) * 4
    score += split_word_count * 180
    score += orphan_line_count * 140
    score += sum(120 for warning in warnings if warning in {"tiny_line", "isolated_punctuation"})
    score += sum(35 for warning in warnings if warning == "weak_line_end")
    if len(lines) >= 5:
        score += (len(lines) - 4) * 35
    if len(lines) > 2 and is_short_single_word_line(lines[-1]):
        score += 120
    if any(len(line.split()) == 1 and len(line.strip(".,!?;:'\"")) <= 3 for line in lines[:-1]):
        score += 70
    score += line_raggedness_score(lines)
    return score


def has_bad_render_wrap(lines: list[str]) -> bool:
    if orphan_line_count(lines) > 0:
        return True
    return any(len(line.split()) == 1 and len(line.strip(".,!?;:'\"")) <= 3 for line in lines[:-1])


def is_short_single_word_line(line: str) -> bool:
    return len(line.split()) == 1 and len(line.strip(".,!?;:'\"")) <= 6


def wrap_warnings_for_lines(source_text: str, lines: list[str]) -> tuple[str, ...]:
    warnings, _split_count, _orphan_count = wrap_metrics_for_lines(source_text, lines)
    return warnings


def wrap_metrics_for_lines(source_text: str, lines: list[str]) -> tuple[tuple[str, ...], int, int]:
    split_count = split_word_count(source_text, lines)
    orphan_count = orphan_line_count(lines)
    warnings: list[str] = []
    if has_bad_render_wrap(lines):
        warnings.append("bad_wrap")
    if orphan_count:
        warnings.append("orphan_line")
    if split_count:
        warnings.append("split_word")
    for line in lines:
        stripped = line.strip()
        core = stripped.strip(".,!?;:'\"")
        if len(core) <= 2 and stripped:
            warnings.append("tiny_line")
        if re.fullmatch(r"[.,!?;:'\"-]+", stripped):
            warnings.append("isolated_punctuation")
        if line_ends_weakly(stripped):
            warnings.append("weak_line_end")
    return tuple(dict.fromkeys(warnings)), split_count, orphan_count


def line_ends_weakly(line: str) -> bool:
    normalized = line.rstrip().lower().strip("\"'")
    if not normalized:
        return False
    return normalized.endswith((",", ":", ";", "-", " and", " or", " the", " a", " an", " to", " of", " in", " on", " for", " with"))


def orphan_line_count(lines: list[str]) -> int:
    if len(lines) <= 1:
        return 0
    count = 0
    for index, line in enumerate(lines):
        if not is_orphan_line(line, is_edge=index in {0, len(lines) - 1}, line_count=len(lines)):
            continue
        if index == len(lines) - 1 or index == 0:
            count += 1
    return count


def is_orphan_line(line: str, *, is_edge: bool = True, line_count: int = 3) -> bool:
    stripped = line.strip().strip(".,!?;:'\"").lower()
    if not stripped:
        return True
    if len(stripped) <= 2:
        return True
    if stripped in {"you", "it", "up", "me", "we", "he", "she", "i", "a", "an", "the", "to", "of", "in", "on", "for"}:
        return True
    if is_edge and line_count <= 2:
        return False
    return len(line.split()) == 1 and len(stripped) <= 4


def split_word_count(source_text: str, lines: list[str]) -> int:
    source_tokens = {normalize_word_token(word) for word in source_text.split()}
    source_tokens = {token for token in source_tokens if token}
    count = 0
    for line in lines:
        parts = line.split()
        if len(parts) != 1:
            continue
        token = normalize_word_token(parts[0])
        if not token or len(token) <= 2 or token in source_tokens:
            continue
        if any(token in source and token != source for source in source_tokens):
            count += 1
    return count


def normalize_word_token(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", text.lower())


def line_raggedness_score(lines: list[str]) -> float:
    if len(lines) <= 1:
        return 0.0
    lengths = [max(1, render_text_length(line)) for line in lines]
    average = sum(lengths) / len(lengths)
    return sum(((length - average) / max(1, average)) ** 2 for length in lengths) * 18


def has_overwide_word(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> bool:
    return any(text_width(draw, word, font) > max_width for word in text.split())


def estimate_start_font_size(
    text: str,
    render_box: tuple[int, int, int, int],
    source_box: tuple[int, int, int, int] | None,
    base_font_size: int,
) -> int:
    text_len = max(1, render_text_length(text))
    render_area = max(1, box_area(render_box))
    x1, y1, x2, y2 = render_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect = width / height

    # The fit loop will shrink as needed, so start from what the chosen render
    # box can plausibly support instead of the often tiny Japanese OCR box.
    density_multiplier = 1.18 if text_len <= 20 else 1.04
    density_cap = int(math.sqrt(render_area / text_len) * density_multiplier)
    height_cap = int(height * (0.72 if text_len <= 16 else 0.58))
    width_cap = int(width * (0.42 if aspect < 1.25 else 0.32))
    dimension_cap = max(6, min(height_cap, width_cap))
    estimated = max(5, min(base_font_size, dimension_cap, max(6, density_cap)))
    if text_len >= 70:
        estimated = min(estimated, 11)
    if source_box is not None and is_long_translation_from_narrow_vertical_source(text, source_box):
        estimated = min(estimated, 16)
        if text_len >= 50:
            estimated = min(estimated, 10)
    if 34 <= text_len <= 46 and render_area <= 6500:
        estimated = min(estimated, 9)
    if 40 <= text_len <= 60 and 0.8 <= aspect <= 1.4:
        estimated = min(estimated, 15)
    logger.debug(
        "Estimated start font size: size=%d base=%d text_len=%d density_cap=%d dimension_cap=%d render_box=%s source_box=%s",
        estimated,
        base_font_size,
        text_len,
        density_cap,
        dimension_cap,
        render_box,
        source_box,
    )
    return estimated


def is_long_translation_from_narrow_vertical_source(text: str, source_box: tuple[int, int, int, int]) -> bool:
    text_len = render_text_length(text)
    source_width = max(1, source_box[2] - source_box[0])
    source_height = max(1, source_box[3] - source_box[1])
    return text_len >= 24 and source_width <= 35 and source_height >= source_width * 4


def render_text_length(text: str) -> int:
    compact = text.strip()
    if not compact:
        return 1
    punctuation_discount = sum(1 for char in compact if char in ".,!?;:'\"-()[]{}<>")
    space_discount = compact.count(" ")
    return max(1, len(compact) - punctuation_discount // 2 - space_discount // 3)


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    font_key = registered_font_key(font)
    cache_key = (font_key, text, max_width) if font_key is not None else None
    if cache_key is not None:
        cached = _WRAP_TEXT_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)
    words = text.split()
    if not words:
        return []
    if any(text_width(draw, word, font) > max_width for word in words):
        lines = wrap_text_greedy(draw, text, font, max_width)
        if cache_key is not None:
            _WRAP_TEXT_CACHE[cache_key] = tuple(lines)
        return lines
    if len(words) <= 14:
        balanced = wrap_text_balanced(draw, words, font, max_width)
        if balanced:
            if cache_key is not None:
                _WRAP_TEXT_CACHE[cache_key] = tuple(balanced)
            return balanced
    lines = wrap_text_greedy(draw, text, font, max_width)
    if cache_key is not None:
        _WRAP_TEXT_CACHE[cache_key] = tuple(lines)
    return lines


def wrap_text_greedy(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.extend(split_long_word(draw, current, font, max_width))
            current = word
    lines.extend(split_long_word(draw, current, font, max_width))
    return lines


def wrap_text_balanced(
    draw: ImageDraw.ImageDraw,
    words: list[str],
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str] | None:
    greedy_lines = wrap_text_greedy(draw, " ".join(words), font, max_width)
    if not greedy_lines:
        return None
    line_options = build_line_options(draw, words, font, max_width)
    min_lines = max(1, len(greedy_lines) - 1)
    max_lines = min(len(words), len(greedy_lines) + 2)
    best_lines: list[str] | None = None
    best_score: float | None = None
    for line_count in range(min_lines, max_lines + 1):
        candidate = best_line_break_for_count(words, max_width, line_count, target_lines=len(greedy_lines), line_options=line_options)
        if candidate is None:
            continue
        lines, score = candidate
        if best_score is None or score < best_score:
            best_score = score
            best_lines = lines
    return best_lines


def build_line_options(
    draw: ImageDraw.ImageDraw,
    words: list[str],
    font: ImageFont.ImageFont,
    max_width: int,
) -> dict[tuple[int, int], tuple[int, str, float]]:
    target_width = max_width * 0.78
    line_options: dict[tuple[int, int], tuple[int, str, float]] = {}
    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            text = " ".join(words[start:end])
            width = text_width(draw, text, font)
            if width > max_width:
                break
            line_options[(start, end)] = (width, text, line_cost(text, width, max_width, target_width))
    return line_options


def best_line_break_for_count(
    words: list[str],
    max_width: int,
    line_count: int,
    *,
    target_lines: int,
    line_options: dict[tuple[int, int], tuple[int, str, float]],
) -> tuple[list[str], float] | None:
    if line_count <= 0 or line_count > len(words):
        return None

    states: dict[tuple[int, int], tuple[float, list[str], list[int]]] = {(0, 0): (0.0, [], [])}
    for used_lines in range(line_count):
        next_states: dict[tuple[int, int], tuple[float, list[str], list[int]]] = {}
        for (start, _used), (score, lines, widths) in states.items():
            remaining_lines = line_count - used_lines
            min_end = start + 1
            max_end = len(words) - remaining_lines + 1
            for end in range(min_end, max_end + 1):
                options = line_options.get((start, end))
                if not options:
                    continue
                width, line, cost = options
                if used_lines + 1 < line_count and line.endswith((",", ":", ";", "-", "and", "or", "the", "a", "an")):
                    cost += 10
                if used_lines + 1 == line_count and line_count > 1:
                    fill = width / max(1, max_width)
                    if fill < 0.42:
                        cost += (0.42 - fill) * 90
                new_score = score + cost
                key = (end, used_lines + 1)
                existing = next_states.get(key)
                if existing is None or new_score < existing[0]:
                    next_states[key] = (new_score, [*lines, line], [*widths, width])
        states = next_states

    final = states.get((len(words), line_count))
    if final is None:
        return None
    score, lines, widths = final
    score += balance_cost(widths, max_width)
    score += max(0, len(lines) - target_lines) * 7
    score += max(0, target_lines - len(lines)) * 5
    return lines, score


def line_cost(line: str, width: int, max_width: int, target_width: float) -> float:
    fill = width / max(1, max_width)
    cost = ((width - target_width) / max(1, max_width)) ** 2 * 90
    if len(line.split()) == 1:
        cost += 18
    if fill < 0.35:
        cost += (0.35 - fill) * 50
    return cost


def balance_cost(widths: list[int], max_width: int) -> float:
    if not widths:
        return 1_000_000.0
    fill_ratios = [width / max(1, max_width) for width in widths]
    average = sum(fill_ratios) / len(fill_ratios)
    return sum((ratio - average) ** 2 for ratio in fill_ratios) * 120


def split_long_word(
    draw: ImageDraw.ImageDraw,
    word: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if text_width(draw, word, font) <= max_width:
        return [word]

    chunks: list[str] = []
    chunk = ""
    for char in word:
        candidate = f"{chunk}{char}"
        if chunk and text_width(draw, candidate, font) > max_width:
            chunks.append(chunk)
            chunk = char
        else:
            chunk = candidate
    if chunk:
        chunks.append(chunk)
    return merge_tiny_word_chunks(draw, chunks, font, max_width)


def merge_tiny_word_chunks(
    draw: ImageDraw.ImageDraw,
    chunks: list[str],
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    merged = list(chunks)
    index = 1
    while index < len(merged):
        previous = merged[index - 1]
        current = merged[index]
        combined = f"{previous}{current}"
        if is_tiny_word_chunk(current) and text_width(draw, combined, font) <= max_width:
            merged[index - 1] = combined
            del merged[index]
            continue
        index += 1
    if len(merged) >= 2 and is_tiny_word_chunk(merged[-1]):
        combined = f"{merged[-2]}{merged[-1]}"
        if text_width(draw, combined, font) <= max_width:
            merged[-2] = combined
            merged.pop()
    return merged


def is_tiny_word_chunk(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) <= 2:
        return True
    if re.fullmatch(r"(?:['’]s|[.,!?;:]+)", stripped):
        return True
    return False


@lru_cache(maxsize=64)
def load_font(font_path: Path | None, size: int) -> ImageFont.ImageFont:
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            logger.debug("Loading font: %s size=%d", candidate, size)
            font = ImageFont.truetype(str(candidate), size=size)
            _FONT_REGISTRY[(str(candidate), size)] = font
            return font
    logger.warning("No TrueType font found; using Pillow default font at requested size=%d", size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    width, _height = text_size(draw, text, font)
    return width


def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    _width, height = text_size(draw, text, font)
    return height


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    font_key = registered_font_key(font)
    if font_key is not None:
        return cached_text_size(font_key, text)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def registered_font_key(font: ImageFont.ImageFont) -> tuple[str, int] | None:
    path = getattr(font, "path", None)
    size = getattr(font, "size", None)
    if path is None or size is None:
        return None
    key = (str(path), int(size))
    if key in _FONT_REGISTRY:
        return key
    return None


@lru_cache(maxsize=32768)
def cached_text_size(font_key: tuple[str, int], text: str) -> tuple[int, int]:
    font = _FONT_REGISTRY[font_key]
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
