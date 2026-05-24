from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import Counter

from .config import PipelineConfig
from .grouping import page_order_for_block
from .text_filter import block_to_debug_dict, suspected_bad_translation
from .vision_artifact import vision_artifact_to_debug_dict

logger = logging.getLogger(__name__)


def write_debug_image(image_bgr, blocks, skipped_blocks, fallback_blocks, render_layouts, output_path: Path) -> None:
    import cv2

    debug = image_bgr.copy()
    fallback_boxes = {tuple(block["box"]) for block in fallback_blocks}
    for block, layout in zip(blocks, render_layouts):
        x1, y1, x2, y2 = block.box
        color = (0, 165, 255) if tuple(block.box) in fallback_boxes else debug_color_for_detector(
            getattr(block, "detector", "unknown")
        )
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, thickness=2)
        rx1, ry1, rx2, ry2 = layout.render_box
        cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), (255, 0, 0), thickness=1)
        if layout.bubble_box is not None:
            bx1, by1, bx2, by2 = layout.bubble_box
            cv2.rectangle(debug, (bx1, by1), (bx2, by2), (255, 180, 0), thickness=1)
        cv2.putText(
            debug,
            f"{getattr(block, 'detector', '?')}:{block.confidence:.0f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    for skipped in skipped_blocks:
        x1, y1, x2, y2 = skipped["box"]
        color = (0, 0, 255)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, thickness=1)
        cv2.putText(
            debug,
            f"skip:{skipped.get('reason', '?')}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    debug_path = output_path.with_name(f"{output_path.stem}.debug{output_path.suffix}")
    if cv2.imwrite(str(debug_path), debug):
        logger.info("Debug image written: %s", debug_path)
    else:
        logger.error("Could not write debug image: %s", debug_path)


def debug_color_for_detector(detector: str) -> tuple[int, int, int]:
    if detector == "ctd":
        return (0, 180, 0)
    if detector == "visual":
        return (255, 0, 0)
    if detector == "tesseract":
        return (0, 0, 255)
    return (0, 180, 180)


def block_to_render_debug_dict(
    block,
    layout,
    fit,
    *,
    image_width: int,
    image_height: int,
    erase_padding: int,
    translated_text: str,
    page_order: int | None,
    translation_context: dict[str, object] | None,
) -> dict[str, object]:
    payload = block_to_debug_dict(block, status="rendered", translated_text=translated_text)
    if page_order is not None:
        payload["page_order"] = page_order
    payload["erase_box"] = pad_box_for_debug(block.box, image_width, image_height, erase_padding)
    payload["render_box"] = layout.render_box
    if fit is not None:
        payload["render_fit"] = render_fit_to_debug_dict(fit)
        if translation_context is not None and translation_context.get("compacted_for_render") is True:
            payload["render_fit"]["compacted_for_render"] = True
    if layout.bubble_box is not None:
        payload["bubble_box"] = layout.bubble_box
    warnings = render_layout_warnings(layout, fit, image_width=image_width, image_height=image_height)
    if warnings:
        payload["layout_warnings"] = warnings
    if translation_context is not None:
        payload.update(translation_context)
    return payload


def render_fit_to_debug_dict(fit) -> dict[str, object]:
    return {
        "font_size": int(getattr(fit, "font_size", 0)),
        "line_count": int(getattr(fit, "line_count", 0)),
        "fit_status": str(getattr(fit, "fit_status", "unknown")),
        "clipped": bool(getattr(fit, "clipped", False)),
        "widest_line": int(getattr(fit, "widest_line", 0)),
        "total_height": int(getattr(fit, "total_height", 0)),
        "usable_width": int(getattr(fit, "usable_width", 0)),
        "usable_height": int(getattr(fit, "usable_height", 0)),
        "attempted_font_size": int(getattr(fit, "attempted_font_size", 0)),
        "line_height": int(getattr(fit, "line_height", 0)),
        "overflow_width": int(getattr(fit, "overflow_width", 0)),
        "overflow_height": int(getattr(fit, "overflow_height", 0)),
        "lines": list(getattr(fit, "lines", ())),
        "wrap_score": float(getattr(fit, "wrap_score", 0.0)),
        "wrap_warnings": list(getattr(fit, "wrap_warnings", ())),
        "split_word_count": int(getattr(fit, "split_word_count", 0)),
        "orphan_line_count": int(getattr(fit, "orphan_line_count", 0)),
        "candidate_font_sizes": list(getattr(fit, "candidate_font_sizes", ())),
        "compacted_for_render": bool(getattr(fit, "compacted_for_render", False)),
    }


def render_layout_warnings(layout, fit, *, image_width: int, image_height: int) -> list[str]:
    warnings: list[str] = []
    if fit is not None:
        if bool(getattr(fit, "clipped", False)):
            warnings.append("text_clipped")
        if int(getattr(fit, "font_size", 99)) <= 7:
            warnings.append("tiny_font")
        if int(getattr(fit, "line_count", 0)) >= 5:
            warnings.append("excessive_line_count")
        if int(getattr(fit, "overflow_width", 0)) > 0:
            warnings.append("overflow_width")
        if int(getattr(fit, "overflow_height", 0)) > 0:
            warnings.append("overflow_height")
        wrap_warnings = set(getattr(fit, "wrap_warnings", ()) or ())
        if "bad_wrap" in wrap_warnings:
            warnings.append("bad_wrap")
        if int(getattr(fit, "orphan_line_count", 0)) > 0:
            warnings.append("orphan_line")
        if int(getattr(fit, "split_word_count", 0)) > 0:
            warnings.append("split_word")
        if int(getattr(fit, "font_size", 99)) <= 8 and int(getattr(fit, "line_count", 0)) >= 4:
            warnings.append("cramped_fit")
        if bool(getattr(fit, "clipped", False)) or int(getattr(fit, "overflow_width", 0)) > 0 or int(getattr(fit, "overflow_height", 0)) > 0:
            warnings.append("text_outside_render_box")
    render_box = tuple(getattr(layout, "render_box", (0, 0, 0, 0)))
    bubble_box = getattr(layout, "bubble_box", None)
    if bubble_box is not None and not box_contains(tuple(bubble_box), render_box):
        warnings.append("render_box_outside_bubble")
    if not box_contains((0, 0, image_width - 1, image_height - 1), render_box):
        warnings.append("render_box_outside_page")
    if touches_image_edge(render_box, image_width, image_height):
        warnings.append("render_box_touches_page_edge")
    return list(dict.fromkeys(warnings))


def box_contains(container: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return container[0] <= inner[0] and container[1] <= inner[1] and container[2] >= inner[2] and container[3] >= inner[3]


def touches_image_edge(box: tuple[int, int, int, int], image_width: int, image_height: int) -> bool:
    return box[0] <= 0 or box[1] <= 0 or box[2] >= image_width - 1 or box[3] >= image_height - 1


def pad_box_for_debug(
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


def write_debug_report(
    image_path: Path,
    output_path: Path,
    config: PipelineConfig,
    raw_blocks,
    blocks,
    translations: dict[str, str],
    skipped_blocks: list[dict[str, object]],
    fallback_blocks: list[dict[str, object]],
    grouping_report: list[dict[str, object]],
    render_layouts,
    render_fits,
    image_width: int,
    image_height: int,
    erase_padding: int,
    page_order_report: list[dict[str, object]],
    translation_contexts: dict[str, dict[str, object]],
    vision_artifact=None,
    vision_facts_artifact=None,
) -> None:
    report_path = output_path.with_name(f"{output_path.stem}.ocr.json")
    kept_blocks = [
        block_to_render_debug_dict(
            block,
            layout,
            fit,
            image_width=image_width,
            image_height=image_height,
            erase_padding=erase_padding,
            translated_text=translations.get(block.text, ""),
            page_order=page_order_for_block(block, page_order_report),
            translation_context=translation_contexts.get(block.text),
        )
        for block, layout, fit in zip(blocks, render_layouts, render_fits)
    ]
    add_render_collision_warnings(kept_blocks)
    page_summary = build_page_debug_summary(blocks, translations, skipped_blocks, fallback_blocks, translation_contexts)
    page_summary["layout_warning_blocks"] = sum(1 for block in kept_blocks if block.get("layout_warnings"))
    page_summary["vision_attempted"] = sum(1 for block in kept_blocks if block.get("vision_attempted") is True)
    page_summary["vision_accepted"] = sum(1 for block in kept_blocks if block.get("vision_accepted") is True)
    page_summary["vision_rejected"] = sum(1 for block in kept_blocks if block.get("vision_attempted") is True and block.get("vision_accepted") is not True)
    page_summary.update(build_vision_debug_summary(kept_blocks, vision_artifact))
    page_summary.update(build_vision_facts_debug_summary(kept_blocks, vision_facts_artifact))
    report = {
        "source_image": str(image_path),
        "output_image": str(output_path),
        "detector": config.detector,
        "ocr_engine": config.ocr_engine,
        "translator": config.translator,
        "qwen_mode": config.qwen_mode,
        "vision_enabled": config.vision_enabled,
        "vision_facts_enabled": config.vision_facts_enabled,
        "vision_mode": config.vision_mode,
        "vision_trigger": config.vision_trigger,
        "page_summary": page_summary,
        "raw_blocks": [
            block_to_debug_dict(block)
            for block in raw_blocks
        ],
        "kept_blocks": kept_blocks,
        "translation_groups": grouping_report,
        "page_order": page_order_report,
        "skipped_blocks": skipped_blocks,
        "fallback_blocks": fallback_blocks,
    }
    vision_debug = vision_artifact_to_debug_dict(vision_artifact)
    if vision_debug is not None:
        report["vision"] = vision_debug
    vision_facts_debug = vision_artifact_to_debug_dict(vision_facts_artifact)
    if vision_facts_debug is not None:
        report["vision_facts"] = vision_facts_debug
    try:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.exception("Could not write OCR debug report: %s", report_path)
        return
    logger.info("OCR debug report written: %s", report_path)
    logger.info("Page debug summary: %s", page_summary)


def build_page_debug_summary(
    blocks,
    translations: dict[str, str],
    skipped_blocks: list[dict[str, object]],
    fallback_blocks: list[dict[str, object]],
    translation_contexts: dict[str, dict[str, object]],
) -> dict[str, object]:
    contexts = [translation_contexts.get(block.text, {}) for block in blocks]
    return {
        "kept_blocks": len(blocks),
        "skipped_blocks": len(skipped_blocks),
        "fallback_blocks": len(fallback_blocks),
        "ellipsis_outputs": sum(1 for block in blocks if translations.get(block.text, "").strip() == "..."),
        "qwen_used": sum(1 for context in contexts if context.get("qwen_used") is True),
        "qwen_rejected": sum(1 for context in contexts if context.get("qwen_rejected") is True),
        "qwen_repairs_attempted": sum(1 for context in contexts if context.get("qwen_repair_used") is True),
        "qwen_repairs_accepted": sum(1 for context in contexts if context.get("qwen_repair_accepted") is True),
        "qwen_fallback_attempted": sum(1 for context in contexts if context.get("qwen_fallback_attempted") is True),
        "qwen_fallback_accepted": sum(1 for context in contexts if context.get("qwen_fallback_accepted") is True),
        "qwen_critic_attempted": sum(1 for context in contexts if context.get("qwen_critic_attempted") is True),
        "qwen_critic_flagged": sum(1 for context in contexts if context.get("qwen_critic_flagged") is True),
        "qwen_critic_issue_types": dict(count_split_values(issue for context in contexts for issue in context.get("qwen_critic_issues", []))),
        "suspected_bad_translations": sum(
            1
            for block in blocks
            if suspected_bad_translation(translations.get(block.text, ""))
        ),
    }


def add_render_collision_warnings(kept_blocks: list[dict[str, object]]) -> None:
    for index, block in enumerate(kept_blocks):
        render_box = as_box(block.get("render_box"))
        if render_box is None:
            continue
        collisions: list[dict[str, object]] = []
        for other_index, other in enumerate(kept_blocks):
            if other_index == index:
                continue
            other_render = as_box(other.get("render_box"))
            if other_render is None or not boxes_intersect(render_box, other_render):
                continue
            collisions.append(
                {
                    "page_order": other.get("page_order"),
                    "source_text": other.get("source_text"),
                    "overlap_area": box_intersection_area(render_box, other_render),
                }
            )
        if not collisions:
            continue
        block["render_collisions"] = collisions
        warnings = list(block.get("layout_warnings") or [])
        if "render_box_collision" not in warnings:
            warnings.append("render_box_collision")
        block["layout_warnings"] = warnings


def as_box(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def box_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if not boxes_intersect(a, b):
        return 0
    return (min(a[2], b[2]) - max(a[0], b[0])) * (min(a[3], b[3]) - max(a[1], b[1]))


def build_vision_debug_summary(kept_blocks: list[dict[str, object]], vision_artifact) -> dict[str, object]:
    attempted = [block for block in kept_blocks if block.get("vision_attempted") is True]
    return {
        "vision_json_repair_attempted": sum(1 for block in attempted if block.get("vision_json_repair_attempted") is True),
        "vision_json_repair_used": sum(1 for block in attempted if block.get("vision_json_repair_used") is True),
        "vision_reject_reasons": dict(count_values(block.get("vision_reject_reason") for block in attempted)),
        "vision_trigger_reasons": dict(count_split_values(block.get("vision_trigger_reason") for block in attempted)),
        "layout_warning_types": dict(count_split_values(warning for block in kept_blocks for warning in block.get("layout_warnings", []))),
        "vision_artifact_path": str(getattr(vision_artifact, "image_path", "")) if vision_artifact is not None else "",
    }


def build_vision_facts_debug_summary(kept_blocks: list[dict[str, object]], vision_facts_artifact) -> dict[str, object]:
    attempted = [block for block in kept_blocks if block.get("vision_facts_attempted") is True]
    return {
        "vision_facts_attempted": len(attempted),
        "vision_facts_accepted": sum(1 for block in attempted if block.get("vision_facts_accepted") is True),
        "vision_facts_rejected": sum(1 for block in attempted if block.get("vision_facts_accepted") is not True),
        "vision_facts_json_repair_attempted": sum(1 for block in attempted if block.get("vision_facts_json_repair_attempted") is True),
        "vision_facts_json_repair_used": sum(1 for block in attempted if block.get("vision_facts_json_repair_used") is True),
        "vision_facts_reject_reasons": dict(count_values(block.get("vision_facts_reject_reason") for block in attempted)),
        "vision_facts_artifact_path": str(getattr(vision_facts_artifact, "image_path", "")) if vision_facts_artifact is not None else "",
    }


def count_values(values) -> Counter:
    return Counter(str(value) for value in values if value)


def count_split_values(values) -> Counter:
    counter: Counter = Counter()
    for value in values:
        if isinstance(value, str):
            parts = [part for part in value.split(",") if part]
        else:
            parts = [str(value)] if value else []
        counter.update(parts)
    return counter
