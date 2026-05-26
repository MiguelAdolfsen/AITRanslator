from __future__ import annotations

from .detect_types import TextBlock
from .line_identity import block_id_for_block, source_hash


def group_text_blocks_for_translation(
    blocks: list[TextBlock],
    *,
    image_bgr=None,
) -> tuple[list[TextBlock], list[dict[str, object]]]:
    if len(blocks) < 2:
        return blocks, [group_debug_payload(index, [block]) for index, block in enumerate(blocks, start=1)]

    groups: list[list[TextBlock]] = []
    for block in sorted(blocks, key=block_sort_key):
        target_index = None
        for index, group in enumerate(groups):
            if any(should_group_blocks(block, existing, image_bgr=image_bgr) for existing in group):
                target_index = index
                break
        if target_index is None:
            groups.append([block])
        else:
            groups[target_index].append(block)

    grouped_blocks: list[TextBlock] = []
    grouping_report: list[dict[str, object]] = []
    for index, group in enumerate(groups, start=1):
        ordered_group = sort_blocks_in_reading_order(group)
        for member in ordered_group:
            member.metadata["translation_group_id"] = index
            member.metadata["translation_group_size"] = len(ordered_group)
        grouped_block = merge_grouped_text_blocks(index, ordered_group)
        grouped_blocks.append(grouped_block)
        grouping_report.append(group_debug_payload(index, ordered_group, grouped_block))

    grouped_blocks.sort(key=block_sort_key)
    return grouped_blocks, grouping_report


def should_group_blocks(a: TextBlock, b: TextBlock, *, image_bgr=None) -> bool:
    if a.detector != "ctd" or b.detector != "ctd":
        return False
    if bool(a.metadata.get("vertical")) != bool(b.metadata.get("vertical")):
        return False

    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    a_width = max(1, ax2 - ax1)
    b_width = max(1, bx2 - bx1)
    a_height = max(1, ay2 - ay1)
    b_height = max(1, by2 - by1)
    horizontal_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
    vertical_gap = max(0, max(ay1, by1) - min(ay2, by2))
    horizontal_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    vertical_overlap = max(0, min(ay2, by2) - max(ay1, by1))
    vertical_overlap_ratio = vertical_overlap / max(1, min(a_height, b_height))
    horizontal_overlap_ratio = horizontal_overlap / max(1, min(a_width, b_width))

    if bool(a.metadata.get("vertical")):
        max_gap = max(14, int(min(a_width, b_width) * 0.75))
        close_column = horizontal_gap <= max_gap and vertical_overlap_ratio >= 0.45
        tight_high_overlap_column = horizontal_gap <= 22 and vertical_overlap_ratio >= 0.65
        stacked_same_bubble_column = (
            image_bgr is not None
            and 50 <= vertical_gap <= 95
            and horizontal_overlap_ratio >= 0.75
            and boxes_share_white_region(image_bgr, a.box, b.box)
        )
        diagonal_same_bubble_column = (
            image_bgr is not None
            and 1 <= horizontal_gap <= 6
            and 0.02 <= vertical_overlap_ratio <= 0.20
            and boxes_share_white_region(image_bgr, a.box, b.box)
        )
        same_bubble_column = (
            image_bgr is not None
            and horizontal_gap <= 62
            and vertical_overlap_ratio >= 0.50
            and boxes_share_white_region(image_bgr, a.box, b.box)
        )
        return (
            close_column
            or tight_high_overlap_column
            or stacked_same_bubble_column
            or diagonal_same_bubble_column
            or same_bubble_column
        )

    max_gap = max(12, int(min(a_height, b_height) * 0.70))
    close_line = vertical_gap <= max_gap and horizontal_overlap_ratio >= 0.45
    tight_high_overlap_line = vertical_gap <= 18 and horizontal_overlap_ratio >= 0.65
    same_bubble_line = (
        image_bgr is not None
        and vertical_gap <= 45
        and horizontal_overlap_ratio >= 0.35
        and boxes_share_white_region(image_bgr, a.box, b.box)
    )
    return close_line or tight_high_overlap_line or same_bubble_line


def boxes_share_white_region(image_bgr, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    import cv2

    height, width = image_bgr.shape[:2]
    union = (
        max(0, min(a[0], b[0]) - 10),
        max(0, min(a[1], b[1]) - 10),
        min(width - 1, max(a[2], b[2]) + 10),
        min(height - 1, max(a[3], b[3]) + 10),
    )
    ux1, uy1, ux2, uy2 = union
    if ux2 <= ux1 or uy2 <= uy1:
        return False
    roi = image_bgr[uy1:uy2 + 1, ux1:ux2 + 1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    white = (gray >= 244).astype("uint8")
    num_labels, labels = cv2.connectedComponents(white, connectivity=4)
    if num_labels <= 1:
        return False

    first = nearest_white_component(labels, a, union)
    second = nearest_white_component(labels, b, union)
    return first > 0 and first == second


def nearest_white_component(labels, box: tuple[int, int, int, int], union: tuple[int, int, int, int]) -> int:
    import numpy as np

    ux1, uy1, ux2, uy2 = union
    x1, y1, x2, y2 = box
    lx1 = max(0, x1 - ux1)
    ly1 = max(0, y1 - uy1)
    lx2 = min(labels.shape[1] - 1, x2 - ux1)
    ly2 = min(labels.shape[0] - 1, y2 - uy1)
    if lx2 < lx1 or ly2 < ly1:
        return 0
    crop = labels[ly1:ly2 + 1, lx1:lx2 + 1]
    ys, xs = np.where(crop > 0)
    if len(xs) == 0:
        return 0
    center_x = (lx1 + lx2) / 2
    center_y = (ly1 + ly2) / 2
    distances = (xs + lx1 - center_x) ** 2 + (ys + ly1 - center_y) ** 2
    index = int(np.argmin(distances))
    return int(crop[ys[index], xs[index]])


def sort_blocks_in_reading_order(blocks: list[TextBlock]) -> list[TextBlock]:
    if not blocks:
        return []
    if bool(blocks[0].metadata.get("vertical")):
        return sorted(blocks, key=lambda block: (-box_center(block.box)[0], block.box[1]))
    return sorted(blocks, key=lambda block: (block.box[1], block.box[0]))


def merge_grouped_text_blocks(group_id: int, blocks: list[TextBlock]) -> TextBlock:
    if len(blocks) == 1:
        block = blocks[0]
        metadata = dict(block.metadata)
        metadata["translation_group_id"] = group_id
        metadata["translation_group_size"] = 1
        return TextBlock(
            text=block.text,
            box=block.box,
            confidence=block.confidence,
            detector=block.detector,
            metadata=metadata,
        )

    boxes = [block.box for block in blocks]
    text = "".join(block.text for block in blocks)
    metadata = {
        "translation_group_id": group_id,
        "translation_group_size": len(blocks),
        "grouped_from": [
            {
                "box": block.box,
                "source_text": block.text,
                "block_id": block_id_for_block(block),
                "source_hash": source_hash(block.text),
                "confidence": block.confidence,
                "metadata": block.metadata,
            }
            for block in blocks
        ],
        "vertical": bool(blocks[0].metadata.get("vertical")),
    }
    return TextBlock(
        text=text,
        box=union_box(boxes),
        confidence=sum(block.confidence for block in blocks) / len(blocks),
        detector=blocks[0].detector,
        metadata=metadata,
    )


def group_debug_payload(
    group_id: int,
    members: list[TextBlock],
    grouped_block: TextBlock | None = None,
) -> dict[str, object]:
    rendered = grouped_block or members[0]
    return {
        "group_id": group_id,
        "box": rendered.box,
        "source_text": rendered.text,
        "block_id": block_id_for_block(rendered),
        "source_hash": source_hash(rendered.text),
        "members": [
            {
                "box": member.box,
                "source_text": member.text,
                "block_id": block_id_for_block(member),
                "source_hash": source_hash(member.text),
                "confidence": member.confidence,
                "detector": member.detector,
            }
            for member in members
        ],
    }


def build_page_order_report(
    blocks: list[TextBlock],
    *,
    width: int,
    height: int,
    chapter_context_before: str | None = None,
) -> list[dict[str, object]]:
    _ = width, height
    ordered_rows: list[list[TextBlock]] = []
    for block in sorted(blocks, key=lambda item: box_center(item.box)[1]):
        center_y = box_center(block.box)[1]
        target_row = None
        for row in ordered_rows:
            row_center = sum(box_center(item.box)[1] for item in row) / len(row)
            row_height = max(item.box[3] - item.box[1] for item in row)
            if abs(center_y - row_center) <= max(70, row_height * 0.65):
                target_row = row
                break
        if target_row is None:
            ordered_rows.append([block])
        else:
            target_row.append(block)

    ordered_blocks: list[TextBlock] = []
    for row in ordered_rows:
        ordered_blocks.extend(sorted(row, key=lambda item: (-box_center(item.box)[0], box_center(item.box)[1])))

    report = []
    for index, block in enumerate(ordered_blocks, start=1):
        report.append(
            {
                "page_order": index,
                "box": block.box,
                "source_text": block.text,
                "block_id": block_id_for_block(block),
                "source_hash": source_hash(block.text),
                "detector": block.detector,
            }
        )
    for index, item in enumerate(report):
        if index > 0:
            item["context_before"] = report[index - 1]["source_text"]
        elif chapter_context_before:
            item["context_before"] = chapter_context_before
        if index + 1 < len(report):
            item["context_after"] = report[index + 1]["source_text"]
        before_window = [
            str(report[window_index]["source_text"])
            for window_index in range(max(0, index - 2), index)
        ]
        if index == 0 and chapter_context_before:
            before_window.insert(0, chapter_context_before)
        after_window = [
            str(report[window_index]["source_text"])
            for window_index in range(index + 1, min(len(report), index + 3))
        ]
        item["context_before_window"] = before_window
        item["context_after_window"] = after_window
    return report


def page_order_for_block(block, page_order_report: list[dict[str, object]]) -> int | None:
    line_id = str(getattr(block, "metadata", {}).get("line_id") or "")
    for item in page_order_report:
        if line_id and str(item.get("line_id", "")) == line_id:
            return int(item["page_order"])
        if tuple(item["box"]) == tuple(block.box) and item["source_text"] == block.text:
            return int(item["page_order"])
    return None


def context_for_block(block, page_order_report: list[dict[str, object]]) -> dict[str, object]:
    line_id = str(getattr(block, "metadata", {}).get("line_id") or "")
    for item in page_order_report:
        if line_id and str(item.get("line_id", "")) != line_id:
            continue
        if tuple(item["box"]) == tuple(block.box) and item["source_text"] == block.text:
            before = item.get("context_before")
            after = item.get("context_after")
            return {
                "line_id": item.get("line_id", line_id),
                "block_id": item.get("block_id", getattr(block, "metadata", {}).get("block_id", "")),
                "source_hash": item.get("source_hash", source_hash(block.text)),
                "page_order": item.get("page_order"),
                "context_before": before,
                "context_after": after,
                "context_before_window": item.get("context_before_window") or ([] if not before else [before]),
                "context_after_window": item.get("context_after_window") or ([] if not after else [after]),
                "context_available": bool(before or after),
                "context_used": False,
                "translation_mode": "single_block",
            }
        if line_id:
            break
    return {"context_available": False, "context_used": False, "translation_mode": "single_block"}


def block_sort_key(block: TextBlock) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = block.box
    return y1, x1, y2, x2


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
