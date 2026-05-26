from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from normalize_text import normalized_region_text, text_similarity


EXTRACTABLE_KINDS = {"dialogue", "narration", "thought", "sign", "sfx"}


@dataclass(frozen=True)
class RegionMatch:
    page_id: str
    region_id: str
    pred_region_id: str | None
    matched: bool
    box_iou: float
    text_similarity: float
    cer: float
    orientation_ok: bool
    gt_text: str
    pred_text: str
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "region_id": self.region_id,
            "pred_region_id": self.pred_region_id,
            "matched": self.matched,
            "box_iou": round(self.box_iou, 6),
            "text_similarity": round(self.text_similarity, 6),
            "cer": round(self.cer, 6),
            "orientation_ok": self.orientation_ok,
            "gt_text": self.gt_text,
            "pred_text": self.pred_text,
            "violations": list(self.violations),
        }


def box_area(box: list[float] | tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: list[float] | tuple[float, float, float, float], b: list[float] | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = box_area(a) + box_area(b) - intersection
    return intersection / max(1.0, union)


def box_contains_ratio(outer: list[float], inner: list[float]) -> float:
    ox1, oy1, ox2, oy2 = [float(v) for v in outer]
    ix1, iy1, ix2, iy2 = [float(v) for v in inner]
    x1 = max(ox1, ix1)
    y1 = max(oy1, iy1)
    x2 = min(ox2, ix2)
    y2 = min(oy2, iy2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return ((x2 - x1) * (y2 - y1)) / max(1.0, box_area(inner))


def is_extractable(region: dict[str, Any]) -> bool:
    return bool(region.get("should_extract")) and str(region.get("kind", "unknown")) in EXTRACTABLE_KINDS


def match_regions(
    page_id: str,
    gt_regions: list[dict[str, Any]],
    pred_regions: list[dict[str, Any]],
    *,
    min_iou: float = 0.35,
) -> tuple[list[RegionMatch], list[dict[str, Any]], list[dict[str, Any]]]:
    extractable = [region for region in gt_regions if is_extractable(region)]
    candidates: list[tuple[float, float, float, str, str, dict[str, Any], dict[str, Any]]] = []
    for gt in extractable:
        gt_text = normalized_region_text(gt, "normalized_source_text", "source_text")
        for pred in pred_regions:
            iou = box_iou(gt["box"], pred["box"])
            contains = max(box_contains_ratio(gt["box"], pred["box"]), box_contains_ratio(pred["box"], gt["box"]))
            if iou < min_iou and contains < 0.78:
                continue
            pred_text = normalized_region_text(pred, "normalized_ocr_text", "ocr_text")
            sim = text_similarity(gt_text, pred_text)
            score = (0.75 * iou) + (0.20 * contains) + (0.05 * sim)
            candidates.append((score, iou, sim, str(gt["region_id"]), str(pred["pred_region_id"]), gt, pred))

    candidates.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3], row[4]))
    matched_gt: set[str] = set()
    matched_pred: set[str] = set()
    matches: list[RegionMatch] = []
    for _score, iou, sim, gt_id, pred_id, gt, pred in candidates:
        if gt_id in matched_gt or pred_id in matched_pred:
            continue
        matched_gt.add(gt_id)
        matched_pred.add(pred_id)
        gt_text = normalized_region_text(gt, "normalized_source_text", "source_text")
        pred_text = normalized_region_text(pred, "normalized_ocr_text", "ocr_text")
        from normalize_text import character_error_rate

        cer = character_error_rate(gt_text, pred_text, cap=1.0)
        orientation_ok = str(gt.get("orientation", "unknown")) in {"unknown", str(pred.get("orientation", "unknown"))}
        violations: list[str] = []
        if cer >= 0.50:
            violations.append("severe_ocr_error")
        if iou < 0.45:
            violations.append("bad_crop")
        if not orientation_ok:
            violations.append("orientation_error")
        matches.append(
            RegionMatch(
                page_id=page_id,
                region_id=gt_id,
                pred_region_id=pred_id,
                matched=True,
                box_iou=iou,
                text_similarity=sim,
                cer=cer,
                orientation_ok=orientation_ok,
                gt_text=gt_text,
                pred_text=pred_text,
                violations=tuple(violations),
            )
        )

    for gt in extractable:
        gt_id = str(gt["region_id"])
        if gt_id in matched_gt:
            continue
        matches.append(
            RegionMatch(
                page_id=page_id,
                region_id=gt_id,
                pred_region_id=None,
                matched=False,
                box_iou=0.0,
                text_similarity=0.0,
                cer=1.0,
                orientation_ok=False,
                gt_text=normalized_region_text(gt, "normalized_source_text", "source_text"),
                pred_text="",
                violations=("missed_region",),
            )
        )

    unmatched_gt = [gt for gt in extractable if str(gt["region_id"]) not in matched_gt]
    unmatched_pred = [pred for pred in pred_regions if str(pred.get("pred_region_id")) not in matched_pred]
    matches.sort(key=lambda item: (item.page_id, item.region_id, item.pred_region_id or ""))
    return matches, unmatched_gt, unmatched_pred


def duplicate_prediction_count(pred_regions: list[dict[str, Any]]) -> int:
    duplicates = 0
    for index, pred in enumerate(pred_regions):
        pred_text = normalized_region_text(pred, "normalized_ocr_text", "ocr_text")
        for other in pred_regions[:index]:
            if box_iou(pred["box"], other["box"]) < 0.70:
                continue
            other_text = normalized_region_text(other, "normalized_ocr_text", "ocr_text")
            if pred_text == other_text or text_similarity(pred_text, other_text) >= 0.85:
                duplicates += 1
                break
    return duplicates


def reading_order_error_count(matches: list[RegionMatch], gt_regions: list[dict[str, Any]], pred_regions: list[dict[str, Any]]) -> int:
    return len(reading_order_error_details(matches, gt_regions, pred_regions))


def reading_order_error_details(matches: list[RegionMatch], gt_regions: list[dict[str, Any]], pred_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gt_by_id = {str(region["region_id"]): region for region in gt_regions}
    pred_by_id = {str(region.get("pred_region_id")): region for region in pred_regions}
    comparable: list[tuple[int, int, str, str]] = []
    for match in matches:
        if not match.matched or not match.pred_region_id:
            continue
        gt_order = gt_by_id.get(match.region_id, {}).get("page_order")
        pred_order = pred_by_id.get(match.pred_region_id, {}).get("page_order")
        if gt_order is None or pred_order is None:
            continue
        comparable.append((int(gt_order), int(pred_order), match.region_id, match.pred_region_id))
    details: list[dict[str, Any]] = []
    for i, left in enumerate(comparable):
        for right in comparable[i + 1:]:
            gt_delta = left[0] - right[0]
            pred_delta = left[1] - right[1]
            if gt_delta and pred_delta and (gt_delta > 0) != (pred_delta > 0):
                details.append(
                    {
                        "type": "reading_order_inversion",
                        "left_region_id": left[2],
                        "left_pred_region_id": left[3],
                        "left_gt_order": left[0],
                        "left_pred_order": left[1],
                        "right_region_id": right[2],
                        "right_pred_region_id": right[3],
                        "right_gt_order": right[0],
                        "right_pred_order": right[1],
                    }
                )
    return details


def group_error_counts(
    matches: list[RegionMatch],
    gt_regions: list[dict[str, Any]],
    pred_regions: list[dict[str, Any]],
    pred_groups: list[dict[str, Any]],
) -> tuple[int, int]:
    details = group_error_details(matches, gt_regions, pred_regions, pred_groups)
    overmerge = sum(1 for item in details if item["type"] == "overmerge")
    undermerge = sum(1 for item in details if item["type"] == "undermerge")
    return overmerge, undermerge


def group_error_details(
    matches: list[RegionMatch],
    gt_regions: list[dict[str, Any]],
    pred_regions: list[dict[str, Any]],
    pred_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gt_by_region = {str(region["region_id"]): region for region in gt_regions}
    pred_by_region = {str(region.get("pred_region_id")): region for region in pred_regions}
    pred_to_gt_group: dict[str, str] = {}
    gt_to_pred_group_ids: dict[str, set[str]] = {}
    gt_to_region_ids: dict[str, set[str]] = {}
    for match in matches:
        if not match.matched or not match.pred_region_id:
            continue
        gt_group = gt_by_region.get(match.region_id, {}).get("group_id")
        pred_group = pred_by_region.get(match.pred_region_id, {}).get("group_id")
        if gt_group is None or pred_group is None:
            continue
        pred_to_gt_group[str(match.pred_region_id)] = str(gt_group)
        gt_to_pred_group_ids.setdefault(str(gt_group), set()).add(str(pred_group))
        gt_to_region_ids.setdefault(str(gt_group), set()).add(match.region_id)

    details: list[dict[str, Any]] = []
    for group in pred_groups:
        members = [str(item) for item in group.get("member_pred_region_ids", [])]
        gt_groups = {pred_to_gt_group[member] for member in members if member in pred_to_gt_group}
        if len(gt_groups) > 1:
            details.append(
                {
                    "type": "overmerge",
                    "pred_group_id": group.get("pred_group_id"),
                    "member_pred_region_ids": members,
                    "gt_group_ids": sorted(gt_groups),
                }
            )
    for gt_group, pred_group_ids in gt_to_pred_group_ids.items():
        if len(pred_group_ids) > 1:
            details.append(
                {
                    "type": "undermerge",
                    "gt_group_id": gt_group,
                    "region_ids": sorted(gt_to_region_ids.get(gt_group, set())),
                    "pred_group_ids": sorted(pred_group_ids),
                }
            )
    return details
