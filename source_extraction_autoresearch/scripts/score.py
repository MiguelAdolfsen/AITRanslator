from __future__ import annotations

import math
from statistics import mean
from typing import Any

from match_regions import (
    box_contains_ratio,
    box_iou,
    duplicate_prediction_count,
    group_error_details,
    group_error_counts,
    is_extractable,
    match_regions,
    reading_order_error_details,
    reading_order_error_count,
)
from normalize_text import contains_japanese, is_empty_or_punctuation, japanese_ratio, normalized_region_text


TRANSLATABLE_KINDS = {"dialogue", "narration", "thought", "sign", "sfx"}


def safe_rate(count: float, total: float) -> float:
    return float(count) / max(1.0, float(total))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[int(index)])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def score_page(
    label: dict[str, Any],
    prediction: dict[str, Any],
    ignore: dict[str, Any] | None = None,
) -> dict[str, Any]:
    page_id = str(label["page_id"])
    gt_regions = list(label.get("text_regions", []))
    pred_regions = list(prediction.get("regions", []))
    pred_groups = list(prediction.get("groups", []))
    matches, missed, unmatched_pred = match_regions(page_id, gt_regions, pred_regions)
    ignore_boxes = [item["box"] for item in (ignore or {}).get("ignore_regions", []) if "box" in item]

    false_positives = len(unmatched_pred)
    destructive_false_positives = 0
    harmless_false_positives = 0
    non_japanese_noise = 0
    false_positive_details: list[dict[str, Any]] = []
    for pred in unmatched_pred:
        pred_text = normalized_region_text(pred, "normalized_ocr_text", "ocr_text")
        in_ignore = any(box_contains_ratio(ignore_box, pred["box"]) >= 0.70 for ignore_box in ignore_boxes)
        nearest_gt = None
        nearest_iou = 0.0
        overlaps_non_extractable = False
        for gt in gt_regions:
            current_iou = box_iou(gt["box"], pred["box"])
            if current_iou > nearest_iou:
                nearest_iou = current_iou
                nearest_gt = gt
            if (
                not bool(gt.get("should_extract"))
                and current_iou >= 0.20
                and str(gt.get("kind", "unknown")) in {"metadata", "credit", "page_number", "noise", "unknown"}
            ):
                overlaps_non_extractable = True
        plausible_text = bool(pred_text) and not is_empty_or_punctuation(pred_text)
        if not in_ignore and (overlaps_non_extractable or plausible_text):
            destructive_false_positives += 1
            false_positive_type = "destructive_false_positive"
        else:
            harmless_false_positives += 1
            false_positive_type = "harmless_false_positive"
        if plausible_text and not contains_japanese(pred_text) and japanese_ratio(pred_text) < 0.20:
            non_japanese_noise += 1
        false_positive_details.append(
            {
                "page_id": page_id,
                "type": false_positive_type,
                "pred_region_id": pred.get("pred_region_id"),
                "box": pred.get("box"),
                "pred_text": pred_text,
                "orientation": pred.get("orientation"),
                "group_id": pred.get("group_id"),
                "page_order": pred.get("page_order"),
                "in_ignore_region": in_ignore,
                "overlaps_non_extractable": overlaps_non_extractable,
                "plausible_text": plausible_text,
                "nearest_region_id": nearest_gt.get("region_id") if nearest_gt else None,
                "nearest_kind": nearest_gt.get("kind") if nearest_gt else None,
                "nearest_should_extract": nearest_gt.get("should_extract") if nearest_gt else None,
                "nearest_iou": round(nearest_iou, 6),
            }
        )

    matched = [item for item in matches if item.matched]
    gt_extractable = [region for region in gt_regions if is_extractable(region)]
    severe_ocr = sum(1 for item in matched if item.cer >= 0.50)
    wrong_text = severe_ocr
    empty_ocr = sum(1 for item in matched if is_empty_or_punctuation(item.pred_text))
    bad_crop = sum(1 for item in matched if item.box_iou < 0.45)
    orientation_errors = sum(1 for item in matched if not item.orientation_ok)
    duplicate_regions = duplicate_prediction_count(pred_regions)
    overmerge, undermerge = group_error_counts(matches, gt_regions, pred_regions, pred_groups)
    order_errors = reading_order_error_count(matches, gt_regions, pred_regions)
    group_details = group_error_details(matches, gt_regions, pred_regions, pred_groups)
    order_details = reading_order_error_details(matches, gt_regions, pred_regions)
    total_ms = float((prediction.get("timing") or {}).get("total_ms", 0.0) or 0.0)

    page_metrics = {
        "page_id": page_id,
        "gt_regions_total": len(gt_regions),
        "gt_extractable_regions": len(gt_extractable),
        "predicted_regions": len(pred_regions),
        "matched_regions": len(matched),
        "missed_dialogue_region_count": len(missed),
        "missed_regions": len(missed),
        "false_positive_region_count": false_positives,
        "false_positives": false_positives,
        "destructive_false_positive_count": destructive_false_positives,
        "harmless_false_positive_count": harmless_false_positives,
        "mean_box_iou": mean([item.box_iou for item in matched]) if matched else 0.0,
        "mean_ocr_cer": mean([item.cer for item in matched]) if matched else (1.0 if gt_extractable else 0.0),
        "severe_ocr_error_count": severe_ocr,
        "wrong_text_region_match_count": wrong_text,
        "empty_ocr_count": empty_ocr,
        "overmerge_count": overmerge,
        "undermerge_count": undermerge,
        "reading_order_error_count": order_errors,
        "orientation_error_count": orientation_errors,
        "bad_crop_count": bad_crop,
        "duplicate_region_count": duplicate_regions,
        "non_japanese_noise_count": non_japanese_noise,
        "total_ms": total_ms,
        "matches": [item.as_dict() for item in matches],
        "false_positive_details": false_positive_details,
        "group_failure_details": [{"page_id": page_id, **item} for item in group_details],
        "reading_order_details": [{"page_id": page_id, **item} for item in order_details],
    }
    page_metrics["page_score_breakdown"] = score_breakdown_from_metrics(page_metrics)
    page_metrics["source_extraction_quality_score"] = quality_score_from_metrics(page_metrics)
    page_metrics["timing_score_component"] = timing_score_component(page_metrics)
    page_metrics["source_extraction_score"] = score_from_metrics(page_metrics)
    return page_metrics


def score_breakdown_from_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    gt = float(metrics.get("gt_extractable_regions", 0) or 0)
    pred = float(metrics.get("predicted_regions", 0) or 0)
    matched = float(metrics.get("matched_regions", 0) or 0)
    groups = max(1.0, pred)
    order_pairs = max(1.0, matched * max(0.0, matched - 1.0) / 2.0)
    return {
        "missed": round(12000 * safe_rate(metrics.get("missed_dialogue_region_count", 0), gt), 6),
        "destructive_false_positive": round(10000 * safe_rate(metrics.get("destructive_false_positive_count", 0), pred), 6),
        "ocr": round(
            8000 * safe_rate(metrics.get("wrong_text_region_match_count", 0), matched)
            + 7000 * float(metrics.get("mean_ocr_cer", 0.0) or 0.0)
            + 6500 * safe_rate(metrics.get("severe_ocr_error_count", 0), matched)
            + 1800 * safe_rate(metrics.get("empty_ocr_count", 0), matched),
            6,
        ),
        "grouping": round(
            6000 * safe_rate(metrics.get("overmerge_count", 0), groups)
            + 6000 * safe_rate(metrics.get("undermerge_count", 0), max(1.0, gt)),
            6,
        ),
        "order": round(5000 * safe_rate(metrics.get("reading_order_error_count", 0), order_pairs), 6),
        "crop_duplicate_noise": round(
            4000 * safe_rate(metrics.get("orientation_error_count", 0), matched)
            + 3000 * safe_rate(metrics.get("bad_crop_count", 0), matched)
            + 2500 * safe_rate(metrics.get("duplicate_region_count", 0), pred)
            + 1500 * safe_rate(metrics.get("non_japanese_noise_count", 0), pred)
            + 500 * safe_rate(metrics.get("harmless_false_positive_count", 0), pred),
            6,
        ),
        "timing_tiebreaker": timing_score_component(metrics),
    }


def quality_score_from_metrics(metrics: dict[str, Any]) -> float:
    gt = float(metrics.get("gt_extractable_regions", 0) or 0)
    pred = float(metrics.get("predicted_regions", 0) or 0)
    matched = float(metrics.get("matched_regions", 0) or 0)
    groups = max(1.0, pred)
    order_pairs = max(1.0, matched * max(0.0, matched - 1.0) / 2.0)
    return round(
        12000 * safe_rate(metrics.get("missed_dialogue_region_count", 0), gt)
        + 10000 * safe_rate(metrics.get("destructive_false_positive_count", 0), pred)
        + 8000 * safe_rate(metrics.get("wrong_text_region_match_count", 0), matched)
        + 7000 * float(metrics.get("mean_ocr_cer", 0.0) or 0.0)
        + 6500 * safe_rate(metrics.get("severe_ocr_error_count", 0), matched)
        + 6000 * safe_rate(metrics.get("overmerge_count", 0), groups)
        + 6000 * safe_rate(metrics.get("undermerge_count", 0), max(1.0, gt))
        + 5000 * safe_rate(metrics.get("reading_order_error_count", 0), order_pairs)
        + 4000 * safe_rate(metrics.get("orientation_error_count", 0), matched)
        + 3000 * safe_rate(metrics.get("bad_crop_count", 0), matched)
        + 2500 * safe_rate(metrics.get("duplicate_region_count", 0), pred)
        + 1800 * safe_rate(metrics.get("empty_ocr_count", 0), matched)
        + 1500 * safe_rate(metrics.get("non_japanese_noise_count", 0), pred)
        + 500 * safe_rate(metrics.get("harmless_false_positive_count", 0), pred),
        6,
    )


def timing_score_component(metrics: dict[str, Any]) -> float:
    return round(float(metrics.get("p95_extraction_ms_per_page", metrics.get("total_ms", 0.0)) or 0.0), 6)


def score_from_metrics(metrics: dict[str, Any]) -> float:
    return quality_score_from_metrics(metrics)


def summarize_pages(page_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "pages_total": len(page_metrics),
        "gt_regions_total": sum(int(row.get("gt_regions_total", 0)) for row in page_metrics),
        "gt_extractable_regions": sum(int(row.get("gt_extractable_regions", 0)) for row in page_metrics),
        "predicted_regions": sum(int(row.get("predicted_regions", 0)) for row in page_metrics),
        "matched_regions": sum(int(row.get("matched_regions", 0)) for row in page_metrics),
        "missed_dialogue_region_count": sum(int(row.get("missed_dialogue_region_count", 0)) for row in page_metrics),
        "false_positive_region_count": sum(int(row.get("false_positive_region_count", 0)) for row in page_metrics),
        "destructive_false_positive_count": sum(int(row.get("destructive_false_positive_count", 0)) for row in page_metrics),
        "harmless_false_positive_count": sum(int(row.get("harmless_false_positive_count", 0)) for row in page_metrics),
        "severe_ocr_error_count": sum(int(row.get("severe_ocr_error_count", 0)) for row in page_metrics),
        "wrong_text_region_match_count": sum(int(row.get("wrong_text_region_match_count", 0)) for row in page_metrics),
        "empty_ocr_count": sum(int(row.get("empty_ocr_count", 0)) for row in page_metrics),
        "overmerge_count": sum(int(row.get("overmerge_count", 0)) for row in page_metrics),
        "undermerge_count": sum(int(row.get("undermerge_count", 0)) for row in page_metrics),
        "reading_order_error_count": sum(int(row.get("reading_order_error_count", 0)) for row in page_metrics),
        "orientation_error_count": sum(int(row.get("orientation_error_count", 0)) for row in page_metrics),
        "bad_crop_count": sum(int(row.get("bad_crop_count", 0)) for row in page_metrics),
        "duplicate_region_count": sum(int(row.get("duplicate_region_count", 0)) for row in page_metrics),
        "non_japanese_noise_count": sum(int(row.get("non_japanese_noise_count", 0)) for row in page_metrics),
    }
    matched = max(1, totals["matched_regions"])
    totals["mean_box_iou"] = round(
        sum(float(row.get("mean_box_iou", 0.0)) * int(row.get("matched_regions", 0)) for row in page_metrics) / matched,
        6,
    )
    totals["mean_ocr_cer"] = round(
        sum(float(row.get("mean_ocr_cer", 0.0)) * int(row.get("matched_regions", 0)) for row in page_metrics) / matched,
        6,
    )
    times = [float(row.get("total_ms", 0.0) or 0.0) for row in page_metrics]
    totals["mean_extraction_ms_per_page"] = round(mean(times), 6) if times else 0.0
    totals["p95_extraction_ms_per_page"] = round(percentile(times, 0.95), 6)
    totals["page_score_breakdown"] = score_breakdown_from_metrics(totals)
    totals["source_extraction_quality_score"] = quality_score_from_metrics(totals)
    totals["timing_score_component"] = timing_score_component(totals)
    totals["source_extraction_score"] = score_from_metrics(totals)
    totals["hard_failure"] = hard_failure(totals)
    return totals


def hard_failure(metrics: dict[str, Any]) -> bool:
    try:
        score_value = float(metrics.get("source_extraction_score"))
    except (TypeError, ValueError):
        return True
    if math.isnan(score_value) or math.isinf(score_value):
        return True
    if int(metrics.get("gt_extractable_regions", 0)) > 0 and int(metrics.get("predicted_regions", 0)) == 0:
        return True
    return False


def hard_guards_pass(metrics: dict[str, Any]) -> bool:
    return not bool(metrics.get("hard_failure"))
