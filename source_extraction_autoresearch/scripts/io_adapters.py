from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from normalize_text import normalize_text


ADAPTER_NOTES: list[str] = []


def _orientation_from_metadata(metadata: dict[str, Any]) -> str:
    if metadata.get("vertical") is True:
        return "vertical"
    if metadata.get("vertical") is False:
        return "horizontal"
    return "unknown"


def _block_id(block: Any, fallback: str) -> str:
    metadata = getattr(block, "metadata", {}) or {}
    return str(metadata.get("line_id") or metadata.get("block_id") or fallback)


def extract_source_page(image_path: str, page_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return predicted text regions, OCR text, groups, order, and timing for one page image."""
    page_config = page_config or {}
    started = time.perf_counter()
    try:
        from manga_local_translator.config import PipelineConfig
        from manga_local_translator.pipeline import prepare_page_for_translation
    except Exception as exc:
        raise RuntimeError(f"Project source-extraction imports failed: {exc}") from exc

    image = Path(image_path)
    config = PipelineConfig(
        detector=str(page_config.get("detector") or "ctd"),
        ocr_engine=str(page_config.get("ocr_engine") or "manga-ocr"),
        tesseract_lang=str(page_config.get("tesseract_lang") or "jpn+jpn_vert"),
        tesseract_psm=int(page_config.get("tesseract_psm") or 11),
        min_confidence=float(page_config.get("min_confidence") or 20.0),
        skip_render=True,
        vision_enabled=False,
        vision_facts_enabled=False,
    )
    try:
        page = prepare_page_for_translation(image, image.with_suffix(".source-extraction.out.png"), config)
    except Exception as exc:
        raise RuntimeError(
            f"Project source extraction failed. Install detector/OCR dependencies or run with --adapter-smoke. Details: {exc}"
        ) from exc

    page_order_by_key: dict[tuple[tuple[int, int, int, int], str], int] = {}
    for item in page.page_order_report:
        page_order_by_key[(tuple(item.get("box", ())), str(item.get("source_text", "")))] = int(item.get("page_order", 0) or 0)

    regions: list[dict[str, Any]] = []
    for index, block in enumerate(page.raw_blocks, start=1):
        metadata = dict(getattr(block, "metadata", {}) or {})
        pred_id = _block_id(block, f"p{index:03d}")
        text = str(getattr(block, "text", "") or "")
        regions.append(
            {
                "pred_region_id": pred_id,
                "box": [int(value) for value in getattr(block, "box")],
                "ocr_text": text,
                "normalized_ocr_text": normalize_text(text),
                "confidence": float(getattr(block, "confidence", 0.0) or 0.0),
                "orientation": _orientation_from_metadata(metadata),
                "kind": "unknown",
                "group_id": str(metadata.get("translation_group_id") or pred_id),
                "page_order": page_order_by_key.get((tuple(getattr(block, "box")), text)),
                "source_engine": config.ocr_engine,
                "detector": str(getattr(block, "detector", config.detector)),
                "metadata": metadata,
            }
        )

    groups: list[dict[str, Any]] = []
    for index, block in enumerate(page.render_blocks, start=1):
        metadata = dict(getattr(block, "metadata", {}) or {})
        group_id = str(metadata.get("translation_group_id") or f"pg{index:03d}")
        grouped_from = metadata.get("grouped_from") if isinstance(metadata.get("grouped_from"), list) else []
        member_ids: list[str] = []
        if grouped_from:
            for member in grouped_from:
                if isinstance(member, dict):
                    member_id = str(member.get("line_id") or member.get("block_id") or "")
                    if member_id:
                        member_ids.append(member_id)
        if not member_ids:
            member_ids = [_block_id(block, f"p{index:03d}")]
        groups.append(
            {
                "pred_group_id": group_id,
                "member_pred_region_ids": member_ids,
                "combined_ocr_text": str(getattr(block, "text", "") or ""),
                "page_order": page_order_by_key.get((tuple(getattr(block, "box")), str(getattr(block, "text", "") or ""))),
                "kind": "unknown",
            }
        )

    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema_version": 1,
        "page_id": image.stem,
        "image_path": str(image),
        "adapter_mode": "project",
        "regions": regions,
        "groups": groups,
        "timing": {"detect_ms": 0.0, "ocr_ms": 0.0, "group_ms": 0.0, "total_ms": round(total_ms, 3)},
        "adapter_notes": list(ADAPTER_NOTES),
    }


def extract_source_page_smoke(image_path: str, label: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fake predictor used only to verify the harness."""
    started = time.perf_counter()
    regions: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    region_id_map: dict[str, str] = {}
    for index, gt in enumerate(label.get("text_regions", []), start=1):
        if not bool(gt.get("should_extract")):
            continue
        x1, y1, x2, y2 = [int(value) for value in gt["box"]]
        jitter = 1 if index % 2 == 0 else 0
        pred_id = f"p{index:03d}"
        region_id_map[str(gt["region_id"])] = pred_id
        text = str(gt.get("source_text", ""))
        regions.append(
            {
                "pred_region_id": pred_id,
                "box": [x1 - jitter, y1 - jitter, x2 + jitter, y2 + jitter],
                "ocr_text": text,
                "normalized_ocr_text": normalize_text(text),
                "confidence": 99.0,
                "orientation": str(gt.get("orientation", "unknown")),
                "kind": str(gt.get("kind", "unknown")),
                "group_id": str(gt.get("group_id") or pred_id),
                "page_order": gt.get("page_order"),
                "source_engine": "adapter-smoke",
                "detector": "adapter-smoke",
                "metadata": {"smoke_gt_region_id": gt.get("region_id")},
            }
        )
    for group in label.get("groups", []):
        member_ids = [region_id_map[item] for item in [str(v) for v in group.get("member_region_ids", [])] if item in region_id_map]
        if not member_ids:
            continue
        text = str(group.get("combined_source_text", ""))
        groups.append(
            {
                "pred_group_id": str(group.get("group_id")),
                "member_pred_region_ids": member_ids,
                "combined_ocr_text": text,
                "page_order": group.get("page_order"),
                "kind": str(group.get("kind", "unknown")),
            }
        )
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema_version": 1,
        "page_id": str(label["page_id"]),
        "image_path": image_path,
        "adapter_mode": "adapter-smoke",
        "regions": regions,
        "groups": groups,
        "timing": {"detect_ms": 0.0, "ocr_ms": 0.0, "group_ms": 0.0, "total_ms": round(total_ms, 3)},
        "adapter_notes": ["adapter-smoke predictions are label-derived and not valid for project improvements"],
    }

