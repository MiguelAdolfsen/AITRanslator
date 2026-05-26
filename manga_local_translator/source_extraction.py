from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .detect_types import TextBlock
from .grouping import build_page_order_report, group_text_blocks_for_translation
from .line_identity import (
    assign_ocr_block_ids,
    assign_render_line_ids,
    enrich_grouping_report,
    enrich_page_order_report,
)
from .text_filter import filter_text_blocks

logger = logging.getLogger(__name__)


@dataclass
class SourceExtractionResult:
    image_path: Path
    output_path: Path
    image_bgr: Any
    width: int
    height: int
    raw_blocks: list[TextBlock]
    render_blocks: list[TextBlock]
    skipped_blocks: list[dict[str, object]]
    grouping_report: list[dict[str, object]]
    page_order_report: list[dict[str, object]]


def extract_source_page(
    image_path: Path,
    output_path: Path,
    config: PipelineConfig,
    *,
    chapter_context_before: str | None = None,
) -> SourceExtractionResult:
    import cv2

    from .detect_ocr import run_ocr

    logger.info("Reading image: %s", image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        logger.error("OpenCV could not read image: %s", image_path)
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image_bgr.shape[:2]
    logger.debug("Image loaded: width=%d height=%d channels=%d", width, height, image_bgr.shape[2])

    logger.info("Running OCR")
    raw_blocks = run_ocr(
        image_path,
        detector=config.detector,
        engine=config.ocr_engine,
        lang=config.tesseract_lang,
        psm=config.tesseract_psm,
        min_confidence=config.min_confidence,
    )
    raw_blocks = assign_ocr_block_ids(raw_blocks, role="ocr")
    logger.info("OCR returned %d raw text block(s)", len(raw_blocks))
    blocks, skipped_blocks = filter_text_blocks(raw_blocks, width=width, height=height)
    logger.info("OCR filter kept %d block(s), skipped %d block(s)", len(blocks), len(skipped_blocks))
    render_blocks, grouping_report = group_text_blocks_for_translation(blocks, image_bgr=image_bgr)
    page_order_report = build_page_order_report(
        render_blocks,
        width=width,
        height=height,
        chapter_context_before=chapter_context_before,
    )
    render_blocks = assign_render_line_ids(render_blocks, page_order_report, output_path)
    page_order_report = enrich_page_order_report(page_order_report, render_blocks)
    grouping_report = enrich_grouping_report(grouping_report, render_blocks)
    logger.info(
        "Source extraction complete: input_blocks=%d render_blocks=%d grouped_sets=%d",
        len(blocks),
        len(render_blocks),
        sum(1 for group in grouping_report if len(group["members"]) > 1),
    )
    return SourceExtractionResult(
        image_path=image_path,
        output_path=output_path,
        image_bgr=image_bgr,
        width=width,
        height=height,
        raw_blocks=raw_blocks,
        render_blocks=render_blocks,
        skipped_blocks=skipped_blocks,
        grouping_report=grouping_report,
        page_order_report=page_order_report,
    )
