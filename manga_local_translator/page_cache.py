from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import PipelineConfig
from .detect_types import TextBlock
from .line_identity import (
    assign_ocr_block_ids,
    assign_render_line_ids,
    enrich_grouping_report,
    enrich_page_order_report,
    line_id_for_block,
)
from .page_types import PreparedPage
from .translated_state import TranslationReviewState, apply_fallback_translation_state

logger = logging.getLogger(__name__)


def resolve_work_dir(output_path: Path, config: PipelineConfig) -> Path:
    if config.work_dir is not None:
        return config.work_dir.resolve()
    base = output_path.parent if output_path.suffix else output_path
    return (base / ".manga-work").resolve()


def cache_page_path(work_dir: Path, page_index: int, image_path: Path, stage: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("._") or "page"
    return work_dir / f"{page_index:04d}-{safe_name}.{stage}.json"


def text_block_to_cache(block: TextBlock) -> dict[str, object]:
    return {
        "text": block.text,
        "box": list(block.box),
        "confidence": block.confidence,
        "detector": block.detector,
        "metadata": block.metadata,
    }


def text_block_from_cache(payload: dict[str, object]) -> TextBlock:
    box = payload.get("box", [0, 0, 0, 0])
    return TextBlock(
        text=str(payload.get("text", "")),
        box=tuple(int(value) for value in box) if is_box_like(box) else (0, 0, 0, 0),
        confidence=float(payload.get("confidence", 0.0)),
        detector=str(payload.get("detector", "unknown")),
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
    )


def save_prepared_page_cache(page: PreparedPage, cache_path: Path) -> None:
    payload = {
        "version": 1,
        "image_path": str(page.image_path),
        "output_path": str(page.output_path),
        "width": page.width,
        "height": page.height,
        "raw_blocks": [text_block_to_cache(block) for block in page.raw_blocks],
        "render_blocks": [text_block_to_cache(block) for block in page.render_blocks],
        "skipped_blocks": page.skipped_blocks,
        "grouping_report": page.grouping_report,
        "page_order_report": page.page_order_report,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Prepared page cache written: %s", cache_path)


def load_prepared_page_cache(cache_path: Path, *, output_path: Path | None = None) -> PreparedPage:
    import cv2

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    image_path = Path(str(payload["image_path"]))
    cached_output_path = Path(str(payload["output_path"]))
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read cached image: {image_path}")
    raw_blocks = [text_block_from_cache(item) for item in payload.get("raw_blocks", [])]
    render_blocks = [text_block_from_cache(item) for item in payload.get("render_blocks", [])]
    page_order_report = list(payload.get("page_order_report", []))
    raw_blocks = assign_ocr_block_ids(raw_blocks, role="ocr")
    render_blocks = assign_render_line_ids(render_blocks, page_order_report, output_path or cached_output_path)
    page_order_report = enrich_page_order_report(page_order_report, render_blocks)
    grouping_report = enrich_grouping_report(list(payload.get("grouping_report", [])), render_blocks)
    return PreparedPage(
        image_path=image_path,
        output_path=output_path or cached_output_path,
        image_bgr=image_bgr,
        width=int(payload.get("width", image_bgr.shape[1])),
        height=int(payload.get("height", image_bgr.shape[0])),
        raw_blocks=raw_blocks,
        render_blocks=render_blocks,
        skipped_blocks=list(payload.get("skipped_blocks", [])),
        grouping_report=grouping_report,
        page_order_report=page_order_report,
    )


def save_translation_cache(page: PreparedPage, cache_path: Path) -> None:
    state = TranslationReviewState.for_page(page)
    payload = {
        "version": 3,
        "cache_kind": "translation",
        "image_path": str(page.image_path),
        "output_path": str(page.output_path),
        "translations_by_id": state.translations_by_id(),
        "translation_contexts_by_id": state.contexts_by_id(),
        "translations": state.source_compat_translations(page.render_blocks),
        "translation_contexts": state.contexts_by_id(),
        "translation_fallback_blocks": page.translation_fallback_blocks,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Translation cache written: %s", cache_path)


def load_translation_cache(page: PreparedPage, cache_path: Path) -> None:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    raw_translations = {
        str(key): str(value)
        for key, value in dict(payload.get("translations_by_id") or payload.get("translations", {})).items()
    }
    raw_contexts = {
        str(key): dict(value) if isinstance(value, dict) else {}
        for key, value in dict(payload.get("translation_contexts_by_id") or payload.get("translation_contexts", {})).items()
    }
    page.translations, page.translation_contexts = TranslationReviewState.migrate_to_line_ids(
        page.render_blocks,
        raw_translations,
        raw_contexts,
    )
    page.translation_fallback_blocks = list(payload.get("translation_fallback_blocks", []))
    hydrate_cached_fallback_translation_state(page)


def hydrate_cached_fallback_translation_state(page: PreparedPage) -> None:
    for fallback_block in page.translation_fallback_blocks:
        if not isinstance(fallback_block, dict):
            continue
        reason = str(fallback_block.get("reason") or "")
        if not reason:
            continue
        block = block_for_fallback_report(page.render_blocks, fallback_block)
        if block is None:
            continue
        apply_fallback_translation_state(
            page,
            block,
            reason=reason,
            source_translation=TranslationReviewState.for_page(page).translation_for(block, ""),
            append_report=False,
        )


def block_for_fallback_report(
    blocks: list[TextBlock],
    fallback_block: dict[str, object],
) -> TextBlock | None:
    line_id = str(fallback_block.get("line_id") or "")
    if line_id:
        for block in blocks:
            if line_id_for_block(block) == line_id:
                return block
    source_text = str(fallback_block.get("source_text") or "")
    box = fallback_block.get("box")
    if is_box_like(box):
        fallback_box = tuple(int(value) for value in box)  # type: ignore[union-attr]
        for block in blocks:
            if block.text == source_text and tuple(block.box) == fallback_box:
                return block
    for block in blocks:
        if block.text == source_text:
            return block
    return None


def is_box_like(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        [int(item) for item in value]
    except (TypeError, ValueError):
        return False
    return True
