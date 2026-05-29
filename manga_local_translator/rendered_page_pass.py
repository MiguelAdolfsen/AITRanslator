from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import PipelineConfig
from .debug_report import write_debug_image, write_debug_report
from .detect_types import TextBlock
from .erase import erase_text
from .image_io import write_image_with_fallback
from .line_identity import lookup_context, lookup_translation, set_context, set_translation
from .logging_utils import shorten
from .page_types import PreparedPage
from .render import plan_render_layouts, plan_text_fits, render_translations
from .text_filter import is_box_like
from .vision_service import apply_vision_repair

logger = logging.getLogger(__name__)


def render_prepared_page(page: PreparedPage, config: PipelineConfig) -> Path:
    if config.skip_render:
        write_translation_only_debug_report(page, config)
        logger.info("Skipped translated image rendering for benchmark: %s", page.output_path)
        return page.output_path

    render_layouts = plan_render_layouts(
        page.image_bgr,
        page.render_blocks,
        render_expand=config.render_expand,
    )
    render_fits = plan_text_fits(
        page.image_bgr,
        page.render_blocks,
        page.translations,
        font_path=config.font_path,
        base_font_size=config.base_font_size,
        render_expand=config.render_expand,
        render_layouts=render_layouts,
    )
    compact_translations_for_render(
        page.image_bgr,
        page.render_blocks,
        page.translations,
        translation_contexts=page.translation_contexts,
        render_layouts=render_layouts,
        render_fits=render_fits,
        font_path=config.font_path,
        base_font_size=config.base_font_size,
        render_expand=config.render_expand,
    )
    render_fits = plan_text_fits(
        page.image_bgr,
        page.render_blocks,
        page.translations,
        font_path=config.font_path,
        base_font_size=config.base_font_size,
        render_expand=config.render_expand,
        render_layouts=render_layouts,
    )
    page.vision_artifact = apply_vision_repair(
        image_bgr=page.image_bgr,
        blocks=page.render_blocks,
        translations=page.translations,
        translation_contexts=page.translation_contexts,
        page_order_report=page.page_order_report,
        render_layouts=render_layouts,
        render_fits=render_fits,
        output_path=page.output_path,
        config=config,
    )
    if any(context.get("vision_accepted") is True for context in page.translation_contexts.values()):
        render_fits = plan_text_fits(
            page.image_bgr,
            page.render_blocks,
            page.translations,
            font_path=config.font_path,
            base_font_size=config.base_font_size,
            render_expand=config.render_expand,
            render_layouts=render_layouts,
        )
        compact_translations_for_render(
            page.image_bgr,
            page.render_blocks,
            page.translations,
            translation_contexts=page.translation_contexts,
            render_layouts=render_layouts,
            render_fits=render_fits,
            font_path=config.font_path,
            base_font_size=config.base_font_size,
            render_expand=config.render_expand,
        )
        render_fits = plan_text_fits(
            page.image_bgr,
            page.render_blocks,
            page.translations,
            font_path=config.font_path,
            base_font_size=config.base_font_size,
            render_expand=config.render_expand,
            render_layouts=render_layouts,
        )
    if config.debug:
        write_debug_report(
            page.image_path,
            page.output_path,
            config,
            page.raw_blocks,
            page.render_blocks,
            page.translations,
            page.skipped_blocks,
            page.translation_fallback_blocks,
            page.grouping_report,
            render_layouts,
            render_fits,
            page.width,
            page.height,
            config.padding,
            page.page_order_report,
            page.translation_contexts,
            vision_artifact=page.vision_artifact,
            vision_facts_artifact=page.vision_facts_artifact,
        )
    erase_blocks = blocks_for_erasing(page.render_blocks)
    cleaned = erase_text(
        page.image_bgr,
        erase_blocks,
        mode=config.erase_mode,
        padding=config.padding,
    )
    rendered = render_translations(
        cleaned,
        page.render_blocks,
        page.translations,
        font_path=config.font_path,
        base_font_size=config.base_font_size,
        render_expand=config.render_expand,
        render_layouts=render_layouts,
    )
    actual_output_path = write_image_with_fallback(rendered, page.output_path)
    logger.info("Translated image written: %s", actual_output_path)
    if config.debug:
        logger.info("Writing debug OCR box image")
        write_debug_image(
            page.image_bgr,
            page.render_blocks,
            page.skipped_blocks,
            page.translation_fallback_blocks,
            render_layouts,
            actual_output_path,
        )
    return actual_output_path


class TranslationOnlyLayout:
    def __init__(self, render_box: tuple[int, int, int, int]) -> None:
        self.render_box = render_box
        self.bubble_box = None


def write_translation_only_debug_report(page: PreparedPage, config: PipelineConfig) -> None:
    if not config.debug:
        return
    render_layouts = [TranslationOnlyLayout(block.box) for block in page.render_blocks]
    render_fits = [None for _block in page.render_blocks]
    write_debug_report(
        page.image_path,
        page.output_path,
        config,
        page.raw_blocks,
        page.render_blocks,
        page.translations,
        page.skipped_blocks,
        page.translation_fallback_blocks,
        page.grouping_report,
        render_layouts,
        render_fits,
        page.width,
        page.height,
        config.padding,
        page.page_order_report,
        page.translation_contexts,
        vision_artifact=page.vision_artifact,
        vision_facts_artifact=page.vision_facts_artifact,
    )


def blocks_for_erasing(blocks: list[TextBlock]) -> list[TextBlock]:
    erase_blocks: list[TextBlock] = []
    for block in blocks:
        grouped_from = block.metadata.get("grouped_from") if getattr(block, "metadata", None) else None
        if not isinstance(grouped_from, list):
            erase_blocks.append(block)
            continue

        for member in grouped_from:
            if not isinstance(member, dict):
                continue
            box = member.get("box")
            if not is_box_like(box):
                continue
            erase_blocks.append(
                TextBlock(
                    text=str(member.get("source_text", block.text)),
                    box=tuple(int(value) for value in box),
                    confidence=float(member.get("confidence", block.confidence)),
                    detector=block.detector,
                    metadata=dict(member.get("metadata", {})) if isinstance(member.get("metadata"), dict) else {},
                )
            )
    logger.debug("Erase block expansion complete: render_blocks=%d erase_blocks=%d", len(blocks), len(erase_blocks))
    return erase_blocks


def compact_translations_for_render(
    image_bgr,
    blocks: list[TextBlock],
    translations: dict[str, str],
    *,
    translation_contexts: dict[str, dict[str, object]] | None = None,
    render_layouts,
    render_fits,
    font_path: Path | None,
    base_font_size: int,
    render_expand: float,
) -> None:
    _ = image_bgr, render_layouts, font_path, base_font_size, render_expand
    compacted = 0
    for block, fit in zip(blocks, render_fits):
        text = lookup_translation(translations, block, "")
        if not text:
            continue
        if getattr(fit, "font_size", 99) > 7 and not getattr(fit, "clipped", False):
            continue
        shortened = compact_english_for_bubble(text)
        if shortened == text:
            continue
        set_translation(translations, block, shortened)
        if translation_contexts is not None:
            context = lookup_context(translation_contexts, block)
            context["compacted_for_render"] = True
            set_context(translation_contexts, block, context)
        compacted += 1
        logger.info(
            "Compacted translation for small render box: box=%s font_size=%s before=%s after=%s",
            block.box,
            getattr(fit, "font_size", "?"),
            shorten(text),
            shorten(shortened),
        )
    if compacted:
        logger.info("Compacted %d translation(s) for render readability", compacted)


def compact_english_for_bubble(text: str) -> str:
    result = text.strip()
    replacements = (
        (r"\bwas able to\b", "managed to"),
        (r"\bwere able to\b", "managed to"),
        (r"\bis able to\b", "can"),
        (r"\bare able to\b", "can"),
        (r"\bmight have been\b", "may be"),
        (r"\bmight be\b", "may be"),
        (r"\bgoing to\b", "gonna"),
        (r"\breally\s+", ""),
        (r"\bvery\s+", ""),
    )
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r"\s+", " ", result).strip()
    return result
