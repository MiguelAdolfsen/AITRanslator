from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .cache_policy import translation_cache_plan
from .detect_types import TextBlock

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - used before dependencies are installed.
    def tqdm(iterable, **_kwargs):
        return iterable

from .config import PipelineConfig
from .grouping import context_for_block
from .image_io import iter_images, resolve_output_path
from .line_identity import (
    lookup_translation,
    set_context,
    set_translation,
    translation_key_for_block,
)
from .logging_utils import shorten
from .page_cache import (
    cache_page_path,
    load_prepared_page_cache,
    load_translation_cache,
    resolve_work_dir,
    save_prepared_page_cache,
    save_translation_cache,
)
from .page_types import PreparedPage
from .rendered_page_pass import render_prepared_page
from .text_filter import unusable_translation_reason
from .translated_state import apply_fallback_translation_state
from .translation_review import (
    visual_facts_for_context,
)
from .translation_review_pass import TranslationReviewPass

logger = logging.getLogger(__name__)


def process_folder(input_path: Path, output_path: Path, config: PipelineConfig) -> None:
    from .detect_ocr import configure_tesseract
    from .translate import build_translator

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    logger.info("Process folder started")
    logger.debug("Input path resolved to: %s", input_path)
    logger.debug("Output path resolved to: %s", output_path)
    logger.debug("Pipeline config: %s", config)
    previous_transformers_offline, previous_hf_offline = enable_hf_offline_runtime()
    try:
        if not config.render_only:
            if config.detector == "tesseract" or config.ocr_engine == "tesseract":
                configure_tesseract(config.tesseract_cmd)
            if config.detector == "ctd":
                from .ctd_setup import require_ctd_installed

                require_ctd_installed()
        image_paths = list(iter_images(input_path, recursive=config.recursive))
        logger.info("Found %d supported image(s)", len(image_paths))
        if not image_paths:
            logger.error("No supported images found in %s", input_path)
            raise SystemExit(f"No supported images found in {input_path}")

        if config.render_only:
            render_cached_pages_only(input_path, output_path, image_paths, config)
            return

        if config.translator == "cat" or (
            config.translator in {"qwen", "hy-mt2"}
            and (
                config.qwen_fallback_model_path is not None
                or config.qwen_critic_model_path is not None
                or config.vision_facts_enabled
            )
        ):
            process_folder_qwen_hybrid_batch(input_path, output_path, image_paths, config)
            return

        translator = build_translator(
            config.translator,
            glossary_path=config.glossary_path,
            qwen_model_path=config.qwen_model_path,
            cat_model_name=config.cat_model_name,
            hy_mt2_model_name=config.hy_mt2_model_name,
        )

        processed = 0
        skipped = 0
        chapter_context_before = None
        for index, image_path in enumerate(tqdm(image_paths, desc="Translating pages", unit="page"), start=1):
            target_path = resolve_output_path(input_path, output_path, image_path)
            logger.info("Page %d/%d: %s -> %s", index, len(image_paths), image_path, target_path)
            if target_path.exists() and not config.overwrite:
                skipped += 1
                logger.info("Skipping existing output because overwrite is off: %s", target_path)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            page = prepare_page_for_translation(
                image_path,
                target_path,
                config,
                chapter_context_before=chapter_context_before,
            )
            translate_prepared_page(page, translator, config)
            render_prepared_page(page, config)
            if page.chapter_context_after:
                chapter_context_before = page.chapter_context_after
            processed += 1
    finally:
        restore_env_var("TRANSFORMERS_OFFLINE", previous_transformers_offline)
        restore_env_var("HF_HUB_OFFLINE", previous_hf_offline)

    logger.info("Process folder finished: processed=%d skipped=%d", processed, skipped)


def render_cached_pages_only(
    input_path: Path,
    output_path: Path,
    image_paths: list[Path],
    config: PipelineConfig,
) -> None:
    work_dir = resolve_work_dir(output_path, config)
    if not work_dir.exists():
        raise RuntimeError(f"Render-only cache folder does not exist: {work_dir}")
    render_config = replace(config, skip_render=False, vision_enabled=False)
    cache_plan = translation_cache_plan(config)
    processed = 0
    skipped = 0
    missing: list[str] = []
    logger.info("Starting render-only flow from cache: work_dir=%s", work_dir)
    for index, image_path in enumerate(tqdm(image_paths, desc="Rendering cached pages", unit="page"), start=1):
        target_path = resolve_output_path(input_path, output_path, image_path)
        logger.info("Render-only page %d/%d: %s -> %s", index, len(image_paths), image_path, target_path)
        if target_path.exists() and not config.overwrite:
            skipped += 1
            logger.info("Skipping existing output because overwrite is off: %s", target_path)
            continue
        prepared_cache = cache_plan.prepared_cache_path(work_dir, index, image_path)
        if not prepared_cache.exists():
            missing.append(str(prepared_cache))
            continue
        translation_cache = cache_plan.find_render_only_translation_cache(work_dir, index, image_path)
        if translation_cache is None:
            expected = ", ".join(cache_plan.render_only_lookup_order)
            missing.append(f"{cache_page_path(work_dir, index, image_path, '<translation-stage>')} expected one of: {expected}")
            continue
        logger.info("Loaded render-only translation cache: %s", translation_cache)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        page = load_prepared_page_cache(prepared_cache, output_path=target_path)
        load_translation_cache(page, translation_cache)
        refresh_translation_fallback_blocks(page, config)
        render_prepared_page(page, render_config)
        processed += 1
    if missing:
        missing_preview = "\n".join(missing[:8])
        extra = f"\n...and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise RuntimeError(
            "Render-only cache is incomplete. Re-run the matching translation benchmark first, "
            "or pass the same translator/model/critic/fallback flags used when the cache was created.\n"
            f"{missing_preview}{extra}"
        )
    logger.info("Render-only flow finished: processed=%d skipped=%d", processed, skipped)


def process_folder_qwen_hybrid_batch(
    input_path: Path,
    output_path: Path,
    image_paths: list[Path],
    config: PipelineConfig,
) -> None:
    from .translate import build_translator

    logger.info(
        "Starting batched hybrid flow: translator=%s primary=%s critic=%s fallback=%s",
        config.translator,
        config.qwen_model_path
        if config.translator == "qwen"
        else config.hy_mt2_model_name
        if config.translator == "hy-mt2"
        else config.cat_model_name,
        config.qwen_critic_model_path,
        config.qwen_fallback_model_path,
    )
    work_dir = resolve_work_dir(output_path, config)
    cache_plan = translation_cache_plan(config)
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Batched hybrid work directory: %s resume=%s", work_dir, config.resume)
    prepared_pages: list[PreparedPage] = []
    processed = 0
    skipped = 0
    chapter_context_before = None
    for index, image_path in enumerate(tqdm(image_paths, desc="Preparing pages", unit="page"), start=1):
        target_path = resolve_output_path(input_path, output_path, image_path)
        logger.info("Prepare page %d/%d: %s -> %s", index, len(image_paths), image_path, target_path)
        if target_path.exists() and not config.overwrite:
            skipped += 1
            logger.info("Skipping existing output because overwrite is off: %s", target_path)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_cache = cache_plan.prepared_cache_path(work_dir, index, image_path)
        if config.resume and prepared_cache.exists():
            page = load_prepared_page_cache(prepared_cache, output_path=target_path)
            logger.info("Loaded prepared page cache: %s", prepared_cache)
        else:
            page = prepare_page_for_translation(
                image_path,
                target_path,
                config,
                chapter_context_before=chapter_context_before,
            )
            save_prepared_page_cache(page, prepared_cache)
        if page.chapter_context_after:
            chapter_context_before = page.chapter_context_after
        prepared_pages.append(page)
        processed += 1

    if not prepared_pages:
        logger.info("Batched Qwen hybrid flow finished: processed=0 skipped=%d", skipped)
        return

    review_summary = TranslationReviewPass(
        config=config,
        cache_plan=cache_plan,
        work_dir=work_dir,
        translator_factory=build_translator,
        primary_page_translator=translate_prepared_page,
        vision_facts_runner=apply_vision_facts_to_pages,
    ).run(prepared_pages)

    for page in prepared_pages:
        refresh_translation_fallback_blocks(page, config)

    for page_index, page in enumerate(tqdm(prepared_pages, desc="Rendering pages", unit="page"), start=1):
        render_prepared_page(page, config)
        if config.vision_enabled:
            save_translation_cache(
                page,
                cache_page_path(work_dir, page_index, page.image_path, cache_plan.final_stage),
            )

    logger.info(
        "Batched hybrid flow finished: translator=%s processed=%d skipped=%d fallback_attempted=%d fallback_accepted=%d",
        config.translator,
        processed,
        skipped,
        review_summary.fallback_attempted,
        review_summary.fallback_accepted,
    )


def apply_vision_facts_to_pages(pages: list[PreparedPage], config: PipelineConfig) -> None:
    from .vision_service import apply_vision_facts, release_qwen_vision_clients

    try:
        for page in tqdm(pages, desc="Vision facts", unit="page"):
            page.vision_facts_artifact = apply_vision_facts(
                image_bgr=page.image_bgr,
                blocks=page.render_blocks,
                translation_contexts=page.translation_contexts,
                page_order_report=page.page_order_report,
                output_path=page.output_path,
                config=config,
            )
    finally:
        release_qwen_vision_clients()


def prepare_page_for_translation(
    image_path: Path,
    output_path: Path,
    config: PipelineConfig,
    *,
    chapter_context_before: str | None = None,
) -> PreparedPage:
    from .source_extraction import extract_source_page

    extraction = extract_source_page(
        image_path,
        output_path,
        config,
        chapter_context_before=chapter_context_before,
    )
    return PreparedPage(
        image_path=extraction.image_path,
        output_path=extraction.output_path,
        image_bgr=extraction.image_bgr,
        width=extraction.width,
        height=extraction.height,
        raw_blocks=extraction.raw_blocks,
        render_blocks=extraction.render_blocks,
        skipped_blocks=extraction.skipped_blocks,
        grouping_report=extraction.grouping_report,
        page_order_report=extraction.page_order_report,
    )


def translate_prepared_page(page: PreparedPage, translator, config: PipelineConfig) -> None:
    from .translate import translation_debug_info

    logger.info("Translating %d text block(s) for %s", len(page.render_blocks), page.image_path)
    pre_translation_contexts = {
        key: dict(value)
        for key, value in page.translation_contexts.items()
    }
    page.translations.clear()
    page.translation_contexts.clear()
    page.translation_fallback_blocks.clear()
    page_translations_by_order: dict[int, str] | None = None
    if should_use_qwen_page_mode(translator, config):
        page_translations_by_order = translator.translate_page(
            qwen_page_items(page.render_blocks, page.page_order_report),
            previous_page_context=previous_page_context(page.page_order_report),
        )
    for block in page.render_blocks:
        state_key = translation_key_for_block(block)
        context = {
            **context_for_block(block, page.page_order_report),
            **pre_translation_contexts.get(state_key, pre_translation_contexts.get(block.text, {})),
        }
        if page_translations_by_order is not None:
            page_order = int(context.get("page_order") or 0)
            translated = page_translations_by_order.get(page_order, "")
            context = {
                **context,
                "context_used": bool(context.get("context_available")),
                "translation_mode": "qwen_page",
            }
            if hasattr(translator, "debug_info_for"):
                context.update(translator_debug_info_for(translator, block))
        elif config.translator in {"qwen", "hy-mt2"}:
            translated = translate_with_line_context(
                translator,
                block,
                block.text,
                before=str(context.get("context_before") or "") or None,
                after=str(context.get("context_after") or "") or None,
                before_contexts=tuple(str(value) for value in context.get("context_before_window") or []),
                after_contexts=tuple(str(value) for value in context.get("context_after_window") or []),
                visual_facts=visual_facts_for_context(context),
            )
            context = {
                **context,
                "context_used": bool(context.get("context_available")),
                "translation_mode": "qwen_context" if config.translator == "qwen" else "hy_mt2_context",
            }
            if hasattr(translator, "debug_info_for"):
                context.update(translator_debug_info_for(translator, block))
        else:
            translated = translator.translate(block.text)
            if hasattr(translator, "debug_info_for"):
                context.update(translator_debug_info_for(translator, block))
        set_translation(page.translations, block, translated)
        set_context(page.translation_contexts, block, {
            **context,
            **translation_debug_info(block.text, translated, glossary_path=config.glossary_path),
        })
        reason = unusable_translation_reason(block.text, translated, translator_name=config.translator)
        if reason:
            logger.info(
                "Using fallback translation: reason=%s box=%s source=%s translation=%s",
                reason,
                block.box,
                shorten(block.text),
                shorten(translated),
            )
            translated = apply_fallback_translation_state(
                page,
                block,
                reason=reason,
                source_translation=translated,
            )
    logger.debug("Translation map contains %d line id(s)", len(page.translations))


def should_use_qwen_page_mode(translator, config: PipelineConfig) -> bool:
    return config.translator == "qwen" and config.qwen_mode == "page" and hasattr(translator, "translate_page")


def translate_with_line_context(translator, block: TextBlock, text: str, **kwargs) -> str:
    try:
        return translator.translate_with_context(text, debug_id=translation_key_for_block(block), **kwargs)
    except TypeError:
        return translator.translate_with_context(text, **kwargs)


def translator_debug_info_for(translator, block: TextBlock) -> dict[str, object]:
    try:
        return dict(translator.debug_info_for(block.text, debug_id=translation_key_for_block(block)))
    except TypeError:
        return dict(translator.debug_info_for(block.text))


def qwen_page_items(blocks: list[TextBlock], page_order_report: list[dict[str, object]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for fallback_index, block in enumerate(blocks, start=1):
        context = context_for_block(block, page_order_report)
        page_order = int(context.get("page_order") or fallback_index)
        items.append(
            {
                "id": page_order,
                "line_id": translation_key_for_block(block),
                "text": block.text,
                "before": str(context.get("context_before") or ""),
                "after": str(context.get("context_after") or ""),
                "before_contexts": [str(value) for value in context.get("context_before_window") or []],
                "after_contexts": [str(value) for value in context.get("context_after_window") or []],
            }
        )
    return items


def previous_page_context(page_order_report: list[dict[str, object]]) -> str | None:
    if not page_order_report:
        return None
    value = page_order_report[0].get("context_before")
    return str(value) if value else None


def refresh_translation_fallback_blocks(page: PreparedPage, config: PipelineConfig) -> None:
    page.translation_fallback_blocks.clear()
    for block in page.render_blocks:
        translated = lookup_translation(page.translations, block, "")
        reason = unusable_translation_reason(block.text, translated, translator_name=config.translator)
        if not reason:
            continue
        logger.info(
            "Using final fallback translation: reason=%s box=%s source=%s translation=%s",
            reason,
            block.box,
            shorten(block.text),
            shorten(translated),
        )
        apply_fallback_translation_state(
            page,
            block,
            reason=reason,
            source_translation=translated,
        )


def enable_hf_offline_runtime() -> tuple[str | None, str | None]:
    previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.debug("Enabled Hugging Face offline runtime mode")
    return previous_transformers_offline, previous_hf_offline


def restore_env_var(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


