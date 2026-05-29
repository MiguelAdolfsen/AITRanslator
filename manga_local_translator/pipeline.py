from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .cache_policy import render_only_translation_stages, translation_cache_stage
from .detect_types import TextBlock

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - used before dependencies are installed.
    def tqdm(iterable, **_kwargs):
        return iterable

from .config import PipelineConfig
from .debug_report import write_debug_image, write_debug_report
from .grouping import context_for_block
from .image_io import iter_images, resolve_output_path, write_image_with_fallback
from .line_identity import (
    lookup_context,
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
from .text_filter import (
    block_to_debug_dict,
    fallback_translation,
    is_box_like,
    unusable_translation_reason,
)
from .translation_review import (
    apply_qwen_critic_reviews,
    apply_qwen_fallback_translations,
    apply_translation_evidence_to_pages,
    retry_cat_failures,
    visual_facts_for_context,
)

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
        prepared_cache = cache_page_path(work_dir, index, image_path, "prepared")
        if not prepared_cache.exists():
            missing.append(str(prepared_cache))
            continue
        translation_cache = find_render_only_translation_cache(work_dir, index, image_path, config)
        if translation_cache is None:
            expected = ", ".join(render_only_translation_stages(config))
            missing.append(f"{cache_page_path(work_dir, index, image_path, '<translation-stage>')} expected one of: {expected}")
            continue
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


def find_render_only_translation_cache(
    work_dir: Path,
    page_index: int,
    image_path: Path,
    config: PipelineConfig,
) -> Path | None:
    for stage in render_only_translation_stages(config):
        cache_path = cache_page_path(work_dir, page_index, image_path, stage)
        if cache_path.exists():
            logger.info("Loaded render-only translation cache: %s", cache_path)
            return cache_path
    return None


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
        prepared_cache = cache_page_path(work_dir, index, image_path, "prepared")
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

    primary_jobs: list[tuple[int, PreparedPage, Path]] = []
    hybrid_cached_pages: set[int] = set()
    critic_cached_pages: set[int] = set()
    for page_index, page in enumerate(prepared_pages, start=1):
        hybrid_cache = cache_page_path(work_dir, page_index, page.image_path, translation_cache_stage(config, "hybrid"))
        critic_cache = cache_page_path(work_dir, page_index, page.image_path, translation_cache_stage(config, "critic"))
        primary_cache = cache_page_path(work_dir, page_index, page.image_path, translation_cache_stage(config, "primary"))
        if config.resume and hybrid_cache.exists():
            load_translation_cache(page, hybrid_cache)
            hybrid_cached_pages.add(page_index)
            logger.info("Loaded hybrid translation cache: %s", hybrid_cache)
        elif config.resume and config.qwen_critic_model_path is not None and critic_cache.exists():
            load_translation_cache(page, critic_cache)
            critic_cached_pages.add(page_index)
            logger.info("Loaded critic translation cache: %s", critic_cache)
        elif config.resume and primary_cache.exists():
            load_translation_cache(page, primary_cache)
            logger.info("Loaded primary translation cache: %s", primary_cache)
        else:
            primary_jobs.append((page_index, page, primary_cache))

    if config.vision_facts_enabled and primary_jobs:
        logger.info("Running pre-translation vision facts for %d pending page(s)", len(primary_jobs))
        apply_vision_facts_to_pages([page for _page_index, page, _cache in primary_jobs], config)

    if primary_jobs:
        logger.info("Building primary %s translator for %d pending page(s)", config.translator, len(primary_jobs))
        primary_translator = build_translator(
            config.translator,
            glossary_path=config.glossary_path,
            qwen_model_path=config.qwen_model_path,
            cat_model_name=config.cat_model_name,
            hy_mt2_model_name=config.hy_mt2_model_name,
        )
        for _page_index, page, primary_cache in tqdm(primary_jobs, desc=f"{config.translator.upper()} primary pass", unit="page"):
            translate_prepared_page(page, primary_translator, config)
            save_translation_cache(page, primary_cache)
        if config.translator == "cat":
            total_cat_retry_attempted = 0
            total_cat_retry_accepted = 0
            for _page_index, page, primary_cache in tqdm(primary_jobs, desc="CAT retry pass", unit="page"):
                attempted, accepted = retry_cat_failures(page, primary_translator, config)
                if attempted:
                    save_translation_cache(page, primary_cache)
                total_cat_retry_attempted += attempted
                total_cat_retry_accepted += accepted
            logger.info(
                "CAT retry scan complete: attempted=%d accepted=%d",
                total_cat_retry_attempted,
                total_cat_retry_accepted,
            )
        release_qwen_translator(primary_translator)
        del primary_translator
    else:
        logger.info("Primary %s pass skipped; all pages loaded from cache", config.translator)

    evidence_memory = None
    if config.qwen_critic_model_path is not None or config.qwen_fallback_model_path is not None:
        evidence_memory = apply_translation_evidence_to_pages(prepared_pages)

    critic_jobs = []
    if config.qwen_critic_model_path is not None:
        critic_jobs = [
            (page_index, page, cache_page_path(work_dir, page_index, page.image_path, translation_cache_stage(config, "critic")))
            for page_index, page in enumerate(prepared_pages, start=1)
            if page_index not in hybrid_cached_pages and page_index not in critic_cached_pages
        ]
    total_critic_attempted = 0
    total_critic_flagged = 0
    if critic_jobs:
        logger.info("Primary %s pass complete; building critic Qwen translator for %d pending page(s)", config.translator, len(critic_jobs))
        critic_translator = build_translator(
            "qwen",
            glossary_path=config.glossary_path,
            qwen_model_path=config.qwen_critic_model_path,
        )
        for _page_index, page, critic_cache in tqdm(critic_jobs, desc="Qwen critic pass", unit="page"):
            attempted, flagged = apply_qwen_critic_reviews(
                page.render_blocks,
                page.translations,
                page.translation_contexts,
                page.page_order_report,
                critic_translator,
                config,
                evidence_memory=evidence_memory,
            )
            save_translation_cache(page, critic_cache)
            total_critic_attempted += attempted
            total_critic_flagged += flagged
        release_qwen_translator(critic_translator)
        del critic_translator
    elif config.qwen_critic_model_path is not None:
        logger.info("Critic Qwen pass skipped; all pages loaded from cache")
    logger.info(
        "Critic Qwen pass complete: attempted=%d flagged=%d",
        total_critic_attempted,
        total_critic_flagged,
    )

    fallback_jobs = []
    if config.qwen_fallback_model_path is not None:
        fallback_jobs = [
            (page_index, page, cache_page_path(work_dir, page_index, page.image_path, translation_cache_stage(config, "hybrid")))
            for page_index, page in enumerate(prepared_pages, start=1)
            if page_index not in hybrid_cached_pages
        ]
    total_attempted = 0
    total_accepted = 0
    for page_index in hybrid_cached_pages:
        page = prepared_pages[page_index - 1]
        total_attempted += sum(1 for context in page.translation_contexts.values() if context.get("qwen_fallback_attempted") is True)
        total_accepted += sum(1 for context in page.translation_contexts.values() if context.get("qwen_fallback_accepted") is True)

    if fallback_jobs:
        logger.info("Primary %s pass complete; building fallback Qwen translator for %d pending page(s)", config.translator, len(fallback_jobs))
        fallback_translator = build_translator(
            "qwen",
            glossary_path=config.glossary_path,
            qwen_model_path=config.qwen_fallback_model_path,
        )
        for _page_index, page, hybrid_cache in tqdm(fallback_jobs, desc="Q8 fallback pass", unit="page"):
            attempted, accepted = apply_qwen_fallback_translations(
                page.render_blocks,
                page.translations,
                page.translation_contexts,
                page.page_order_report,
                fallback_translator,
                config,
                evidence_memory=evidence_memory,
            )
            save_translation_cache(page, hybrid_cache)
            total_attempted += attempted
            total_accepted += accepted
        release_qwen_translator(fallback_translator)
        del fallback_translator
    else:
        logger.info("Fallback Qwen pass skipped; all pages loaded from hybrid cache")
    logger.info(
        "Fallback Qwen pass complete: attempted=%d accepted=%d",
        total_attempted,
        total_accepted,
    )

    for page in prepared_pages:
        refresh_translation_fallback_blocks(page, config)

    final_stage = "hybrid" if config.qwen_fallback_model_path is not None else "primary"
    for page_index, page in enumerate(tqdm(prepared_pages, desc="Rendering pages", unit="page"), start=1):
        render_prepared_page(page, config)
        if config.vision_enabled:
            save_translation_cache(
                page,
                cache_page_path(work_dir, page_index, page.image_path, translation_cache_stage(config, final_stage)),
            )

    logger.info(
        "Batched hybrid flow finished: translator=%s processed=%d skipped=%d fallback_attempted=%d fallback_accepted=%d",
        config.translator,
        processed,
        skipped,
        total_attempted,
        total_accepted,
    )


def release_qwen_translator(translator) -> None:
    ollama = getattr(translator, "_ollama", None)
    model_name = getattr(translator, "_ollama_model_name", None)
    if not ollama or not model_name:
        return
    completed = subprocess.run(
        [ollama, "stop", str(model_name)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=60,
        check=False,
    )
    if completed.returncode == 0:
        logger.info("Stopped Ollama model to free VRAM: %s", model_name)
    else:
        logger.debug("Could not stop Ollama model %s: %s", model_name, shorten(completed.stderr))


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
            translated = fallback_translation(block.text, reason=reason)
            set_translation(page.translations, block, translated)
            page.translation_fallback_blocks.append(
                block_to_debug_dict(block, status="fallback", reason=reason, translated_text=translated)
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
        fallback = fallback_translation(block.text, reason=reason)
        set_translation(page.translations, block, fallback)
        page.translation_fallback_blocks.append(
            block_to_debug_dict(block, status="fallback", reason=reason, translated_text=fallback)
        )


def render_prepared_page(page: PreparedPage, config: PipelineConfig) -> Path:
    from .erase import erase_text
    from .render import plan_render_layouts, plan_text_fits, render_translations
    from .vision_service import apply_vision_repair

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
