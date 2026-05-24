from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .detect_types import TextBlock

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - used before dependencies are installed.
    def tqdm(iterable, **_kwargs):
        return iterable

from .config import PipelineConfig
from .debug_report import write_debug_image, write_debug_report
from .grouping import (
    build_page_order_report,
    context_for_block,
    group_text_blocks_for_translation,
)
from .image_io import iter_images, resolve_output_path, write_image_with_fallback
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
    filter_text_blocks,
    is_box_like,
    suspected_bad_translation,
    unusable_translation_reason,
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

        if config.translator == "qwen" and (
            config.qwen_fallback_model_path is not None
            or config.qwen_critic_model_path is not None
            or config.vision_facts_enabled
        ):
            process_folder_qwen_hybrid_batch(input_path, output_path, image_paths, config)
            return

        translator = build_translator(
            config.translator,
            glossary_path=config.glossary_path,
            qwen_model_path=config.qwen_model_path,
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
            page_context_after = process_image(
                image_path,
                target_path,
                translator,
                config,
                chapter_context_before=chapter_context_before,
            )
            if page_context_after:
                chapter_context_before = page_context_after
            processed += 1
    finally:
        restore_env_var("TRANSFORMERS_OFFLINE", previous_transformers_offline)
        restore_env_var("HF_HUB_OFFLINE", previous_hf_offline)

    logger.info("Process folder finished: processed=%d skipped=%d", processed, skipped)


def process_folder_qwen_hybrid_batch(
    input_path: Path,
    output_path: Path,
    image_paths: list[Path],
    config: PipelineConfig,
) -> None:
    from .translate import build_translator

    logger.info(
        "Starting batched Qwen hybrid flow: primary=%s critic=%s fallback=%s",
        config.qwen_model_path,
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
        logger.info("Building primary Qwen translator for %d pending page(s)", len(primary_jobs))
        primary_translator = build_translator(
            "qwen",
            glossary_path=config.glossary_path,
            qwen_model_path=config.qwen_model_path,
        )
        for _page_index, page, primary_cache in tqdm(primary_jobs, desc="Q4 primary pass", unit="page"):
            translate_prepared_page(page, primary_translator, config)
            save_translation_cache(page, primary_cache)
        release_qwen_translator(primary_translator)
        del primary_translator
    else:
        logger.info("Primary Qwen pass skipped; all pages loaded from cache")

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
        logger.info("Primary Qwen pass complete; building critic Qwen translator for %d pending page(s)", len(critic_jobs))
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
        logger.info("Primary Qwen pass complete; building fallback Qwen translator for %d pending page(s)", len(fallback_jobs))
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
        "Batched Qwen hybrid flow finished: processed=%d skipped=%d fallback_attempted=%d fallback_accepted=%d",
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


def translation_cache_stage(config: PipelineConfig, stage: str) -> str:
    if config.translator == "qwen":
        suffix = f"{stage}-{config.qwen_mode}"
        if config.vision_facts_enabled:
            suffix += "-vision-facts"
        if config.qwen_critic_model_path is not None:
            suffix += "-critic"
        return suffix
    return stage


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
    logger.info(
        "Translation grouping complete: input_blocks=%d render_blocks=%d grouped_sets=%d",
        len(blocks),
        len(render_blocks),
        sum(1 for group in grouping_report if len(group["members"]) > 1),
    )
    return PreparedPage(
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
        context = {
            **context_for_block(block, page.page_order_report),
            **pre_translation_contexts.get(block.text, {}),
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
                context.update(translator.debug_info_for(block.text))
        elif config.translator == "qwen":
            translated = translator.translate_with_context(
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
                "translation_mode": "qwen_context",
            }
            if hasattr(translator, "debug_info_for"):
                context.update(translator.debug_info_for(block.text))
        else:
            translated = translator.translate(block.text)
        page.translations[block.text] = translated
        page.translation_contexts[block.text] = {
            **context,
            **translation_debug_info(block.text, translated, glossary_path=config.glossary_path),
        }
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
            page.translations[block.text] = translated
            page.translation_fallback_blocks.append(
                block_to_debug_dict(block, status="fallback", reason=reason, translated_text=translated)
            )
    logger.debug("Translation map contains %d unique source string(s)", len(page.translations))


def should_use_qwen_page_mode(translator, config: PipelineConfig) -> bool:
    return config.translator == "qwen" and config.qwen_mode == "page" and hasattr(translator, "translate_page")


def qwen_page_items(blocks: list[TextBlock], page_order_report: list[dict[str, object]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for fallback_index, block in enumerate(blocks, start=1):
        context = context_for_block(block, page_order_report)
        page_order = int(context.get("page_order") or fallback_index)
        items.append(
            {
                "id": page_order,
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
        translated = page.translations.get(block.text, "")
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
        page.translations[block.text] = fallback
        page.translation_fallback_blocks.append(
            block_to_debug_dict(block, status="fallback", reason=reason, translated_text=fallback)
        )


def render_prepared_page(page: PreparedPage, config: PipelineConfig) -> Path:
    from .erase import erase_text
    from .render import plan_render_layouts, plan_text_fits, render_translations
    from .vision_service import apply_vision_repair

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


def process_image(
    image_path: Path,
    output_path: Path,
    translator,
    config: PipelineConfig,
    *,
    chapter_context_before: str | None = None,
    qwen_fallback_translator=None,
) -> str | None:
    import cv2

    from .detect_ocr import run_ocr
    from .erase import erase_text
    from .render import plan_render_layouts, plan_text_fits, render_translations
    from .vision_service import apply_vision_repair

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
    logger.info(
        "Translation grouping complete: input_blocks=%d render_blocks=%d grouped_sets=%d",
        len(blocks),
        len(render_blocks),
        sum(1 for group in grouping_report if len(group["members"]) > 1),
    )

    logger.info("Translating %d text block(s)", len(render_blocks))
    from .translate import translation_debug_info

    translations: dict[str, str] = {}
    translation_contexts: dict[str, dict[str, object]] = {}
    translation_fallback_blocks: list[dict[str, object]] = []
    vision_facts_artifact = None
    if config.translator == "qwen" and config.vision_facts_enabled:
        from .vision_service import apply_vision_facts, release_qwen_vision_clients

        try:
            vision_facts_artifact = apply_vision_facts(
                image_bgr=image_bgr,
                blocks=render_blocks,
                translation_contexts=translation_contexts,
                page_order_report=page_order_report,
                output_path=output_path,
                config=config,
            )
        finally:
            release_qwen_vision_clients()
    page_translations_by_order: dict[int, str] | None = None
    if should_use_qwen_page_mode(translator, config):
        page_translations_by_order = translator.translate_page(
            qwen_page_items(render_blocks, page_order_report),
            previous_page_context=previous_page_context(page_order_report),
        )
    for block in render_blocks:
        context = {
            **context_for_block(block, page_order_report),
            **translation_contexts.get(block.text, {}),
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
                context.update(translator.debug_info_for(block.text))
        elif config.translator == "qwen":
            translated = translator.translate_with_context(
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
                "translation_mode": "qwen_context",
            }
            if hasattr(translator, "debug_info_for"):
                context.update(translator.debug_info_for(block.text))
        else:
            translated = translator.translate(block.text)
        translations[block.text] = translated
        translation_contexts[block.text] = {
            **context,
            **translation_debug_info(block.text, translated, glossary_path=config.glossary_path),
        }
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
            translations[block.text] = translated
            translation_fallback_blocks.append(
                block_to_debug_dict(block, status="fallback", reason=reason, translated_text=translated)
            )
    if config.translator == "qwen" and qwen_fallback_translator is not None:
        apply_qwen_fallback_translations(
            render_blocks,
            translations,
            translation_contexts,
            page_order_report,
            qwen_fallback_translator,
            config,
        )
    logger.debug("Translation map contains %d unique source string(s)", len(translations))
    render_layouts = plan_render_layouts(
        image_bgr,
        render_blocks,
        render_expand=config.render_expand,
    )
    render_fits = plan_text_fits(
        image_bgr,
        render_blocks,
        translations,
        font_path=config.font_path,
        base_font_size=config.base_font_size,
        render_expand=config.render_expand,
        render_layouts=render_layouts,
    )
    compact_translations_for_render(
        image_bgr,
        render_blocks,
        translations,
        render_layouts=render_layouts,
        render_fits=render_fits,
        font_path=config.font_path,
        base_font_size=config.base_font_size,
        render_expand=config.render_expand,
    )
    render_fits = plan_text_fits(
        image_bgr,
        render_blocks,
        translations,
        font_path=config.font_path,
        base_font_size=config.base_font_size,
        render_expand=config.render_expand,
        render_layouts=render_layouts,
    )
    vision_artifact = apply_vision_repair(
        image_bgr=image_bgr,
        blocks=render_blocks,
        translations=translations,
        translation_contexts=translation_contexts,
        page_order_report=page_order_report,
        render_layouts=render_layouts,
        render_fits=render_fits,
        output_path=output_path,
        config=config,
    )
    if any(context.get("vision_accepted") is True for context in translation_contexts.values()):
        render_fits = plan_text_fits(
            image_bgr,
            render_blocks,
            translations,
            font_path=config.font_path,
            base_font_size=config.base_font_size,
            render_expand=config.render_expand,
            render_layouts=render_layouts,
        )
        compact_translations_for_render(
            image_bgr,
            render_blocks,
            translations,
            translation_contexts=translation_contexts,
            render_layouts=render_layouts,
            render_fits=render_fits,
            font_path=config.font_path,
            base_font_size=config.base_font_size,
            render_expand=config.render_expand,
        )
        render_fits = plan_text_fits(
            image_bgr,
            render_blocks,
            translations,
            font_path=config.font_path,
            base_font_size=config.base_font_size,
            render_expand=config.render_expand,
            render_layouts=render_layouts,
        )
    if config.debug:
        write_debug_report(
            image_path,
            output_path,
            config,
            raw_blocks,
            render_blocks,
            translations,
            skipped_blocks,
            translation_fallback_blocks,
            grouping_report,
            render_layouts,
            render_fits,
            width,
            height,
            config.padding,
            page_order_report,
            translation_contexts,
            vision_artifact=vision_artifact,
            vision_facts_artifact=vision_facts_artifact,
        )
    erase_blocks = blocks_for_erasing(render_blocks)
    cleaned = erase_text(
        image_bgr,
        erase_blocks,
        mode=config.erase_mode,
        padding=config.padding,
    )
    rendered = render_translations(
        cleaned,
        render_blocks,
        translations,
        font_path=config.font_path,
        base_font_size=config.base_font_size,
        render_expand=config.render_expand,
        render_layouts=render_layouts,
    )

    actual_output_path = write_image_with_fallback(rendered, output_path)
    logger.info("Translated image written: %s", actual_output_path)

    if config.debug:
        logger.info("Writing debug OCR box image")
        write_debug_image(
            image_bgr,
            render_blocks,
            skipped_blocks,
            translation_fallback_blocks,
            render_layouts,
            actual_output_path,
        )
    return str(page_order_report[-1]["source_text"]) if page_order_report else chapter_context_before


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


def apply_qwen_fallback_translations(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    page_order_report: list[dict[str, object]],
    qwen_fallback_translator,
    config: PipelineConfig,
) -> tuple[int, int]:
    from .translate import translation_debug_info

    attempted = 0
    accepted = 0
    for block in blocks:
        primary_translation = translations.get(block.text, "")
        primary_context = translation_contexts.get(block.text, {})
        if not should_try_qwen_fallback(block.text, primary_translation, primary_context):
            continue

        attempted += 1
        context = context_for_block(block, page_order_report)
        logger.info(
            "Trying Qwen fallback model: box=%s source=%s primary=%s",
            block.box,
            shorten(block.text),
            shorten(primary_translation),
        )
        fallback_translation = qwen_fallback_translator.translate_with_context(
            block.text,
            before=str(context.get("context_before") or "") or None,
            after=str(context.get("context_after") or "") or None,
            before_contexts=tuple(str(value) for value in context.get("context_before_window") or []),
            after_contexts=tuple(str(value) for value in context.get("context_after_window") or []),
            visual_facts=visual_facts_for_context(primary_context),
        )
        fallback_debug = (
            qwen_fallback_translator.debug_info_for(block.text)
            if hasattr(qwen_fallback_translator, "debug_info_for")
            else {}
        )
        reject_reason = fallback_reject_reason(block.text, fallback_translation, fallback_debug)
        primary_context.update(
            {
                "qwen_fallback_attempted": True,
                "qwen_fallback_source_translation": primary_translation,
                "qwen_fallback_translation": fallback_translation,
                "qwen_fallback_debug": fallback_debug,
                "qwen_fallback_accepted": reject_reason is None,
            }
        )
        if reject_reason is not None:
            primary_context["qwen_fallback_reject_reason"] = reject_reason
            translation_contexts[block.text] = primary_context
            logger.info(
                "Rejected Qwen fallback translation: reason=%s source=%s fallback=%s",
                reject_reason,
                shorten(block.text),
                shorten(fallback_translation),
            )
            continue

        translations[block.text] = fallback_translation
        primary_context.update(
            {
                "qwen_hybrid_used": True,
                "qwen_final_model": fallback_debug.get("qwen_model", "fallback"),
                **translation_debug_info(block.text, fallback_translation, glossary_path=config.glossary_path),
            }
        )
        translation_contexts[block.text] = primary_context
        accepted += 1
        logger.info(
            "Accepted Qwen fallback translation: source=%s fallback=%s",
            shorten(block.text),
            shorten(fallback_translation),
        )

    if attempted:
        logger.info("Qwen fallback pass finished: attempted=%d accepted=%d", attempted, accepted)
    return attempted, accepted


def apply_qwen_critic_reviews(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    page_order_report: list[dict[str, object]],
    qwen_critic_translator,
    config: PipelineConfig,
) -> tuple[int, int]:
    attempted = 0
    flagged = 0
    for block in blocks:
        current_translation = translations.get(block.text, "")
        primary_context = translation_contexts.get(block.text, {})
        trigger_reasons = qwen_critic_trigger_reasons(block.text, current_translation, primary_context)
        if not trigger_reasons:
            continue

        attempted += 1
        context = context_for_block(block, page_order_report)
        logger.info(
            "Trying Qwen critic model: reasons=%s source=%s current=%s",
            ",".join(trigger_reasons),
            shorten(block.text),
            shorten(current_translation),
        )
        decision, critic_debug = qwen_critic_translator.critique_translation(
            block.text,
            current_translation=current_translation,
            before=str(context.get("context_before") or "") or None,
            after=str(context.get("context_after") or "") or None,
            before_contexts=tuple(str(value) for value in context.get("context_before_window") or []),
            after_contexts=tuple(str(value) for value in context.get("context_after_window") or []),
            baseline=str(primary_context.get("qwen_baseline") or ""),
            trigger_reasons=trigger_reasons,
            visual_facts=visual_facts_for_context(primary_context),
        )
        should_repair = qwen_critic_should_trigger_repair(decision)
        if should_repair:
            flagged += 1
        primary_context.update(
            {
                **critic_debug,
                "qwen_critic_flagged": should_repair,
                "qwen_critic_should_repair": should_repair,
            }
        )
        translation_contexts[block.text] = primary_context
        logger.info(
            "Qwen critic result: severity=%s issues=%s should_repair=%s source=%s reason=%s",
            decision.severity,
            ",".join(decision.issues),
            should_repair,
            shorten(block.text),
            shorten(decision.reason),
        )

    if attempted:
        logger.info("Qwen critic pass finished: attempted=%d flagged=%d", attempted, flagged)
    return attempted, flagged


def qwen_critic_trigger_reasons(
    source_text: str,
    translated_text: str,
    primary_context: dict[str, object],
) -> tuple[str, ...]:
    if primary_context.get("qwen_rejected") is True:
        return ()
    if unusable_translation_reason(source_text, translated_text, translator_name="qwen") is not None:
        return ()

    reasons: list[str] = []
    if suspected_bad_translation(translated_text):
        reasons.append("suspected_bad_translation")
    if primary_context.get("qwen_repair_used") is True:
        reasons.append("qwen_repair_used")
    if primary_context.get("qwen_unsupported_terms"):
        reasons.append("qwen_unsupported_terms")
    if has_critic_risk_source_terms(source_text):
        reasons.append("risk_source_terms")
    if has_awkward_english_pattern(translated_text):
        reasons.append("awkward_english_pattern")
    return tuple(dict.fromkeys(reasons))


def has_critic_risk_source_terms(source_text: str) -> bool:
    if ("\u7236" in source_text and "\u6bcd" in source_text) or "\u3061\u3061\u3082\u306f\u306f\u3082" in source_text:
        return True
    risk_terms = (
        "\u7d44\u7e54",
        "\u5b50\u5206",
        "\u30aa\u30da\u30ec\u30fc\u30b7\u30e7\u30f3",
        "\u3008",
    )
    return any(term in source_text for term in risk_terms)


def has_awkward_english_pattern(translated_text: str) -> bool:
    normalized = translated_text.lower()
    patterns = (
        r"\b(?:father|mother|dad|mom|papa|mama)\s+and\s+(?:father|mother|dad|mom|papa|mama)\s+are\s+hated\b",
        r"\b[a-z]+ and [a-z]+ are hated\b",
        r"\bwatch my organization\b",
        r"\bson of a bitch\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def qwen_critic_should_trigger_repair(decision) -> bool:
    return not bool(getattr(decision, "ok", False)) and str(getattr(decision, "severity", "")) in {"medium", "high"}


def should_try_qwen_fallback(
    source_text: str,
    translated_text: str,
    primary_context: dict[str, object],
) -> bool:
    if primary_context.get("qwen_critic_should_repair") is True:
        return True
    if primary_context.get("qwen_rejected") is True:
        return True
    if primary_context.get("qwen_repair_used") is True and primary_context.get("qwen_repair_accepted") is not True:
        return True
    if suspected_bad_translation(translated_text):
        return True
    return unusable_translation_reason(source_text, translated_text, translator_name="qwen") is not None


def fallback_reject_reason(
    source_text: str,
    translated_text: str,
    fallback_debug: dict[str, object],
) -> str | None:
    if fallback_debug.get("qwen_rejected") is True:
        return f"fallback_qwen_rejected:{fallback_debug.get('qwen_reject_reason', 'unknown')}"
    if suspected_bad_translation(translated_text):
        return "fallback_suspected_bad_translation"
    unusable_reason = unusable_translation_reason(source_text, translated_text, translator_name="qwen")
    if unusable_reason:
        return f"fallback_unusable:{unusable_reason}"
    return None


def visual_facts_for_context(context: dict[str, object]) -> tuple[str, ...]:
    if context.get("vision_facts_accepted") is not True:
        return ()
    facts: list[str] = []
    bubble_type = str(context.get("bubble_type") or "").strip()
    speaker_position = str(context.get("speaker_position") or "").strip()
    visible_emotion = str(context.get("visible_emotion") or "").strip()
    observable_action = str(context.get("observable_action") or "").strip()
    if bubble_type and bubble_type != "unknown":
        facts.append(f"Bubble type: {bubble_type}")
    if speaker_position and speaker_position != "unknown":
        facts.append(f"Speaker position: {speaker_position}")
    if visible_emotion and visible_emotion != "unknown":
        facts.append(f"Visible emotion: {visible_emotion}")
    if observable_action and observable_action != "unknown":
        facts.append(f"Observable action: {observable_action}")
    for value in context.get("visual_facts") or []:
        text = str(value).strip()
        if text:
            facts.append(text)
    return tuple(dict.fromkeys(facts))


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
        text = translations.get(block.text, "")
        if not text:
            continue
        if getattr(fit, "font_size", 99) > 7 and not getattr(fit, "clipped", False):
            continue
        shortened = compact_english_for_bubble(text)
        if shortened == text:
            continue
        translations[block.text] = shortened
        if translation_contexts is not None:
            translation_contexts.setdefault(block.text, {})["compacted_for_render"] = True
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
