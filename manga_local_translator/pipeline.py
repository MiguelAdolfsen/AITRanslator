from __future__ import annotations

import logging
import os
import re
import hashlib
import subprocess
from dataclasses import replace
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
    context_for_block,
)
from .image_io import iter_images, resolve_output_path, write_image_with_fallback
from .line_identity import (
    enrich_grouping_report,
    enrich_page_order_report,
    lookup_context,
    lookup_translation,
    set_context,
    set_translation,
    translation_for_page_order_item,
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
    count_japanese_chars,
    fallback_translation,
    is_box_like,
    suspected_bad_translation,
    unusable_translation_reason,
)
from .translation_evidence import (
    ConsistencyMemory,
    analyze_translation_evidence,
    build_consistency_memory,
    build_evidence_context,
)
from .qwen_validation import (
    source_has_bracket_term,
    translation_preserves_bracket_term,
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

        if config.translator == "cat" or config.translator == "qwen" and (
            config.qwen_fallback_model_path is not None
            or config.qwen_critic_model_path is not None
            or (config.translator == "qwen" and config.vision_facts_enabled)
        ):
            process_folder_qwen_hybrid_batch(input_path, output_path, image_paths, config)
            return

        translator = build_translator(
            config.translator,
            glossary_path=config.glossary_path,
            qwen_model_path=config.qwen_model_path,
            cat_model_name=config.cat_model_name,
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


def render_only_translation_stages(config: PipelineConfig) -> list[str]:
    stage_names: list[str] = []
    if config.qwen_fallback_model_path is not None:
        stage_names.append("hybrid")
    if config.qwen_critic_model_path is not None:
        stage_names.append("critic")
    stage_names.append("primary")
    if config.qwen_fallback_model_path is None:
        stage_names.append("hybrid")
    if config.qwen_critic_model_path is None:
        stage_names.append("critic")
    seen: set[str] = set()
    stages: list[str] = []
    for stage in stage_names:
        cache_stage = translation_cache_stage(config, stage)
        if cache_stage not in seen:
            seen.add(cache_stage)
            stages.append(cache_stage)
    return stages


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
        config.qwen_model_path if config.translator == "qwen" else config.cat_model_name,
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


def translation_cache_stage(config: PipelineConfig, stage: str) -> str:
    if config.translator == "qwen":
        suffix = f"{stage}-{config.qwen_mode}"
    elif config.translator == "cat":
        from .hf_translators import (
            CAT_BYPASS_VERSION,
            CAT_PROMPT_VERSION,
            CAT_RETRY_PROMPT_VERSION,
            CAT_SECOND_RETRY_PROMPT_VERSION,
            CAT_VALIDATION_VERSION,
            cat_bypass_enabled,
            cat_num_predict,
            cat_second_retry_enabled,
            resolve_cat_gguf_path,
        )

        cat_model = str(config.cat_model_name or "")
        cat_path = resolve_cat_gguf_path(Path(cat_model)) if cat_model else resolve_cat_gguf_path("cyberagent/CAT-Translate-7b")
        model_identity = str(cat_path or cat_model or "default")
        backend_identity = "ollama" if cat_path else "transformers"
        identity = cache_identity_token(
            f"{model_identity}|{backend_identity}|{CAT_PROMPT_VERSION}|{CAT_RETRY_PROMPT_VERSION}|{CAT_SECOND_RETRY_PROMPT_VERSION}|{CAT_VALIDATION_VERSION}|{CAT_BYPASS_VERSION}|bypass={cat_bypass_enabled()}|second_retry={cat_second_retry_enabled()}|num_predict={cat_num_predict()}"
        )
        bypass_token = "bypass" if cat_bypass_enabled() else "nobypass"
        second_retry_token = "secondretry" if cat_second_retry_enabled() else "nosecondretry"
        suffix = f"{stage}-cat-{identity}-{backend_identity}-{CAT_PROMPT_VERSION}-{CAT_RETRY_PROMPT_VERSION}-{CAT_SECOND_RETRY_PROMPT_VERSION}-{CAT_VALIDATION_VERSION}-{CAT_BYPASS_VERSION}-{bypass_token}-{second_retry_token}-np{cat_num_predict()}"
    else:
        return stage
    if config.qwen_critic_model_path is not None or config.qwen_fallback_model_path is not None:
        suffix += "-evidence"
    if config.vision_facts_enabled:
        suffix += "-vision-facts"
    if config.qwen_critic_model_path is not None:
        suffix += "-critic"
    return suffix


def cache_identity_token(value: str) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]
    return digest


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


def apply_translation_evidence_to_pages(pages: list[PreparedPage]) -> ConsistencyMemory:
    memory = build_consistency_memory(
        (block.text, lookup_translation(page.translations, block, ""))
        for page in pages
        for block in page.render_blocks
    )
    total_risks = 0
    total_repairs = 0
    for page in pages:
        apply_translation_evidence(
            page.render_blocks,
            page.translations,
            page.translation_contexts,
            memory,
        )
        total_risks += sum(1 for context in page.translation_contexts.values() if context.get("evidence_risk_flags"))
        total_repairs += sum(1 for context in page.translation_contexts.values() if context.get("evidence_repair_reasons"))
    logger.info(
        "Translation evidence prepared: pages=%d memory_terms=%d risk_blocks=%d repair_reason_blocks=%d",
        len(pages),
        len(memory.entries),
        total_risks,
        total_repairs,
    )
    return memory


def apply_translation_evidence(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    memory: ConsistencyMemory | None,
) -> None:
    for block in blocks:
        translated = lookup_translation(translations, block, "")
        context = lookup_context(translation_contexts, block)
        context.update(build_evidence_context(block.text, translated, memory))
        set_context(translation_contexts, block, context)


def retry_cat_failures(page: PreparedPage, translator, config: PipelineConfig) -> tuple[int, int]:
    if not hasattr(translator, "retry_translation"):
        return (0, 0)
    from .translate import translation_debug_info

    attempted = 0
    accepted = 0
    suspect_attempted = 0
    suspect_accepted = 0
    for block in page.render_blocks:
        context = lookup_context(page.translation_contexts, block)
        previous_translation = lookup_translation(page.translations, block, "")
        if context.get("cat_rejected") is not True and not should_retry_suspected_cat_translation(
            block.text,
            previous_translation,
            context,
            translator,
        ):
            continue
        attempted += 1
        previous_reason = str(context.get("cat_reject_reason") or "")
        if context.get("cat_rejected") is True:
            logger.info(
                "Retrying CAT rejected line: reason=%s source=%s",
                previous_reason,
                shorten(block.text),
            )
            retry_translation = translator.retry_translation(block.text)
            retry_mode = "rejected"
        else:
            suspect_attempted += 1
            logger.info(
                "Retrying suspected bad CAT line with strict second pass: translation=%s source=%s",
                shorten(previous_translation),
                shorten(block.text),
            )
            retry_translation = translator.second_retry_translation(block.text, previous_translation)
            retry_mode = "suspected_bad"
        retry_debug = translator_debug_info_for(translator, block)
        context.update(
            {
                "cat_retry_attempted": True,
                "cat_retry_previous_translation": previous_translation,
                "cat_retry_previous_reject_reason": previous_reason,
                "cat_retry_mode": retry_mode,
                **retry_debug,
            }
        )
        if retry_debug.get("cat_retry_accepted") is True or retry_debug.get("cat_suspect_second_pass_accepted") is True:
            if retry_debug.get("cat_suspect_second_pass_accepted") is True:
                suspect_accepted += 1
            set_translation(page.translations, block, retry_translation)
            context.update(translation_debug_info(block.text, retry_translation, glossary_path=config.glossary_path))
            accepted += 1
            logger.info(
                "Accepted CAT retry translation: source=%s retry=%s",
                shorten(block.text),
                shorten(retry_translation),
            )
        else:
            logger.info(
                "Rejected CAT retry translation: source=%s reason=%s",
                shorten(block.text),
                retry_debug.get("cat_retry_reject_reason")
                or retry_debug.get("cat_suspect_second_pass_reject_reason")
                or "unknown",
            )
        set_context(page.translation_contexts, block, context)
    if attempted:
        logger.info(
            "CAT retry pass finished for %s: attempted=%d accepted=%d suspect_attempted=%d suspect_accepted=%d",
            page.image_path,
            attempted,
            accepted,
            suspect_attempted,
            suspect_accepted,
        )
    return attempted, accepted


def should_retry_suspected_cat_translation(
    source_text: str,
    translated_text: str,
    context: dict[str, object],
    translator,
) -> bool:
    if not hasattr(translator, "second_retry_translation"):
        return False
    if context.get("cat_used") is not True:
        return False
    if context.get("cat_reason") == "phrasebook":
        return False
    if context.get("cat_retry_attempted") is True or context.get("cat_suspect_second_pass_attempted") is True:
        return False
    if context.get("cat_rejected") is True:
        return False
    if unusable_translation_reason(source_text, translated_text, translator_name="cat") is not None:
        return True
    return suspected_bad_translation(translated_text)


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
        elif config.translator == "qwen":
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
                "translation_mode": "qwen_context",
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
            **lookup_context(translation_contexts, block),
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
        elif config.translator == "qwen":
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
                "translation_mode": "qwen_context",
            }
            if hasattr(translator, "debug_info_for"):
                context.update(translator_debug_info_for(translator, block))
        else:
            translated = translator.translate(block.text)
            if hasattr(translator, "debug_info_for"):
                context.update(translator_debug_info_for(translator, block))
        set_translation(translations, block, translated)
        set_context(translation_contexts, block, {
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
            set_translation(translations, block, translated)
            translation_fallback_blocks.append(
                block_to_debug_dict(block, status="fallback", reason=reason, translated_text=translated)
            )
    if config.translator in {"qwen", "cat"} and qwen_fallback_translator is not None:
        evidence_memory = build_consistency_memory((block.text, lookup_translation(translations, block, "")) for block in render_blocks)
        apply_translation_evidence(render_blocks, translations, translation_contexts, evidence_memory)
        apply_qwen_fallback_translations(
            render_blocks,
            translations,
            translation_contexts,
            page_order_report,
            qwen_fallback_translator,
            config,
            evidence_memory=evidence_memory,
        )
    logger.debug("Translation map contains %d line id(s)", len(translations))
    if config.skip_render:
        if config.debug:
            render_layouts = [TranslationOnlyLayout(block.box) for block in render_blocks]
            render_fits = [None for _block in render_blocks]
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
                vision_artifact=None,
                vision_facts_artifact=vision_facts_artifact,
            )
        logger.info("Skipped translated image rendering for benchmark: %s", output_path)
        return page_order_report[-1]["source_text"] if page_order_report else None
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
    *,
    evidence_memory: ConsistencyMemory | None = None,
) -> tuple[int, int]:
    from .translate import translation_debug_info

    attempted = 0
    accepted = 0
    for block in blocks:
        primary_translation = lookup_translation(translations, block, "")
        primary_context = lookup_context(translation_contexts, block)
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
        before_contexts = tuple(str(value) for value in context.get("context_before_window") or [])
        after_contexts = tuple(str(value) for value in context.get("context_after_window") or [])
        before_translations, after_translations = english_context_for_block(block, translations, page_order_report)
        guided_by_critic = primary_context.get("qwen_critic_should_repair") is True
        guided_by_evidence = bool(primary_context.get("evidence_repair_reasons"))
        use_guided_repair = (guided_by_critic or guided_by_evidence) and hasattr(qwen_fallback_translator, "repair_translation_with_guidance")
        if use_guided_repair:
            fallback_translation = qwen_fallback_translator.repair_translation_with_guidance(
                block.text,
                current_translation=primary_translation,
                before=str(context.get("context_before") or "") or None,
                after=str(context.get("context_after") or "") or None,
                before_contexts=before_contexts,
                after_contexts=after_contexts,
                before_translations=before_translations,
                after_translations=after_translations,
                baseline=str(primary_context.get("qwen_baseline") or ""),
                critic_issues=tuple(str(value) for value in primary_context.get("qwen_critic_issues") or []),
                critic_reason=str(primary_context.get("qwen_critic_reason") or ""),
                visual_facts=visual_facts_for_context(primary_context),
                source_features=primary_context.get("source_features") if isinstance(primary_context.get("source_features"), dict) else None,
                translation_evidence=primary_context.get("translation_evidence") if isinstance(primary_context.get("translation_evidence"), dict) else None,
                consistency_memory=tuple(
                    entry for entry in primary_context.get("consistency_memory_entries", []) if isinstance(entry, dict)
                ),
                critic_source_evidence=tuple(str(value) for value in primary_context.get("qwen_critic_source_evidence") or []),
                critic_translation_evidence=tuple(str(value) for value in primary_context.get("qwen_critic_translation_evidence") or []),
            )
        else:
            fallback_translation = translate_with_line_context(
                qwen_fallback_translator,
                block,
                block.text,
                before=str(context.get("context_before") or "") or None,
                after=str(context.get("context_after") or "") or None,
                before_contexts=before_contexts,
                after_contexts=after_contexts,
                visual_facts=visual_facts_for_context(primary_context),
            )
        fallback_debug = (
            translator_debug_info_for(qwen_fallback_translator, block)
            if hasattr(qwen_fallback_translator, "debug_info_for")
            else {}
        )
        fallback_memory_entries = evidence_memory.entries_for_source(block.text, fallback_translation) if evidence_memory is not None else []
        fallback_evidence = analyze_translation_evidence(block.text, fallback_translation, memory_entries=fallback_memory_entries)
        reject_reason = fallback_reject_reason(block.text, fallback_translation, fallback_debug, fallback_evidence=fallback_evidence)
        primary_context.update(
            {
                "qwen_fallback_attempted": True,
                "qwen_fallback_source_translation": primary_translation,
                "qwen_fallback_translation": fallback_translation,
                "qwen_fallback_debug": fallback_debug,
                "qwen_fallback_accepted": reject_reason is None,
                "qwen_fallback_guided_by_critic": guided_by_critic,
                "qwen_fallback_guided_by_evidence": guided_by_evidence,
                "qwen_fallback_before_translations": list(before_translations),
                "qwen_fallback_after_translations": list(after_translations),
                "qwen_fallback_translation_evidence": fallback_evidence,
                "qwen_fallback_consistency_memory_entries": fallback_memory_entries,
                "cat_q8_fallback_attempted": primary_context.get("primary_translator") == "cat" or primary_context.get("cat_used") is True,
            }
        )
        if reject_reason is not None:
            primary_context["qwen_fallback_reject_reason"] = reject_reason
            set_context(translation_contexts, block, primary_context)
            logger.info(
                "Rejected Qwen fallback translation: reason=%s source=%s fallback=%s",
                reject_reason,
                shorten(block.text),
                shorten(fallback_translation),
            )
            continue

        set_translation(translations, block, fallback_translation)
        primary_context.update(
            {
                "qwen_hybrid_used": True,
                "qwen_final_model": fallback_debug.get("qwen_model", "fallback"),
                "cat_q8_fallback_accepted": primary_context.get("primary_translator") == "cat" or primary_context.get("cat_used") is True,
                **translation_debug_info(block.text, fallback_translation, glossary_path=config.glossary_path),
            }
        )
        set_context(translation_contexts, block, primary_context)
        accepted += 1
        logger.info(
            "Accepted Qwen fallback translation: source=%s fallback=%s",
            shorten(block.text),
            shorten(fallback_translation),
        )

    if attempted:
        logger.info("Qwen fallback pass finished: attempted=%d accepted=%d", attempted, accepted)
    return attempted, accepted


def english_context_for_block(
    block: TextBlock,
    translations: dict[str, str],
    page_order_report: list[dict[str, object]],
    *,
    window: int = 3,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    target_index = None
    for index, item in enumerate(page_order_report):
        if str(item.get("line_id", "") or "") and str(item.get("line_id", "")) == str(block.metadata.get("line_id", "")):
            target_index = index
            break
        if tuple(item.get("box", ())) == tuple(block.box) and str(item.get("source_text", "")) == block.text:
            target_index = index
            break
    if target_index is None:
        return (), ()

    before: list[str] = []
    for item in page_order_report[max(0, target_index - window):target_index]:
        translated = translation_for_page_order_item(translations, item)
        if translated and translated != "...":
            before.append(translated)

    after: list[str] = []
    for item in page_order_report[target_index + 1:target_index + 1 + window]:
        translated = translation_for_page_order_item(translations, item)
        if translated and translated != "...":
            after.append(translated)

    return tuple(before), tuple(after)


def apply_qwen_critic_reviews(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    page_order_report: list[dict[str, object]],
    qwen_critic_translator,
    config: PipelineConfig,
    *,
    evidence_memory: ConsistencyMemory | None = None,
) -> tuple[int, int]:
    attempted = 0
    flagged = 0
    for block in blocks:
        current_translation = lookup_translation(translations, block, "")
        primary_context = lookup_context(translation_contexts, block)
        if "source_features" not in primary_context or "translation_evidence" not in primary_context:
            primary_context.update(build_evidence_context(block.text, current_translation, evidence_memory))
            set_context(translation_contexts, block, primary_context)
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
            source_features=primary_context.get("source_features") if isinstance(primary_context.get("source_features"), dict) else None,
            translation_evidence=primary_context.get("translation_evidence") if isinstance(primary_context.get("translation_evidence"), dict) else None,
            consistency_memory=tuple(
                entry for entry in primary_context.get("consistency_memory_entries", []) if isinstance(entry, dict)
            ),
        )
        repair_gate = qwen_critic_repair_decision(
            decision,
            source_text=block.text,
            translated_text=current_translation,
            translation_evidence=primary_context.get("translation_evidence") if isinstance(primary_context.get("translation_evidence"), dict) else None,
        )
        should_repair = bool(repair_gate["should_repair"])
        if should_repair:
            flagged += 1
        primary_context.update(
            {
                **critic_debug,
                "qwen_critic_flagged": should_repair,
                "qwen_critic_should_repair": should_repair,
                "qwen_critic_effective_issues": repair_gate["effective_issues"],
                "qwen_critic_repair_gate_reason": repair_gate["reason"],
                "qwen_critic_evidence_gate_reason": repair_gate["reason"],
            }
        )
        set_context(translation_contexts, block, primary_context)
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
    if primary_context.get("qwen_used") is not True and primary_context.get("cat_used") is not True:
        return ()
    if primary_context.get("qwen_reason") == "phrasebook" or primary_context.get("cat_reason") == "phrasebook":
        return ()
    if unusable_translation_reason(source_text, translated_text, translator_name="qwen") is not None:
        return ()
    if is_probable_standalone_japanese_name_or_credit(source_text):
        return ()
    if is_probable_short_sfx_or_reaction(source_text):
        return ()

    reasons: list[str] = ["accepted_line_review"]
    if suspected_bad_translation(translated_text):
        reasons.append("suspected_bad_translation")
    if primary_context.get("qwen_repair_used") is True:
        reasons.append("qwen_repair_used")
    if primary_context.get("qwen_unsupported_terms"):
        reasons.append("qwen_unsupported_terms")
    if primary_context.get("evidence_risk_flags"):
        reasons.append("evidence_risk")
    if primary_context.get("evidence_repair_reasons"):
        reasons.append("evidence_repair_reason")
    if has_critic_risk_source_terms(source_text):
        reasons.append("risk_source_terms")
    if has_awkward_english_pattern(translated_text):
        reasons.append("awkward_english_pattern")
    return tuple(dict.fromkeys(reasons))


def has_critic_risk_source_terms(source_text: str) -> bool:
    return "\u3008" in source_text


def is_probable_standalone_japanese_name_or_credit(source_text: str) -> bool:
    compact = re.sub(r"\s+", "", source_text)
    return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,5}", compact))


def is_probable_short_sfx_or_reaction(source_text: str) -> bool:
    compact = re.sub(r"\s+", "", source_text)
    compact = compact.strip("\u300c\u300d\u300e\u300f")
    if len(compact) > 5:
        return False
    return bool(re.fullmatch(r"[\u3040-\u30ff\u31f0-\u31ff\u2026.!?\uff01\uff1f\u30fc]+", compact))


def has_awkward_english_pattern(translated_text: str) -> bool:
    normalized = translated_text.lower()
    patterns = (
        r"\b(?:father|mother|dad|mom|papa|mama)\s+and\s+(?:father|mother|dad|mom|papa|mama)\s+are\s+hated\b",
        r"\b[a-z]+ and [a-z]+ are hated\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def qwen_critic_should_trigger_repair(
    decision,
    *,
    source_text: str = "",
    translated_text: str = "",
    translation_evidence: dict[str, object] | None = None,
) -> bool:
    return bool(
        qwen_critic_repair_decision(
            decision,
            source_text=source_text,
            translated_text=translated_text,
            translation_evidence=translation_evidence,
        )["should_repair"]
    )


def qwen_critic_repair_decision(
    decision,
    *,
    source_text: str = "",
    translated_text: str = "",
    translation_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    if bool(getattr(decision, "ok", False)):
        return {"should_repair": False, "effective_issues": [], "reason": "critic_ok"}
    severity = str(getattr(decision, "severity", ""))
    issues = set(getattr(decision, "issues", ()) or ())
    hard_issues = {
        "context_mismatch",
        "omitted_term",
        "invented_detail",
        "name_drift",
        "untranslated_text",
        "glossary_conflict",
        "grammar_problem",
    }
    effective_issues = [
        issue
        for issue in sorted(issues & hard_issues)
        if critic_issue_has_local_repair_evidence(
            issue,
            source_text=source_text,
            translated_text=translated_text,
            translation_evidence=translation_evidence,
        )
    ]
    should_repair = bool(effective_issues)
    reason = "hard_issue_supported" if should_repair else "no_local_repair_evidence"
    if severity == "high" and effective_issues:
        reason = "high_severity_supported"
    return {
        "should_repair": should_repair,
        "effective_issues": effective_issues,
        "reason": reason,
    }


def critic_issue_has_local_repair_evidence(
    issue: str,
    *,
    source_text: str,
    translated_text: str,
    translation_evidence: dict[str, object] | None = None,
) -> bool:
    evidence_failures = set(str(value) for value in (translation_evidence or {}).get("preservation_failures", []))
    if issue == "omitted_term":
        return has_local_omission_evidence(source_text, translated_text) or bool(
            evidence_failures & {"dropped_number", "dropped_bracket_term"}
        )
    if issue == "invented_detail":
        return has_local_invention_evidence(translated_text) or "invented_english_name_on_noisy_source" in evidence_failures
    if issue == "name_drift":
        return has_local_name_drift_evidence(source_text, translated_text) or "consistency_conflict" in evidence_failures
    if issue == "untranslated_text":
        return has_local_untranslated_text_evidence(translated_text) or "untranslated_japanese" in evidence_failures
    if issue == "context_mismatch":
        return "consistency_conflict" in evidence_failures
    if issue == "grammar_problem":
        return "broken_english" in evidence_failures
    if issue in {"glossary_conflict", "grammar_problem"}:
        return True
    return False


def has_local_omission_evidence(source_text: str, translated_text: str) -> bool:
    if source_has_bracket_term(source_text) and not translation_preserves_bracket_term(source_text, translated_text):
        return True
    return False


def has_local_invention_evidence(translated_text: str) -> bool:
    return suspected_bad_translation(translated_text)


def has_local_name_drift_evidence(source_text: str, translated_text: str) -> bool:
    return False


def has_local_untranslated_text_evidence(translated_text: str) -> bool:
    if count_japanese_chars(translated_text) > 0:
        return True
    return bool(re.search(r"\b(?:wo|wa|ga|desu|kudasai|mite|shite|suru)\b", translated_text, flags=re.IGNORECASE))


def should_try_qwen_fallback(
    source_text: str,
    translated_text: str,
    primary_context: dict[str, object],
) -> bool:
    if primary_context.get("cat_rejected") is True:
        return True
    if primary_context.get("cat_bypassed") is True:
        return True
    if primary_context.get("qwen_critic_should_repair") is True:
        return True
    if primary_context.get("evidence_repair_reasons"):
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
    *,
    fallback_evidence: dict[str, object] | None = None,
) -> str | None:
    if fallback_debug.get("qwen_rejected") is True:
        return f"fallback_qwen_rejected:{fallback_debug.get('qwen_reject_reason', 'unknown')}"
    if suspected_bad_translation(translated_text):
        return "fallback_suspected_bad_translation"
    unusable_reason = unusable_translation_reason(source_text, translated_text, translator_name="qwen")
    if unusable_reason:
        return f"fallback_unusable:{unusable_reason}"
    evidence_failures = set(str(value) for value in (fallback_evidence or {}).get("preservation_failures", []))
    blocking_failures = {
        "dropped_number",
        "dropped_bracket_term",
        "invented_english_name_on_noisy_source",
        "broken_english",
        "untranslated_japanese",
    }
    blocking = sorted(evidence_failures & blocking_failures)
    if blocking:
        return f"fallback_evidence_failure:{','.join(blocking)}"
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
