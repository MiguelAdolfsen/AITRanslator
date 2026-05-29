from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Callable, Sequence

from .config import PipelineConfig

logger = logging.getLogger(__name__)

TranslationCacheLoader = Callable[[object, Path], None]


@dataclass(frozen=True)
class HybridResumeCacheJobs:
    primary_jobs: list[tuple[int, object, Path]]
    hybrid_cached_pages: set[int]
    critic_cached_pages: set[int]


@dataclass(frozen=True)
class TranslationCachePlan:
    primary_stage: str
    critic_stage: str
    hybrid_stage: str
    resume_lookup_order: tuple[str, ...]
    render_only_lookup_order: tuple[str, ...]
    final_stage: str

    def stage(self, stage: str) -> str:
        if stage == "primary":
            return self.primary_stage
        if stage == "critic":
            return self.critic_stage
        if stage == "hybrid":
            return self.hybrid_stage
        return stage

    def prepared_cache_path(self, work_dir: Path, page_index: int, image_path: Path) -> Path:
        from .page_cache import cache_page_path

        return cache_page_path(work_dir, page_index, image_path, "prepared")

    def translation_cache_path(self, work_dir: Path, page_index: int, image_path: Path, stage: str) -> Path:
        from .page_cache import cache_page_path

        return cache_page_path(work_dir, page_index, image_path, self.stage(stage))

    def render_only_cache_paths(self, work_dir: Path, page_index: int, image_path: Path) -> list[Path]:
        from .page_cache import cache_page_path

        return [
            cache_page_path(work_dir, page_index, image_path, stage)
            for stage in self.render_only_lookup_order
        ]

    def find_render_only_translation_cache(self, work_dir: Path, page_index: int, image_path: Path) -> Path | None:
        for cache_path in self.render_only_cache_paths(work_dir, page_index, image_path):
            if cache_path.exists():
                return cache_path
        return None

    def resume_cache_paths(self, work_dir: Path, page_index: int, image_path: Path) -> list[Path]:
        from .page_cache import cache_page_path

        return [
            cache_page_path(work_dir, page_index, image_path, stage)
            for stage in self.resume_lookup_order
        ]

    def select_hybrid_resume_cache_jobs(
        self,
        work_dir: Path,
        pages: Sequence[object],
        *,
        resume: bool,
        load_translation_cache: TranslationCacheLoader | None = None,
    ) -> HybridResumeCacheJobs:
        if load_translation_cache is None:
            from .page_cache import load_translation_cache as cache_loader
        else:
            cache_loader = load_translation_cache

        primary_jobs: list[tuple[int, object, Path]] = []
        hybrid_cached_pages: set[int] = set()
        critic_cached_pages: set[int] = set()
        for page_index, page in enumerate(pages, start=1):
            image_path = Path(getattr(page, "image_path"))
            hybrid_cache = self.translation_cache_path(work_dir, page_index, image_path, "hybrid")
            critic_cache = self.translation_cache_path(work_dir, page_index, image_path, "critic")
            primary_cache = self.translation_cache_path(work_dir, page_index, image_path, "primary")
            loaded_cache = None
            if resume:
                for cache_path in self.resume_cache_paths(work_dir, page_index, image_path):
                    if cache_path.exists():
                        cache_loader(page, cache_path)
                        loaded_cache = cache_path
                        logger.info("Loaded translation cache: %s", loaded_cache)
                        break
            if loaded_cache == hybrid_cache:
                hybrid_cached_pages.add(page_index)
            elif loaded_cache == critic_cache:
                critic_cached_pages.add(page_index)
            elif loaded_cache is None:
                primary_jobs.append((page_index, page, primary_cache))
        return HybridResumeCacheJobs(
            primary_jobs=primary_jobs,
            hybrid_cached_pages=hybrid_cached_pages,
            critic_cached_pages=critic_cached_pages,
        )


def translation_cache_plan(config: PipelineConfig) -> TranslationCachePlan:
    primary_stage = _translation_cache_stage_name(config, "primary")
    critic_stage = _translation_cache_stage_name(config, "critic")
    hybrid_stage = _translation_cache_stage_name(config, "hybrid")
    resume_stage_names = ["hybrid"]
    if config.qwen_critic_model_path is not None:
        resume_stage_names.append("critic")
    resume_stage_names.append("primary")

    render_only_stage_names: list[str] = []
    if config.qwen_fallback_model_path is not None:
        render_only_stage_names.append("hybrid")
    if config.qwen_critic_model_path is not None:
        render_only_stage_names.append("critic")
    render_only_stage_names.append("primary")
    if config.qwen_fallback_model_path is None:
        render_only_stage_names.append("hybrid")
    if config.qwen_critic_model_path is None:
        render_only_stage_names.append("critic")

    return TranslationCachePlan(
        primary_stage=primary_stage,
        critic_stage=critic_stage,
        hybrid_stage=hybrid_stage,
        resume_lookup_order=_unique_stages(_stage_names(primary_stage, critic_stage, hybrid_stage, resume_stage_names)),
        render_only_lookup_order=_unique_stages(_stage_names(primary_stage, critic_stage, hybrid_stage, render_only_stage_names)),
        final_stage=hybrid_stage if config.qwen_fallback_model_path is not None else primary_stage,
    )


def translation_cache_stage(config: PipelineConfig, stage: str) -> str:
    return translation_cache_plan(config).stage(stage)


def _translation_cache_stage_name(config: PipelineConfig, stage: str) -> str:
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
    elif config.translator == "hy-mt2":
        from .hf_translators import HY_MT2_MODEL_NAME, HY_MT2_PROMPT_VERSION, hy_mt2_num_predict, resolve_hy_mt2_model_name

        model_identity = resolve_hy_mt2_model_name(config.hy_mt2_model_name or HY_MT2_MODEL_NAME)
        identity = cache_identity_token(
            f"{model_identity}|{HY_MT2_PROMPT_VERSION}|num_predict={hy_mt2_num_predict()}"
        )
        suffix = f"{stage}-hy-mt2-{identity}-{HY_MT2_PROMPT_VERSION}-np{hy_mt2_num_predict()}"
    else:
        return stage
    if config.qwen_critic_model_path is not None or config.qwen_fallback_model_path is not None:
        suffix += "-evidence"
    if config.vision_facts_enabled:
        suffix += "-vision-facts"
    if config.qwen_critic_model_path is not None:
        suffix += "-critic"
    return suffix


def render_only_translation_stages(config: PipelineConfig) -> list[str]:
    return list(translation_cache_plan(config).render_only_lookup_order)


def _stage_names(primary_stage: str, critic_stage: str, hybrid_stage: str, stages: list[str]) -> list[str]:
    by_stage = {
        "primary": primary_stage,
        "critic": critic_stage,
        "hybrid": hybrid_stage,
    }
    return [by_stage.get(stage, stage) for stage in stages]


def _unique_stages(stage_names: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    stages: list[str] = []
    for stage in stage_names:
        if stage not in seen:
            seen.add(stage)
            stages.append(stage)
    return tuple(stages)


def cache_identity_token(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]
