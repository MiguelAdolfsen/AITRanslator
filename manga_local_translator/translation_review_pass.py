from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .cache_policy import TranslationCachePlan
from .config import PipelineConfig
from .logging_utils import shorten
from .page_cache import save_translation_cache
from .page_types import PreparedPage
from .translation_review import (
    apply_qwen_critic_reviews,
    apply_qwen_fallback_translations,
    apply_translation_evidence_to_pages,
    retry_cat_failures,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - used before dependencies are installed.
    def tqdm(iterable, **_kwargs):
        return iterable

logger = logging.getLogger(__name__)

PrimaryTranslationJob = tuple[int, PreparedPage, Path]
TranslatorFactory = Callable[..., object]
TranslationCacheWriter = Callable[[PreparedPage, Path], None]
TranslationCacheLoader = Callable[[PreparedPage, Path], None]
TranslatorReleaser = Callable[[object], None]
PrimaryPageTranslator = Callable[[PreparedPage, object, PipelineConfig], None]
VisionFactsRunner = Callable[[list[PreparedPage], PipelineConfig], None]


@dataclass(frozen=True)
class TranslationReviewSummary:
    cat_retry_attempted: int = 0
    cat_retry_accepted: int = 0
    critic_attempted: int = 0
    critic_flagged: int = 0
    fallback_attempted: int = 0
    fallback_accepted: int = 0


@dataclass
class TranslationReviewPass:
    config: PipelineConfig
    cache_plan: TranslationCachePlan
    work_dir: Path
    translator_factory: TranslatorFactory
    primary_page_translator: PrimaryPageTranslator | None = None
    vision_facts_runner: VisionFactsRunner | None = None
    cache_writer: TranslationCacheWriter = save_translation_cache
    cache_loader: TranslationCacheLoader | None = None
    release_translator: TranslatorReleaser | None = None

    def __post_init__(self) -> None:
        if self.release_translator is None:
            self.release_translator = release_qwen_translator
        if self.cache_loader is None:
            from .page_cache import load_translation_cache

            self.cache_loader = load_translation_cache

    def run(
        self,
        pages: Sequence[PreparedPage],
        *,
        primary_jobs: Sequence[PrimaryTranslationJob] | None = None,
        hybrid_cached_pages: set[int] | None = None,
        critic_cached_pages: set[int] | None = None,
        primary_translator=None,
    ) -> TranslationReviewSummary:
        if primary_jobs is None or hybrid_cached_pages is None or critic_cached_pages is None:
            primary_jobs, hybrid_cached_pages, critic_cached_pages, primary_translator = self._select_and_run_primary_jobs(pages)

        try:
            cat_retry_attempted, cat_retry_accepted = self._run_cat_retry(primary_jobs, primary_translator)
        finally:
            if primary_translator is not None:
                self.release_translator(primary_translator)
        evidence_memory = None
        if self.config.qwen_critic_model_path is not None or self.config.qwen_fallback_model_path is not None:
            evidence_memory = apply_translation_evidence_to_pages(list(pages))

        critic_attempted, critic_flagged = self._run_critic(
            pages,
            hybrid_cached_pages=hybrid_cached_pages,
            critic_cached_pages=critic_cached_pages,
            evidence_memory=evidence_memory,
        )
        fallback_attempted, fallback_accepted = self._run_fallback(
            pages,
            hybrid_cached_pages=hybrid_cached_pages,
            evidence_memory=evidence_memory,
        )
        return TranslationReviewSummary(
            cat_retry_attempted=cat_retry_attempted,
            cat_retry_accepted=cat_retry_accepted,
            critic_attempted=critic_attempted,
            critic_flagged=critic_flagged,
            fallback_attempted=fallback_attempted,
            fallback_accepted=fallback_accepted,
        )

    def _select_and_run_primary_jobs(
        self,
        pages: Sequence[PreparedPage],
    ):
        resume_jobs = self.cache_plan.select_hybrid_resume_cache_jobs(
            self.work_dir,
            pages,
            resume=self.config.resume,
            load_translation_cache=self.cache_loader,
        )
        primary_jobs = resume_jobs.primary_jobs
        if self.config.vision_facts_enabled and primary_jobs and self.vision_facts_runner is not None:
            self.vision_facts_runner([page for _page_index, page, _cache in primary_jobs], self.config)

        primary_translator = None
        if primary_jobs:
            if self.primary_page_translator is None:
                raise RuntimeError("TranslationReviewPass needs primary_page_translator to run uncached primary jobs")
            logger.info("Building primary %s translator for %d pending page(s)", self.config.translator, len(primary_jobs))
            primary_translator = self.translator_factory(
                self.config.translator,
                glossary_path=self.config.glossary_path,
                qwen_model_path=self.config.qwen_model_path,
                cat_model_name=self.config.cat_model_name,
                hy_mt2_model_name=self.config.hy_mt2_model_name,
            )
            for _page_index, page, primary_cache in tqdm(primary_jobs, desc=f"{self.config.translator.upper()} primary pass", unit="page"):
                self.primary_page_translator(page, primary_translator, self.config)
                self.cache_writer(page, primary_cache)
        else:
            logger.info("Primary %s pass skipped; all pages loaded from cache", self.config.translator)
        return (
            primary_jobs,
            resume_jobs.hybrid_cached_pages,
            resume_jobs.critic_cached_pages,
            primary_translator,
        )

    def _run_cat_retry(
        self,
        primary_jobs: Sequence[PrimaryTranslationJob],
        primary_translator,
    ) -> tuple[int, int]:
        if self.config.translator != "cat" or primary_translator is None or not primary_jobs:
            return (0, 0)
        total_attempted = 0
        total_accepted = 0
        for _page_index, page, primary_cache in tqdm(primary_jobs, desc="CAT retry pass", unit="page"):
            attempted, accepted = retry_cat_failures(page, primary_translator, self.config)
            if attempted:
                self.cache_writer(page, primary_cache)
            total_attempted += attempted
            total_accepted += accepted
        logger.info(
            "CAT retry scan complete: attempted=%d accepted=%d",
            total_attempted,
            total_accepted,
        )
        return total_attempted, total_accepted

    def _run_critic(
        self,
        pages: Sequence[PreparedPage],
        *,
        hybrid_cached_pages: set[int],
        critic_cached_pages: set[int],
        evidence_memory,
    ) -> tuple[int, int]:
        critic_jobs = []
        if self.config.qwen_critic_model_path is not None:
            critic_jobs = [
                (
                    page_index,
                    page,
                    self.cache_plan.translation_cache_path(self.work_dir, page_index, page.image_path, "critic"),
                )
                for page_index, page in enumerate(pages, start=1)
                if page_index not in hybrid_cached_pages and page_index not in critic_cached_pages
            ]
        total_attempted = 0
        total_flagged = 0
        if critic_jobs:
            logger.info(
                "Primary %s pass complete; building critic Qwen translator for %d pending page(s)",
                self.config.translator,
                len(critic_jobs),
            )
            critic_translator = self.translator_factory(
                "qwen",
                glossary_path=self.config.glossary_path,
                qwen_model_path=self.config.qwen_critic_model_path,
            )
            try:
                for _page_index, page, critic_cache in tqdm(critic_jobs, desc="Qwen critic pass", unit="page"):
                    attempted, flagged = apply_qwen_critic_reviews(
                        page.render_blocks,
                        page.translations,
                        page.translation_contexts,
                        page.page_order_report,
                        critic_translator,
                        self.config,
                        evidence_memory=evidence_memory,
                    )
                    self.cache_writer(page, critic_cache)
                    total_attempted += attempted
                    total_flagged += flagged
            finally:
                self.release_translator(critic_translator)
        elif self.config.qwen_critic_model_path is not None:
            logger.info("Critic Qwen pass skipped; all pages loaded from cache")
        logger.info(
            "Critic Qwen pass complete: attempted=%d flagged=%d",
            total_attempted,
            total_flagged,
        )
        return total_attempted, total_flagged

    def _run_fallback(
        self,
        pages: Sequence[PreparedPage],
        *,
        hybrid_cached_pages: set[int],
        evidence_memory,
    ) -> tuple[int, int]:
        fallback_jobs = []
        if self.config.qwen_fallback_model_path is not None:
            fallback_jobs = [
                (
                    page_index,
                    page,
                    self.cache_plan.translation_cache_path(self.work_dir, page_index, page.image_path, "hybrid"),
                )
                for page_index, page in enumerate(pages, start=1)
                if page_index not in hybrid_cached_pages
            ]
        total_attempted = 0
        total_accepted = 0
        for page_index in hybrid_cached_pages:
            page = pages[page_index - 1]
            total_attempted += sum(1 for context in page.translation_contexts.values() if context.get("qwen_fallback_attempted") is True)
            total_accepted += sum(1 for context in page.translation_contexts.values() if context.get("qwen_fallback_accepted") is True)

        if fallback_jobs:
            logger.info(
                "Primary %s pass complete; building fallback Qwen translator for %d pending page(s)",
                self.config.translator,
                len(fallback_jobs),
            )
            fallback_translator = self.translator_factory(
                "qwen",
                glossary_path=self.config.glossary_path,
                qwen_model_path=self.config.qwen_fallback_model_path,
            )
            try:
                for _page_index, page, hybrid_cache in tqdm(fallback_jobs, desc="Q8 fallback pass", unit="page"):
                    attempted, accepted = apply_qwen_fallback_translations(
                        page.render_blocks,
                        page.translations,
                        page.translation_contexts,
                        page.page_order_report,
                        fallback_translator,
                        self.config,
                        evidence_memory=evidence_memory,
                    )
                    self.cache_writer(page, hybrid_cache)
                    total_attempted += attempted
                    total_accepted += accepted
            finally:
                self.release_translator(fallback_translator)
        else:
            logger.info("Fallback Qwen pass skipped; all pages loaded from hybrid cache")
        logger.info(
            "Fallback Qwen pass complete: attempted=%d accepted=%d",
            total_attempted,
            total_accepted,
        )
        return total_attempted, total_accepted


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
