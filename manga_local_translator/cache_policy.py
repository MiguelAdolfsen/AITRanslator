from __future__ import annotations

import hashlib
from pathlib import Path

from .config import PipelineConfig


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


def cache_identity_token(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]
