from __future__ import annotations

import logging
from pathlib import Path

from .argos_translator import ArgosTranslator, translate_argos_subprocess
from .hf_translators import (
    CAT_MODEL_NAME,
    MADLAD_MODEL_NAME,
    OPUS_MODEL_NAME,
    CatTranslator,
    MadladTranslator,
    OpusTranslator,
    local_hf_cache_only,
    restore_env_var,
)
from .qwen_validation import (
    accept_qwen_translation,
    clean_qwen_output,
    count_japanese_chars,
    extract_qwen_array_strings,
    extract_qwen_bool_field,
    extract_qwen_json_payload,
    extract_qwen_json_payloads,
    extract_qwen_loose_payload,
    extract_qwen_string_field,
    extract_qwen_string_literal,
    has_repeated_ngram,
    is_placeholder_qwen_value,
    parse_qwen_critic,
    parse_qwen_translation,
    parse_qwen_page_translations,
    parse_qwen_verification,
    qwen_verification_decision_from_payload,
    should_try_qwen_repair,
    strip_qwen_wrappers,
    strip_qwen_wrappers_for_json,
    verifier_reason_says_reject,
)
from .qwen_translator import QwenTranslator
from .qwen_types import (
    QWEN_REPAIR_SETTINGS,
    QWEN_CRITIC_SETTINGS,
    QWEN_PAGE_SETTINGS,
    QWEN_TRANSLATION_SETTINGS,
    QWEN_VERIFICATION_SETTINGS,
    QwenGenerationSettings,
    QwenCriticDecision,
    QwenPageTranslation,
    QwenVerificationDecision,
    qwen_settings_to_debug_dict,
)
from .qwen_ollama import (
    QWEN_MODEL_DIR,
    ensure_ollama_model,
    find_ollama_executable,
    find_qwen_model_path,
    is_qwen_text_model_file,
    ollama_model_exists,
    ollama_model_has_projector,
    qwen_ollama_model_name,
    run_ollama_api_prompt,
    run_ollama_prompt,
    select_preferred_qwen_text_model,
)
from .qwen_prompts import (
    build_qwen_critic_prompt,
    build_qwen_repair_prompt,
    build_qwen_page_translation_prompt,
    build_qwen_translation_prompt,
    build_qwen_verification_prompt,
    normalize_context_window,
)
from .translator_base import NoopTranslator, Translator
from .translation_rules import (
    SourceReplacement,
    TargetReplacement,
    TranslationGlossary,
    apply_honorific_title_replacements,
    apply_source_pattern_replacements,
    apply_target_replacements,
    choose_contextual_translation,
    clean_translation_unit,
    default_source_replacements,
    default_target_replacements,
    extract_context_focus,
    is_watch_out_trap_senpai_phrase,
    load_translation_glossary,
    normalize_japanese_for_translation,
    normalize_phrase_key,
    postprocess_translation,
    prepare_source_for_translation,
    preserve_source_honorifics,
    resolve_glossary_path,
    should_use_context_translation,
    split_english_sentences,
    split_japanese_translation_units,
    translate_known_phrase,
    translation_debug_info,
)

logger = logging.getLogger(__name__)


def build_translator(
    name: str,
    glossary_path: Path | None = None,
    qwen_model_path: Path | None = None,
    cat_model_name: str | None = None,
) -> Translator:
    logger.info("Building translator backend: %s", name)
    if name == "none":
        return NoopTranslator(glossary_path=glossary_path)
    if name == "opus":
        return OpusTranslator(glossary_path=glossary_path)
    if name == "qwen":
        return QwenTranslator(model_path=qwen_model_path, glossary_path=glossary_path)
    if name == "cat":
        return CatTranslator(model_name=cat_model_name or CAT_MODEL_NAME, glossary_path=glossary_path)
    if name == "madlad":
        return MadladTranslator(glossary_path=glossary_path)
    if name == "argos":
        return ArgosTranslator(glossary_path=glossary_path)
    raise ValueError(f"Unknown translator: {name}")
