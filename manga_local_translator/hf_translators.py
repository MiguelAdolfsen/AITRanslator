from __future__ import annotations

import logging
import os
import re
import subprocess
import unicodedata
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path

from .logging_utils import shorten
from .translation_rules import (
    choose_contextual_translation,
    clean_translation_unit,
    normalize_japanese_for_translation,
    postprocess_translation,
    prepare_source_for_translation,
    should_use_context_translation,
    split_japanese_translation_units,
    translate_known_phrase,
)
from .translator_base import Translator

logger = logging.getLogger(__name__)


MADLAD_MODEL_NAME = "google/madlad400-3b-mt"
OPUS_MODEL_NAME = "Helsinki-NLP/opus-mt-ja-en"
CAT_MODEL_NAME = "cyberagent/CAT-Translate-7b"
CAT_MODEL_DIR = Path(".models") / "CAT-Translate"
CAT_PROMPT_VERSION = "cat-translate-v9-token-complete-primary"
CAT_RETRY_PROMPT_VERSION = "cat-retry-v1-source-only"
CAT_SECOND_RETRY_PROMPT_VERSION = "cat-retry-v2-incomplete-fragment"
CAT_VALIDATION_VERSION = "cat-validation-v17-honorific-name-shape"
CAT_BYPASS_VERSION = "cat-bypass-v1"
CAT_DEFAULT_NUM_PREDICT = 48


class CatTranslator(Translator):
    def __init__(self, model_name: str | Path = CAT_MODEL_NAME, glossary_path: Path | None = None) -> None:
        super().__init__(glossary_path)
        self._gguf_path = resolve_cat_gguf_path(model_name)
        if self._gguf_path is not None:
            self._init_ollama(self._gguf_path)
            return

        logger.info("Initializing CAT translator: model=%s", model_name)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            logger.exception("CAT translator dependencies are missing")
            raise RuntimeError(
                "CAT translator dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        self._cache: dict[str, str] = {}
        self._debug: dict[str, dict[str, object]] = {}
        self._model_name = str(model_name)
        self._backend = "transformers"
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with local_hf_cache_only():
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(str(model_name), local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "CAT translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_cat"
                ) from exc

        model_kwargs = {"low_cpu_mem_usage": True}
        if self._device.type == "cuda":
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        with local_hf_cache_only():
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(model_name),
                    local_files_only=True,
                    **model_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    "CAT translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_cat"
                ) from exc
        self._model.to(self._device)
        self._model.eval()
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info("CAT translator ready on %s", self._device)

    def _init_ollama(self, gguf_path: Path) -> None:
        logger.info("Initializing CAT translator from GGUF: %s", gguf_path)
        from .qwen_ollama import find_ollama_executable, run_ollama_prompt
        from .qwen_types import QwenGenerationSettings

        self._cache = {}
        self._debug = {}
        self._model_name = cat_ollama_model_name(gguf_path)
        self._backend = "ollama"
        self._device = "ollama"
        self._gguf_path = gguf_path
        self._ollama = find_ollama_executable()
        if not self._ollama:
            raise RuntimeError("Ollama was not found. Install/start Ollama to use CAT GGUF models.")
        self._ollama_model_name = ensure_cat_ollama_model(self._ollama, self._model_name, gguf_path)
        self._cat_settings = QwenGenerationSettings(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            min_p=0.0,
            num_predict=cat_num_predict(),
        )
        self._run_cat_prompt = run_ollama_prompt
        logger.info("CAT Ollama translator ready: %s", self._ollama_model_name)

    def translate(self, text: str) -> str:
        if text not in self._cache:
            logger.debug("CAT translating text=%s", shorten(text))
            translated, debug = self._translate_normalized(text)
            self._cache[text] = translated
            self._debug[text] = debug
            logger.debug("CAT translated result=%s", shorten(translated))
        else:
            logger.debug("CAT translation cache hit for text=%s", shorten(text))
        return self._cache[text]

    def retry_translation(self, text: str) -> str:
        logger.debug("CAT retry translating text=%s", shorten(text))
        translated, debug = self._translate_normalized(text, retry=True)
        second_debug: dict[str, object] | None = None
        if debug.get("cat_rejected") is True and cat_second_retry_enabled():
            logger.debug("CAT second retry translating text=%s", shorten(text))
            second_translated, second_debug = self._translate_normalized(
                text,
                retry=True,
                retry_variant="incomplete_fragment",
            )
            if second_debug.get("cat_rejected") is not True:
                translated = second_translated
                debug = second_debug
        self._cache[text] = translated
        existing = dict(self._debug.get(text, {}))
        primary_debug = {
            "cat_primary_prompt": existing.get("cat_prompt", ""),
            "cat_primary_prompt_version": existing.get("cat_prompt_version", ""),
            "cat_primary_raw_translation": existing.get("cat_raw_translation", ""),
            "cat_primary_cleaned_translation": existing.get("cat_cleaned_translation", ""),
            "cat_primary_final": existing.get("cat_final", ""),
            "cat_primary_rejected": existing.get("cat_rejected", False),
            "cat_primary_reject_reason": existing.get("cat_reject_reason", ""),
        }
        merged = {
            **existing,
            **primary_debug,
            "cat_retry_attempted": True,
            "cat_retry_raw_translation": debug.get("cat_raw_translation", ""),
            "cat_retry_cleaned_translation": debug.get("cat_cleaned_translation", ""),
            "cat_retry_final": debug.get("cat_final", translated),
            "cat_retry_rejected": debug.get("cat_rejected", False),
            "cat_retry_reject_reason": debug.get("cat_reject_reason", ""),
            "cat_retry_prompt": debug.get("cat_prompt", ""),
            "cat_retry_prompt_version": debug.get("cat_prompt_version", CAT_RETRY_PROMPT_VERSION),
            "cat_retry_chatter_rejected": debug.get("cat_chatter_rejected", False),
            "cat_retried": True,
        }
        if second_debug is not None:
            merged.update(
                {
                    "cat_second_retry_attempted": True,
                    "cat_second_retry_raw_translation": second_debug.get("cat_raw_translation", ""),
                    "cat_second_retry_cleaned_translation": second_debug.get("cat_cleaned_translation", ""),
                    "cat_second_retry_final": second_debug.get("cat_final", ""),
                    "cat_second_retry_rejected": second_debug.get("cat_rejected", False),
                    "cat_second_retry_reject_reason": second_debug.get("cat_reject_reason", ""),
                    "cat_second_retry_prompt": second_debug.get("cat_prompt", ""),
                    "cat_second_retry_prompt_version": second_debug.get(
                        "cat_prompt_version",
                        CAT_SECOND_RETRY_PROMPT_VERSION,
                    ),
                    "cat_second_retry_chatter_rejected": second_debug.get("cat_chatter_rejected", False),
                }
            )
        if debug.get("cat_rejected") is not True:
            merged.update(debug)
            merged["cat_retry_accepted"] = True
            merged["cat_rejected"] = False
            merged["cat_reject_reason"] = ""
        else:
            merged["cat_retry_accepted"] = False
        self._debug[text] = merged
        logger.debug("CAT retry result=%s rejected=%s", shorten(translated), debug.get("cat_rejected"))
        return translated

    def second_retry_translation(self, text: str, previous_translation: str = "") -> str:
        logger.debug("CAT strict second-pass translating text=%s", shorten(text))
        translated, debug = self._translate_normalized(
            text,
            retry=True,
            retry_variant="incomplete_fragment",
        )
        existing = dict(self._debug.get(text, {}))
        merged = {
            **existing,
            "cat_suspect_second_pass_attempted": True,
            "cat_suspect_second_pass_previous_translation": previous_translation,
            "cat_suspect_second_pass_raw_translation": debug.get("cat_raw_translation", ""),
            "cat_suspect_second_pass_cleaned_translation": debug.get("cat_cleaned_translation", ""),
            "cat_suspect_second_pass_final": debug.get("cat_final", translated),
            "cat_suspect_second_pass_rejected": debug.get("cat_rejected", False),
            "cat_suspect_second_pass_reject_reason": debug.get("cat_reject_reason", ""),
            "cat_suspect_second_pass_prompt": debug.get("cat_prompt", ""),
            "cat_suspect_second_pass_prompt_version": debug.get(
                "cat_prompt_version",
                CAT_SECOND_RETRY_PROMPT_VERSION,
            ),
        }
        if debug.get("cat_rejected") is not True:
            self._cache[text] = translated
            merged.update(debug)
            merged["cat_suspect_second_pass_accepted"] = True
            merged["cat_rejected"] = False
            merged["cat_reject_reason"] = ""
        else:
            merged["cat_suspect_second_pass_accepted"] = False
        self._debug[text] = merged
        logger.debug("CAT strict second-pass result=%s rejected=%s", shorten(translated), debug.get("cat_rejected"))
        return translated

    def debug_info_for(self, text: str, debug_id: str | None = None) -> dict[str, object]:
        _ = debug_id
        return dict(self._debug.get(text, {}))

    def _translate_normalized(
        self,
        text: str,
        *,
        retry: bool = False,
        retry_variant: str = "source_only",
    ) -> tuple[str, dict[str, object]]:
        normalized = normalize_japanese_for_translation(text)
        prompt_source = normalized
        prepared, _replacements = prepare_source_for_translation(prompt_source, self._glossary)
        phrase = translate_known_phrase(normalized, self._glossary) or translate_known_phrase(prepared, self._glossary)
        if phrase is not None:
            return phrase, {
                "primary_translator": "cat",
                "cat_used": False,
                "cat_reason": "phrasebook",
                "cat_model": self._model_name,
                "cat_model_path": str(getattr(self, "_gguf_path", "")),
                "cat_backend": getattr(self, "_backend", "unknown"),
                "cat_prompt_version": CAT_PROMPT_VERSION,
                "cat_validation_version": CAT_VALIDATION_VERSION,
                "cat_num_predict": cat_num_predict(),
                "cat_source_text": normalized,
                "cat_prompt_source_text": prompt_source,
                "cat_rejected": False,
                "cat_final": phrase,
            }
        bypass_reason = cat_bypass_reason(normalized)
        if bypass_reason is not None:
            return "", {
                "primary_translator": "cat",
                "cat_used": False,
                "cat_bypassed": True,
                "cat_bypass_reason": bypass_reason,
                "cat_model": self._model_name,
                "cat_model_path": str(getattr(self, "_gguf_path", "")),
                "cat_backend": getattr(self, "_backend", "unknown"),
                "cat_prompt_version": CAT_PROMPT_VERSION,
                "cat_validation_version": CAT_VALIDATION_VERSION,
                "cat_num_predict": cat_num_predict(),
                "cat_source_text": normalized,
                "cat_prompt_source_text": prompt_source,
                "cat_rejected": False,
                "cat_final": "",
            }

        prompt_version = cat_prompt_version(retry=retry, retry_variant=retry_variant)
        prompt = build_cat_prompt(prepared, retry=retry, retry_variant=retry_variant)
        if getattr(self, "_backend", "") == "ollama":
            raw = self._run_cat_prompt(self._ollama, self._ollama_model_name, prompt, settings=self._cat_settings).strip()
            translated, result, reject_reason = finalize_cat_translation(normalized, prepared, raw, self._glossary)
            return result, {
                "primary_translator": "cat",
                "cat_used": True,
                "cat_model": self._ollama_model_name,
                "cat_model_path": str(self._gguf_path),
                "cat_backend": "ollama",
                "cat_prompt_version": prompt_version,
                "cat_validation_version": CAT_VALIDATION_VERSION,
                "cat_num_predict": self._cat_settings.num_predict,
                "cat_source_text": normalized,
                "cat_prompt_source_text": prompt_source,
                "cat_prompt": prompt,
                "cat_raw_translation": raw,
                "cat_cleaned_translation": translated,
                "cat_candidate": translated,
                "cat_final": result,
                "cat_rejected": reject_reason is not None,
                "cat_reject_reason": reject_reason or "",
                "cat_chatter_rejected": reject_reason in {"cat_chatter", "cat_prompt_fragment", "cat_schema_fragment"},
            }

        messages = [{"role": "user", "content": prompt}]
        with local_hf_cache_only():
            encoded = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            input_length = int(encoded["input_ids"].shape[-1])
            with self._torch.inference_mode():
                outputs = self._model.generate(
                    **encoded,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
        generated = outputs[0][input_length:]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        translated, result, reject_reason = finalize_cat_translation(normalized, prepared, raw, self._glossary)
        return result, {
            "primary_translator": "cat",
            "cat_used": True,
            "cat_model": self._model_name,
            "cat_model_path": str(model_path_for_debug(self._model_name)),
            "cat_backend": self._backend,
            "cat_device": str(self._device),
            "cat_prompt_version": prompt_version,
            "cat_validation_version": CAT_VALIDATION_VERSION,
            "cat_num_predict": CAT_DEFAULT_NUM_PREDICT,
            "cat_source_text": normalized,
            "cat_prompt_source_text": prompt_source,
            "cat_prompt": prompt,
            "cat_raw_translation": raw,
            "cat_cleaned_translation": translated,
            "cat_candidate": translated,
            "cat_final": result,
            "cat_rejected": reject_reason is not None,
            "cat_reject_reason": reject_reason or "",
            "cat_chatter_rejected": reject_reason in {"cat_chatter", "cat_prompt_fragment", "cat_schema_fragment"},
        }


def build_cat_prompt(prepared_text: str, *, retry: bool = False, retry_variant: str = "source_only") -> str:
    if retry:
        if retry_variant == "incomplete_fragment":
            return (
                "Translate exactly. Preserve names, terms, relationship words, and romanized honorifics such as san, sama, kun, chan, senpai, and sensei. "
                "If the source is incomplete, translate the incomplete fragment. Never ask for clarification.\n"
                f"Japanese: \"{prepared_text}\"\n"
                "English:"
            )
        return f"Japanese:\n{prepared_text}\n\nEnglish:"
    return (
        "Translate exactly. Return only concise English. Preserve names, terms, relationship words, and romanized honorifics such as san, sama, kun, chan, senpai, and sensei. "
        "Translate every source token; romanize katakana names instead of dropping them. "
        "Do not replace honorifics with English titles. Do not explain, apologize, ask for clarification, or continue the scene. "
        "Never say you do not understand or need more context. "
        "If the source is incomplete, translate the fragment as a fragment.\n"
        f"Japanese: \"{prepared_text}\"\n"
        "English:"
    )


def cat_prompt_version(*, retry: bool = False, retry_variant: str = "source_only") -> str:
    if not retry:
        return CAT_PROMPT_VERSION
    if retry_variant == "incomplete_fragment":
        return CAT_SECOND_RETRY_PROMPT_VERSION
    return CAT_RETRY_PROMPT_VERSION


def cat_num_predict() -> int:
    raw = os.environ.get("MANGA_CAT_NUM_PREDICT", "").strip()
    if not raw:
        return CAT_DEFAULT_NUM_PREDICT
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid MANGA_CAT_NUM_PREDICT=%s", raw)
        return CAT_DEFAULT_NUM_PREDICT
    return max(16, min(256, value))


def cat_bypass_reason(source_text: str) -> str | None:
    if not cat_bypass_enabled():
        return None
    text = str(source_text).strip()
    if not text:
        return "empty_source"
    japanese_count = count_japanese_chars_local(text)
    if japanese_count == 0:
        return None
    if has_noisy_credit_markers(text):
        return "noisy_credit_or_metadata"
    if japanese_count <= 5:
        return "short_ambiguous_fragment"
    if is_short_kana_name_or_term(text):
        return "short_kana_name_or_term"
    return None


def has_noisy_credit_markers(text: str) -> bool:
    if any(marker in text for marker in ("▽", "▼", "■", "□", "◆", "◇", "※", "©", "＠", "@", "http")):
        return True
    normalized = unicodedata.normalize("NFKC", str(text))
    if is_date_like_metadata_fragment(normalized):
        return True
    if is_numbered_metadata_header(normalized):
        return True
    if "『" in text and "』" not in text:
        return True
    if re.search(r"(?:\d{1,2}|[０-９]{1,2})\s*(?:/|月)\s*(?:\d{1,2}|[０-９]{1,2})", text) and re.search(
        r"(?:更新|作者|コメント|COMIC|comic|page|頁)",
        text,
    ):
        return True
    if re.search(r"第\s*[0-9０-９一二三四五六七八九十百]+\s*(?:話|章|巻)", text) and re.search(
        r"(?:つづく|続く|完|[12][0-9]{3})",
        text,
    ):
        return True
    return False


def is_date_like_metadata_fragment(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    if not normalized:
        return False
    if len(normalized) > 16:
        return False
    return bool(
        re.fullmatch(
            r"\d{2,4}(?:\s*年\s*\d{1,2}(?:\s*月\s*\d{0,2}(?:\s*日)?)?|\s*[/-]\s*\d{1,2}(?:\s*[/-]\s*\d{1,2})?)",
            normalized,
        )
    )


def is_numbered_metadata_header(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    if not re.search(r"\u7b2c\s*[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]+\s*(?:\u8a71|\u7ae0|\u5dfb|\u56de|\u90e8)", normalized):
        return False
    terminal_markers = (
        "\u304a\u308f\u308a",
        "\u7d42\u308f\u308a",
        "\u5b8c",
        "\u5b8c\u7d50",
        "\u6b21\u56de",
        "\u66f4\u65b0",
        "\u544a\u77e5",
        "\u4e88\u544a",
        "\u30b3\u30e1\u30f3\u30c8",
        "END",
        "comment",
    )
    if any(marker.lower() in normalized.lower() for marker in terminal_markers):
        return True
    if len(normalized) > 40 or re.search(r"[\u3002\uff01\uff1f!?]", normalized):
        return False
    parts = re.findall(r"\S+", normalized)
    return len(parts) <= 4 and bool(re.search(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", normalized))


def is_short_kana_name_or_term(text: str) -> bool:
    stripped = re.sub(r"[\s\u3000・ーｰ！!？?。．…,\.\-―〜～]+", "", text)
    if not stripped or len(stripped) > 8:
        return False
    return all(0x3040 <= ord(char) <= 0x30FF or 0x31F0 <= ord(char) <= 0x31FF for char in stripped)


def cat_bypass_enabled() -> bool:
    return os.environ.get("MANGA_CAT_BYPASS_RISKY_SOURCE", "").strip().lower() in {"1", "true", "yes", "on"}


def cat_second_retry_enabled() -> bool:
    raw = os.environ.get("MANGA_CAT_SECOND_RETRY", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def finalize_cat_translation(source_text: str, prepared_text: str, raw_text: str, glossary) -> tuple[str, str, str | None]:
    cleaned = clean_cat_output(raw_text)
    phrase = short_source_phrase_override(source_text, prepared_text, glossary)
    if phrase is not None:
        result = postprocess_translation(source_text, prepared_text, phrase, glossary)
        final_reject_reason = cat_reject_reason(source_text, result, result)
        if final_reject_reason is None:
            return phrase, result, None
    reject_reason = cat_reject_reason(source_text, cleaned, raw_text)
    if reject_reason is not None:
        salvaged = salvage_cat_translation(raw_text, source_text=source_text)
        if salvaged:
            salvage_reject_reason = cat_reject_reason(source_text, salvaged, salvaged)
            if salvage_reject_reason is None:
                result = postprocess_translation(source_text, prepared_text, salvaged, glossary)
                final_reject_reason = cat_reject_reason(source_text, result, result)
                if final_reject_reason is None:
                    return salvaged, result, None
        return cleaned, "", reject_reason
    result = postprocess_translation(source_text, prepared_text, cleaned, glossary)
    final_reject_reason = cat_reject_reason(source_text, result, raw_text)
    if final_reject_reason is not None:
        return cleaned, "", final_reject_reason
    return cleaned, result, None


def short_source_phrase_override(source_text: str, prepared_text: str, glossary) -> str | None:
    normalized = normalize_japanese_for_translation(source_text)
    if has_noisy_credit_markers(normalized):
        return None
    if not is_short_symbolic_source(normalized):
        return None
    phrase = translate_known_phrase(normalized, glossary) or translate_known_phrase(prepared_text, glossary)
    if phrase is not None:
        return phrase
    return translate_kana_honorific_name_fragment(normalized)


def is_short_symbolic_source(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text).strip())
    if not compact or count_japanese_chars_local(compact) == 0:
        return False
    if len(compact) <= 12:
        return True
    return bool(re.fullmatch(r"[\u3040-\u30ff\u31f0-\u31ff\u2026.!?\-ー〜～]+", compact)) and len(compact) <= 16


def translate_kana_honorific_name_fragment(source_text: str) -> str | None:
    compact = normalize_japanese_for_translation(source_text)
    compact = compact.strip("\u300c\u300d\u300e\u300f\"'")
    trailing = ""
    punctuation_match = re.search(r"(?P<trailing>(?:\u2026|[.!?\uff01\uff1f\uff0e]){1,4})$", compact)
    if punctuation_match:
        trailing = normalize_fragment_punctuation(punctuation_match.group("trailing"))
        compact = compact[: punctuation_match.start()].strip()
    suffixes = {
        "\u3055\u3093": "san",
        "\u30b5\u30f3": "san",
        "\u3055\u307e": "sama",
        "\u30b5\u30de": "sama",
        "\u69d8": "sama",
        "\u3061\u3083\u3093": "chan",
        "\u30c1\u30e3\u30f3": "chan",
        "\u304f\u3093": "kun",
        "\u30af\u30f3": "kun",
        "\u541b": "kun",
        "\u5148\u8f29": "senpai",
        "\u305b\u3093\u3071\u3044": "senpai",
        "\u30bb\u30f3\u30d1\u30a4": "senpai",
        "\u5148\u751f": "sensei",
        "\u305b\u3093\u305b\u3044": "sensei",
        "\u30bb\u30f3\u30bb\u30a4": "sensei",
    }
    family_bases = {"\u304a\u3058", "\u304a\u3070", "\u3058\u3044", "\u3070\u3042", "\u3068\u3046", "\u304b\u3042", "\u306b\u3044", "\u306d\u3048"}
    for suffix, honorific in sorted(suffixes.items(), key=lambda item: len(item[0]), reverse=True):
        if not compact.endswith(suffix):
            continue
        name = compact[: -len(suffix)]
        if not name or name in family_bases:
            return None
        if honorific == "san" and is_short_hiragana_family_like_base(name):
            return None
        romanized = romanize_kana_name(name)
        if romanized is None:
            return None
        return f"{romanized} {honorific}{trailing}"
    return None


def normalize_fragment_punctuation(text: str) -> str:
    result = str(text)
    result = result.replace("\uff01", "!").replace("\uff1f", "?").replace("\uff0e", ".")
    result = result.replace("\u2026", "...")
    return result


def is_short_hiragana_family_like_base(text: str) -> bool:
    return len(text) <= 2 and bool(re.fullmatch(r"[\u3040-\u309f]+", text))


def romanize_kana_name(text: str) -> str | None:
    kana = unicodedata.normalize("NFKC", text).replace("\u30fb", "")
    if "\u30fc" in kana:
        return None
    if not kana or not re.fullmatch(r"[\u3040-\u30ff\u31f0-\u31ff\u30fc]+", kana):
        return None
    hiragana = "".join(katakana_to_hiragana(char) for char in kana)
    romanized = romanize_hiragana(hiragana)
    if not romanized or len(romanized) > 40:
        return None
    return romanized[:1].upper() + romanized[1:]


def katakana_to_hiragana(char: str) -> str:
    code = ord(char)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    return char


def romanize_hiragana(text: str) -> str:
    digraphs = {
        "\u304d\u3083": "kya", "\u304d\u3085": "kyu", "\u304d\u3087": "kyo",
        "\u304e\u3083": "gya", "\u304e\u3085": "gyu", "\u304e\u3087": "gyo",
        "\u3057\u3083": "sha", "\u3057\u3085": "shu", "\u3057\u3087": "sho",
        "\u3058\u3083": "ja", "\u3058\u3085": "ju", "\u3058\u3087": "jo",
        "\u3061\u3083": "cha", "\u3061\u3085": "chu", "\u3061\u3087": "cho",
        "\u306b\u3083": "nya", "\u306b\u3085": "nyu", "\u306b\u3087": "nyo",
        "\u3072\u3083": "hya", "\u3072\u3085": "hyu", "\u3072\u3087": "hyo",
        "\u3073\u3083": "bya", "\u3073\u3085": "byu", "\u3073\u3087": "byo",
        "\u3074\u3083": "pya", "\u3074\u3085": "pyu", "\u3074\u3087": "pyo",
        "\u307f\u3083": "mya", "\u307f\u3085": "myu", "\u307f\u3087": "myo",
        "\u308a\u3083": "rya", "\u308a\u3085": "ryu", "\u308a\u3087": "ryo",
    }
    singles = {
        "\u3042": "a", "\u3044": "i", "\u3046": "u", "\u3048": "e", "\u304a": "o",
        "\u304b": "ka", "\u304d": "ki", "\u304f": "ku", "\u3051": "ke", "\u3053": "ko",
        "\u304c": "ga", "\u304e": "gi", "\u3050": "gu", "\u3052": "ge", "\u3054": "go",
        "\u3055": "sa", "\u3057": "shi", "\u3059": "su", "\u305b": "se", "\u305d": "so",
        "\u3056": "za", "\u3058": "ji", "\u305a": "zu", "\u305c": "ze", "\u305e": "zo",
        "\u305f": "ta", "\u3061": "chi", "\u3064": "tsu", "\u3066": "te", "\u3068": "to",
        "\u3060": "da", "\u3062": "ji", "\u3065": "zu", "\u3067": "de", "\u3069": "do",
        "\u306a": "na", "\u306b": "ni", "\u306c": "nu", "\u306d": "ne", "\u306e": "no",
        "\u306f": "ha", "\u3072": "hi", "\u3075": "fu", "\u3078": "he", "\u307b": "ho",
        "\u3070": "ba", "\u3073": "bi", "\u3076": "bu", "\u3079": "be", "\u307c": "bo",
        "\u3071": "pa", "\u3074": "pi", "\u3077": "pu", "\u307a": "pe", "\u307d": "po",
        "\u307e": "ma", "\u307f": "mi", "\u3080": "mu", "\u3081": "me", "\u3082": "mo",
        "\u3084": "ya", "\u3086": "yu", "\u3088": "yo",
        "\u3089": "ra", "\u308a": "ri", "\u308b": "ru", "\u308c": "re", "\u308d": "ro",
        "\u308f": "wa", "\u3092": "o", "\u3093": "n",
        "\u3041": "a", "\u3043": "i", "\u3045": "u", "\u3047": "e", "\u3049": "o",
        "\u3083": "ya", "\u3085": "yu", "\u3087": "yo", "\u3094": "vu",
    }
    result: list[str] = []
    double_next = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\u3063":
            double_next = True
            index += 1
            continue
        if char == "\u30fc":
            extend_last_vowel(result)
            index += 1
            continue
        unit = text[index : index + 2]
        if unit in digraphs:
            roma = digraphs[unit]
            index += 2
        elif char in singles:
            roma = singles[char]
            index += 1
        else:
            return ""
        if double_next and roma:
            result.append(first_roman_consonant(roma))
            double_next = False
        result.append(roma)
    return "".join(result)


def first_roman_consonant(text: str) -> str:
    first = text[:1]
    return first if first and first not in "aeiou" else ""


def extend_last_vowel(parts: list[str]) -> None:
    for part in reversed(parts):
        for char in reversed(part):
            if char in "aeiou":
                parts.append(char)
                return


def salvage_cat_translation(raw_text: str, *, source_text: str = "") -> str:
    raw = str(raw_text).strip()
    candidates: list[str] = []
    english_match = re.search(r"\bEnglish:\s*(.+?)(?:\n|$)", raw, flags=re.IGNORECASE)
    if english_match:
        candidates.append(clean_salvaged_english_marker(english_match.group(1)))
    if can_salvage_explanatory_cat_answer(raw, source_text=source_text):
        candidates.extend(extract_salvage_quoted_candidates(raw))
    if can_salvage_direct_translation_explanation(raw, source_text=source_text):
        candidates.extend(extract_salvage_quoted_candidates(raw))
    for candidate in candidates:
        cleaned = clean_cat_output(candidate)
        cleaned = cleaned.strip(" '\"“”‘’`")
        if not cleaned or count_japanese_chars_local(cleaned) > 0:
            continue
        if len(cleaned.split()) > 12 or len(cleaned) > 80:
            continue
        if looks_like_cat_chatter(cleaned) or has_cat_prompt_fragment(cleaned):
            continue
        return cleaned
    return ""


def can_salvage_explanatory_cat_answer(raw_text: str, *, source_text: str = "") -> bool:
    lower = " ".join(str(raw_text).lower().split())
    if "can be translated as" not in lower and "could be translated as" not in lower:
        return False
    unsafe_markers = (
        "i don't understand",
        "i do not understand",
        "could you please clarify",
        "please clarify",
        "not standard",
        "not commonly used",
        "misspelling",
        "typo",
        "if you're referring",
        "if you mean",
        "appears to be",
        "would be expressed as",
    )
    return not any(marker in lower for marker in unsafe_markers)


def can_salvage_direct_translation_explanation(raw_text: str, *, source_text: str = "") -> bool:
    raw = str(raw_text)
    lower = " ".join(raw.lower().split())
    direct_markers = (
        " means ",
        "translates to",
        "translation is",
        "correct english translation is",
        "translation of what you provided",
    )
    if not any(marker in lower for marker in direct_markers):
        return False
    source_risk = cat_source_salvage_risk(source_text)
    if source_risk and source_risk != "short_ambiguous_fragment":
        return False
    unsafe_markers = (
        "not standard",
        "not commonly used",
        "misspelling",
        "typo",
        "if you're referring",
        "if you mean",
        "appears to be",
        "could you please clarify",
        "please clarify",
        "not sure what it refers",
    )
    if any(marker in lower for marker in unsafe_markers):
        return False
    if re.search(r"[\u30A0-\u30FF\u31F0-\u31FF]", raw):
        return False
    return True


def cat_source_salvage_risk(source_text: str) -> str | None:
    text = str(source_text).strip()
    if not text:
        return "empty_source"
    if has_noisy_credit_markers(text):
        return "noisy_credit_or_metadata"
    japanese_count = count_japanese_chars_local(text)
    if japanese_count <= 5:
        return "short_ambiguous_fragment"
    if is_short_kana_name_or_term(text):
        return "short_kana_name_or_term"
    if any(marker in text for marker in ("〈", "〉", "《", "》")):
        return "bracketed_term_source"
    return None


def extract_salvage_quoted_candidates(raw_text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r'["“”]([^"“”]{1,80})["“”]', str(raw_text)):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        if count_japanese_chars_local(candidate) > 0:
            continue
        candidates.append(candidate)
    return candidates


def clean_salvaged_english_marker(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.split(
        r"\s{2,}(?:This|It|In|Depending|The)\b|\s+(?:This captures|This means|It means|Depending on)\b",
        cleaned,
        maxsplit=1,
    )[0].strip()
    return cleaned


def clean_cat_output(text: str) -> str:
    cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", str(text)).strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.lower().startswith("translation:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    if cleaned.lower().startswith("english:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"\b(?:[A-Z][a-z]+) is a Japanese translator\b.*$", "", cleaned, flags=re.DOTALL).strip()
    if looks_like_cat_chatter(cleaned):
        return ""
    cleaned = re.sub(r"\s*\((?:Note|Explanation):.*?\)\s*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    filtered = [
        line
        for line in lines
        if line.lower() != "assistant"
        and not line.lower().startswith("please provide")
        and not line.lower().startswith("once i have")
        and not line.lower().startswith("return english translation")
    ]
    if filtered:
        cleaned = " ".join(filtered)
    if re.fullmatch(r"(?:assistant\s*){2,}", cleaned, flags=re.IGNORECASE):
        return ""
    return cleaned.strip()


def cat_reject_reason(source_text: str, cleaned_text: str, raw_text: str = "") -> str | None:
    text = str(cleaned_text).strip()
    raw = str(raw_text).strip()
    combined = f"{raw}\n{text}"
    if not text:
        if looks_like_cat_chatter(raw):
            return "cat_chatter"
        return "cat_empty"
    if looks_like_cat_chatter(combined):
        return "cat_chatter"
    if looks_like_explanatory_cat_output(raw):
        return "cat_explanatory_output"
    if has_cat_prompt_fragment(combined):
        return "cat_prompt_fragment"
    if has_cat_schema_fragment(combined):
        return "cat_schema_fragment"
    if has_noisy_credit_markers(source_text):
        return "cat_source_metadata_or_noise"
    if count_japanese_chars_local(text) > 0:
        return "cat_untranslated_japanese"
    if looks_repetitive_or_verbose(source_text, text):
        return "cat_verbose_or_repetitive"
    if re.search(r"\b[A-Za-z]+(?:-[A-Za-z]+){4,}\b", text):
        return "cat_malformed_hyphen_chain"
    if source_text.strip() and text.strip() == source_text.strip():
        return "cat_unchanged_source"
    return None


def looks_like_cat_chatter(text: str) -> bool:
    lower = " ".join(str(text).lower().split())
    chatter_patterns = (
        "please provide the japanese",
        "provide the japanese passage",
        "provide the japanese text",
        "could you please provide",
        "i need you to provide",
        "i need the complete japanese",
        "i don't see any japanese",
        "i do not see any japanese",
        "i don't understand what you",
        "i do not understand what you",
        "i don't understand the japanese",
        "i do not understand the japanese",
        "i don't understand the text",
        "i do not understand the text",
        "i can't provide that translation",
        "i cannot provide that translation",
        "i can't provide a translation",
        "i cannot provide a translation",
        "i can't help with that",
        "i cannot help with that",
        "could you please clarify",
        "please clarify",
        "provide more context",
        "this will help me give",
        "if you have any other questions",
        "feel free to ask",
        "please let me know how i can assist",
        "translated by:",
        "the user provided",
        "the user has requested",
        "the assistant has translated",
        "here is the english translation",
        "here is a translation of your japanese",
        "here's a translation of your japanese",
        "here is the translation of your japanese",
        "here's the translation of your japanese",
        "sure here is a translation",
        "sure here's a translation",
        "sure, here is a translation",
        "sure, here's a translation",
        "here's a short english fragment",
        "short english fragment:",
        "short english rendering",
        "complete translation of that japanese fragment",
        "scene is a short fragment",
        "english translation of the japanese sentence",
        "translation of your japanese sentence",
        "translation of your japanese text",
        "if you have specific information",
        "assist with translations",
        "assist with the translation",
        "provide accurate translations",
        "assistive technology",
        "japanese translator who specializes",
        "translation services",
        "quality assurance",
        "community management",
    )
    return any(pattern in lower for pattern in chatter_patterns)


def looks_like_explanatory_cat_output(text: str) -> bool:
    lower = " ".join(str(text).lower().split())
    if not lower:
        return False
    if looks_like_cat_chatter(text):
        return True
    explanatory_markers = (
        "the japanese phrase",
        "the japanese word",
        "the japanese text",
        "translation of your japanese sentence",
        "translation of your japanese text",
        "here is a translation of your japanese",
        "here's a translation of your japanese",
        "translates to",
        "can be translated as",
        "could be translated as",
        "translation is",
        "correct english translation is",
        "is a sound effect",
        "it means",
        "used in manga",
        "common response",
        "exact nuance",
        "depending on context",
        "context of the",
        "does not require further explanation",
    )
    return any(marker in lower for marker in explanatory_markers)


def looks_repetitive_or_verbose(source_text: str, translated_text: str) -> bool:
    text = str(translated_text).strip()
    source = str(source_text).strip()
    if len(source) <= 24 and len(text) > 180:
        return True
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < 10:
        return False
    most_common = max((words.count(word) for word in set(words)), default=0)
    return most_common >= 8 or most_common / max(1, len(words)) >= 0.6


def has_cat_prompt_fragment(text: str) -> bool:
    lower = " ".join(str(text).lower().split())
    if "japanese:" in lower and "english:" in lower:
        return True
    fragments = (
        "translate the following japanese text",
        "return only the translation",
        "return english translation only",
        "return only a short english fragment",
        "do not explain",
        "preserve ellipses",
        "no explanation",
        "<|im_start|>",
        "<|im_end|>",
        "[/inst]",
        "[inst]",
    )
    return any(fragment in lower for fragment in fragments)


def has_cat_schema_fragment(text: str) -> bool:
    lower = " ".join(str(text).lower().split())
    if "json" in lower or "schema" in lower:
        return True
    return bool(re.search(r"[\{\}]\s*\"(?:translation|source|target|text)\"", text, flags=re.IGNORECASE))


def count_japanese_chars_local(text: str) -> int:
    return sum(is_japanese_char(char) for char in str(text))


def is_japanese_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
    )


def find_cat_gguf_path(preferred: Path | None = None) -> Path | None:
    if preferred is not None and preferred.exists() and preferred.suffix.lower() == ".gguf":
        return preferred.resolve()
    model_dir = CAT_MODEL_DIR
    candidates = list(model_dir.glob("*.gguf")) if model_dir.exists() else []
    if not candidates:
        return None
    q8 = [path for path in candidates if "q8" in path.name.lower()]
    if q8:
        return sorted(q8, key=lambda path: path.stat().st_mtime, reverse=True)[0].resolve()
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0].resolve()


def resolve_cat_gguf_path(model_name: str | Path) -> Path | None:
    if str(model_name) == CAT_MODEL_NAME:
        return find_cat_gguf_path()
    requested = Path(model_name)
    if requested.suffix.lower() == ".gguf":
        return find_cat_gguf_path(requested)
    return None


def model_path_for_debug(model_name: str) -> str:
    path = Path(str(model_name))
    return str(path.resolve()) if path.exists() else str(model_name)


def cat_ollama_model_name(model_path: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", model_path.stem.lower()).strip("-")
    stem = re.sub(r"-+", "-", stem)
    return f"manga-cat-{stem or 'translate'}-hfchat"


def ensure_cat_ollama_model(ollama: str, model_name: str, gguf_path: Path) -> str:
    from .qwen_ollama import ollama_model_exists

    if ollama_model_exists(ollama, model_name):
        logger.info("Reusing existing CAT Ollama model: %s", model_name)
        return model_name
    logger.info("Creating CAT Ollama model: name=%s gguf=%s", model_name, gguf_path)
    modelfile = gguf_path.parent / f"{model_name}.Modelfile"
    modelfile.write_text(
        "\n".join(
            [
                f"FROM {gguf_path.resolve()}",
                "PARAMETER num_ctx 4096",
                "PARAMETER temperature 0",
                "PARAMETER top_p 1",
                "PARAMETER top_k 1",
                'PARAMETER stop "<|im_end|>"',
                'PARAMETER stop "<|im_start|>"',
                'SYSTEM """You are a helpful assistant."""',
                'TEMPLATE """<s><|im_start|>system',
                '{{ .System }}<|im_end|>',
                '<|im_start|>user',
                '{{ .Prompt }}<|im_end|>',
                '<|im_start|>assistant',
                '"""',
                "",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [ollama, "create", model_name, "-f", str(modelfile)],
        cwd=str(Path.cwd()),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not create Ollama model from CAT GGUF. "
            f"stdout={shorten(completed.stdout)} stderr={shorten(completed.stderr)}"
        )
    return model_name


class MadladTranslator(Translator):
    def __init__(
        self,
        model_name: str = MADLAD_MODEL_NAME,
        target_tag: str = "<2en>",
        glossary_path: Path | None = None,
    ) -> None:
        super().__init__(glossary_path)
        logger.info("Initializing MADLAD translator: model=%s target=%s", model_name, target_tag)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError as exc:
            logger.exception("MADLAD dependencies are missing")
            raise RuntimeError(
                "MADLAD translator dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        self._cache: dict[str, str] = {}
        self._target_tag = target_tag
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with local_hf_cache_only():
            try:
                self._tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "MADLAD translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_madlad"
                ) from exc
        model_kwargs = {"low_cpu_mem_usage": True}
        if self._device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        with local_hf_cache_only():
            try:
                self._model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    local_files_only=True,
                    **model_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    "MADLAD translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_madlad"
                ) from exc
        self._model.to(self._device)
        self._model.eval()
        logger.info("MADLAD translator ready on %s", self._device)

    def translate(self, text: str) -> str:
        if text not in self._cache:
            logger.debug("MADLAD translating text=%s", shorten(text))
            self._cache[text] = self._translate_normalized(text)
            logger.debug("MADLAD translated result=%s", shorten(self._cache[text]))
        else:
            logger.debug("MADLAD translation cache hit for text=%s", shorten(text))
        return self._cache[text]

    def _translate_normalized(self, text: str) -> str:
        normalized = normalize_japanese_for_translation(text)
        prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
        phrase = translate_known_phrase(normalized, self._glossary) or translate_known_phrase(prepared, self._glossary)
        if phrase is not None:
            logger.debug("Phrasebook translation: source=%s result=%s", shorten(normalized), shorten(phrase))
            return phrase

        prompt = f"{self._target_tag} {prepared}"
        with local_hf_cache_only():
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=True,
                )
        translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return postprocess_translation(normalized, prepared, translated, self._glossary)


class OpusTranslator(Translator):
    def __init__(self, model_name: str = OPUS_MODEL_NAME, glossary_path: Path | None = None) -> None:
        super().__init__(glossary_path)
        logger.info("Initializing OPUS translator: model=%s", model_name)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            logger.exception("OPUS dependencies are missing")
            raise RuntimeError(
                "OPUS translator dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        self._cache: dict[str, str] = {}
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with local_hf_cache_only():
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "OPUS translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_opus"
                ) from exc
            try:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "OPUS translator model is not installed locally. "
                    "Run: python -m manga_local_translator.install_opus"
                ) from exc
        self._model.to(self._device)
        self._model.eval()
        logger.info("OPUS translator ready on %s", self._device)

    def translate(self, text: str) -> str:
        if text not in self._cache:
            logger.debug("OPUS translating text=%s", shorten(text))
            self._cache[text] = self._translate_normalized(text)
            logger.debug("OPUS translated result=%s", shorten(self._cache[text]))
        else:
            logger.debug("OPUS translation cache hit for text=%s", shorten(text))
        return self._cache[text]

    def translate_with_context(
        self,
        text: str,
        *,
        before: str | None = None,
        after: str | None = None,
        before_contexts: tuple[str, ...] = (),
        after_contexts: tuple[str, ...] = (),
        visual_facts: tuple[str, ...] = (),
    ) -> str:
        _ = before_contexts, after_contexts, visual_facts
        normalized = normalize_japanese_for_translation(text)
        if not should_use_context_translation(normalized, before=before, after=after):
            return self.translate(text)

        context_units = [
            normalize_japanese_for_translation(value)
            for value in (before, text, after)
            if value
        ]
        joined = "\u3002".join(context_units)
        logger.debug("OPUS translating with context: text=%s joined=%s", shorten(text), shorten(joined))
        contextual = self._translate_normalized(joined)
        focused = self.translate(text)
        return choose_contextual_translation(
            focused=focused,
            contextual=contextual,
            source_text=normalized,
            before_count=1 if before else 0,
            source_unit_count=max(1, len(split_japanese_translation_units(normalized))),
        )

    def _translate_normalized(self, text: str) -> str:
        normalized = normalize_japanese_for_translation(text)
        prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
        phrase = translate_known_phrase(normalized, self._glossary) or translate_known_phrase(prepared, self._glossary)
        if phrase is not None:
            logger.debug("Phrasebook translation: source=%s result=%s", shorten(normalized), shorten(phrase))
            return phrase

        units = split_japanese_translation_units(prepared)
        if len(units) > 1:
            translations = [self._translate_unit(unit) for unit in units]
            translated = " ".join(translation for translation in translations if translation).strip()
            return postprocess_translation(normalized, prepared, translated, self._glossary)

        translated = self._translate_unit(prepared)
        return postprocess_translation(normalized, prepared, translated, self._glossary)

    def _translate_unit(self, text: str) -> str:
        unit = clean_translation_unit(text)
        phrase = translate_known_phrase(unit, self._glossary)
        if phrase is not None:
            return phrase

        with local_hf_cache_only():
            inputs = self._tokenizer(unit, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=5,
                    do_sample=False,
                    early_stopping=True,
                )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


@contextmanager
def local_hf_cache_only():
    previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    captured_stderr = StringIO()
    try:
        with redirect_stderr(captured_stderr):
            yield
    finally:
        restore_env_var("TRANSFORMERS_OFFLINE", previous_transformers_offline)
        restore_env_var("HF_HUB_OFFLINE", previous_hf_offline)
        captured = captured_stderr.getvalue().strip()
        if captured:
            logger.debug("Suppressed local Hugging Face load stderr: %s", shorten(captured))


def restore_env_var(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
