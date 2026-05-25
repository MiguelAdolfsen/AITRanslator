from __future__ import annotations

import logging
import os
import re
import subprocess
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
CAT_PROMPT_VERSION = "cat-translate-v2"
CAT_VALIDATION_VERSION = "cat-validation-v2"


class CatTranslator(Translator):
    def __init__(self, model_name: str | Path = CAT_MODEL_NAME, glossary_path: Path | None = None) -> None:
        super().__init__(glossary_path)
        requested = Path(model_name)
        self._gguf_path = find_cat_gguf_path(requested) if model_name == CAT_MODEL_NAME or requested.suffix.lower() == ".gguf" or requested.exists() else None
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
        self._cat_settings = QwenGenerationSettings(temperature=0.0, top_p=1.0, top_k=1, min_p=0.0, num_predict=128)
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

    def debug_info_for(self, text: str, debug_id: str | None = None) -> dict[str, object]:
        _ = debug_id
        return dict(self._debug.get(text, {}))

    def _translate_normalized(self, text: str) -> tuple[str, dict[str, object]]:
        normalized = normalize_japanese_for_translation(text)
        prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
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
                "cat_rejected": False,
                "cat_final": phrase,
            }

        prompt = build_cat_prompt(prepared)
        if getattr(self, "_backend", "") == "ollama":
            raw = self._run_cat_prompt(self._ollama, self._ollama_model_name, prompt, settings=self._cat_settings).strip()
            translated, result, reject_reason = finalize_cat_translation(normalized, prepared, raw, self._glossary)
            return result, {
                "primary_translator": "cat",
                "cat_used": True,
                "cat_model": self._ollama_model_name,
                "cat_model_path": str(self._gguf_path),
                "cat_backend": "ollama",
                "cat_prompt_version": CAT_PROMPT_VERSION,
                "cat_validation_version": CAT_VALIDATION_VERSION,
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
            "cat_prompt_version": CAT_PROMPT_VERSION,
            "cat_validation_version": CAT_VALIDATION_VERSION,
            "cat_prompt": prompt,
            "cat_raw_translation": raw,
            "cat_cleaned_translation": translated,
            "cat_candidate": translated,
            "cat_final": result,
            "cat_rejected": reject_reason is not None,
            "cat_reject_reason": reject_reason or "",
            "cat_chatter_rejected": reject_reason in {"cat_chatter", "cat_prompt_fragment", "cat_schema_fragment"},
        }


def build_cat_prompt(prepared_text: str) -> str:
    return (
        "Translate the following Japanese text into English. "
        "Return only the translation, no explanation.\n\n"
        f"{prepared_text}"
    )


def finalize_cat_translation(source_text: str, prepared_text: str, raw_text: str, glossary) -> tuple[str, str, str | None]:
    cleaned = clean_cat_output(raw_text)
    reject_reason = cat_reject_reason(source_text, cleaned, raw_text)
    if reject_reason is not None:
        return cleaned, "", reject_reason
    result = postprocess_translation(source_text, prepared_text, cleaned, glossary)
    final_reject_reason = cat_reject_reason(source_text, result, raw_text)
    if final_reject_reason is not None:
        return cleaned, "", final_reject_reason
    return cleaned, result, None


def clean_cat_output(text: str) -> str:
    cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", str(text)).strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.lower().startswith("translation:"):
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
    if has_cat_prompt_fragment(combined):
        return "cat_prompt_fragment"
    if has_cat_schema_fragment(combined):
        return "cat_schema_fragment"
    if count_japanese_chars_local(text) > 0:
        return "cat_untranslated_japanese"
    if re.search(r"\b[A-Za-z]+(?:-[A-Za-z]+){4,}\b", text):
        return "cat_malformed_hyphen_chain"
    if source_text.strip() and text.strip() == source_text.strip():
        return "cat_unchanged_source"
    return None


def looks_like_cat_chatter(text: str) -> bool:
    lower = " ".join(str(text).lower().split())
    chatter_patterns = (
        "please provide the japanese",
        "could you please provide",
        "i don't see any japanese",
        "i do not see any japanese",
        "i don't understand what you",
        "i do not understand what you",
        "i can't provide that translation",
        "i cannot provide that translation",
        "please let me know how i can assist",
        "japanese translator who specializes",
        "translation services",
        "quality assurance",
        "community management",
    )
    return any(pattern in lower for pattern in chatter_patterns)


def has_cat_prompt_fragment(text: str) -> bool:
    lower = " ".join(str(text).lower().split())
    fragments = (
        "translate the following japanese text",
        "return only the translation",
        "return english translation only",
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


def model_path_for_debug(model_name: str) -> str:
    path = Path(str(model_name))
    return str(path.resolve()) if path.exists() else str(model_name)


def cat_ollama_model_name(model_path: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", model_path.stem.lower()).strip("-")
    stem = re.sub(r"-+", "-", stem)
    return f"manga-cat-{stem or 'translate'}"


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
                'SYSTEM """You are a professional Japanese-to-English translation engine. Return the English translation only."""',
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
