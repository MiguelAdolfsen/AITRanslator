from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "official": {
        "name": "official",
        "prompt_template": "Translate the following Japanese text into English.\n\n{source_text}",
        "retry_prompt_template": "Japanese:\n{source_text}\n\nEnglish:",
        "system_prompt": "You are a machine translation engine. Return only the English translation.",
        "num_predict": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "min_p": 0.0,
        "stop": ["<|im_end|>", "\n\n"],
    },
    "fragment_strict": {
        "name": "fragment_strict",
        "prompt_template": (
            "Translate exactly. Return only concise English. Do not explain, apologize, ask for clarification, "
            "or continue the scene. If the source is incomplete, translate the fragment as a fragment.\n"
            "Japanese: \"{source_text}\"\nEnglish:"
        ),
        "retry_prompt_template": (
            "Translate exactly. If the source is incomplete, translate the incomplete fragment. Never ask for clarification.\n"
            "Japanese: \"{source_text}\"\nEnglish:"
        ),
        "system_prompt": "You are a strict Japanese-to-English machine translation engine.",
        "num_predict": 48,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "min_p": 0.0,
        "stop": ["<|im_end|>", "\n\n"],
    },
}


def load_profile(name: str = "official", profile_json: Path | None = None) -> dict[str, Any]:
    if profile_json is not None:
        payload = json.loads(profile_json.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Profile must be a JSON object: {profile_json}")
        return {**BUILTIN_PROFILES["official"], **payload}
    if name not in BUILTIN_PROFILES:
        raise RuntimeError(f"Unknown CAT profile {name!r}. Known profiles: {sorted(BUILTIN_PROFILES)}")
    return dict(BUILTIN_PROFILES[name])


def profile_hash(profile: dict[str, Any]) -> str:
    payload = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_prompt(source_text: str, profile: dict[str, Any], *, retry: bool = False) -> str:
    key = "retry_prompt_template" if retry else "prompt_template"
    template = str(profile.get(key) or BUILTIN_PROFILES["official"][key])
    return template.format(source_text=source_text)


def load_fake_outputs(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    outputs: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            outputs[str(row["case_id"])] = str(row.get("raw_output", ""))
    return outputs


def apply_stop_strings(text: str, stop: Any) -> str:
    result = str(text)
    if not isinstance(stop, list):
        return result.strip()
    for marker in stop:
        marker_text = str(marker)
        if marker_text and marker_text in result:
            result = result.split(marker_text, 1)[0]
    return result.strip()


def run_cat_case(case: dict[str, Any], profile: dict[str, Any], *, fake_outputs: dict[str, str] | None = None) -> dict[str, Any]:
    from manga_local_translator.hf_translators import finalize_cat_translation

    fake_outputs = fake_outputs or {}
    source_text = str(case.get("source_text", ""))
    prompt = build_prompt(source_text, profile)
    started = time.perf_counter()
    model_error = ""
    if str(case["case_id"]) in fake_outputs:
        raw = fake_outputs[str(case["case_id"])]
        backend = "fake"
        model_name = "fake-cat"
    else:
        try:
            raw, backend, model_name = run_live_cat_prompt(prompt, profile)
        except Exception as exc:  # pragma: no cover - exercised manually with local model
            raw = ""
            backend = "ollama"
            model_name = ""
            model_error = str(exc)
    latency_ms = (time.perf_counter() - started) * 1000.0
    raw = apply_stop_strings(raw, profile.get("stop"))
    cleaned, final, reject_reason = finalize_cat_translation(source_text, source_text, raw, None)
    if model_error and not reject_reason:
        reject_reason = "model_error"
    return {
        "case_id": case.get("case_id"),
        "source_text": source_text,
        "profile_name": profile.get("name", "custom"),
        "profile_hash": profile_hash(profile),
        "backend": backend,
        "model": model_name,
        "prompt": prompt,
        "raw_output": raw,
        "cleaned_output": cleaned,
        "final_output": final,
        "reject_reason": reject_reason or "",
        "model_error": model_error,
        "latency_ms": round(latency_ms, 3),
    }


def run_live_cat_prompt(prompt: str, profile: dict[str, Any]) -> tuple[str, str, str]:
    from manga_local_translator.hf_translators import cat_ollama_model_name, find_cat_gguf_path, ensure_cat_ollama_model
    from manga_local_translator.qwen_ollama import find_ollama_executable, run_ollama_prompt
    from manga_local_translator.qwen_types import QwenGenerationSettings

    gguf = find_cat_gguf_path()
    ollama = find_ollama_executable()
    if not ollama:
        raise RuntimeError("Ollama was not found. Install/start Ollama to use CAT GGUF models.")
    model_name = ensure_cat_ollama_model(ollama, cat_ollama_model_name(gguf), gguf)
    settings = QwenGenerationSettings(
        temperature=float(profile.get("temperature", 0.0)),
        top_p=float(profile.get("top_p", 1.0)),
        top_k=int(profile.get("top_k", 1)),
        min_p=float(profile.get("min_p", 0.0)),
        num_predict=int(profile.get("num_predict", 128)),
    )
    return run_ollama_prompt(ollama, model_name, prompt, settings=settings), "ollama", model_name

