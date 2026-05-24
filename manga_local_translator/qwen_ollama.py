from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .logging_utils import shorten
from .qwen_types import QwenGenerationSettings

logger = logging.getLogger(__name__)
QWEN_MODEL_DIR = Path(".models") / "qwen"


def find_qwen_model_path(preferred: str | Path | None = None) -> Path:
    model_dir = QWEN_MODEL_DIR
    env_preferred = os.environ.get("MANGA_QWEN_MODEL")
    requested = Path(preferred or env_preferred) if preferred or env_preferred else None
    if requested is not None:
        if requested.exists():
            return requested
        if not requested.is_absolute():
            named = model_dir / requested
            if named.exists():
                return named
        raise RuntimeError(f"Requested Qwen model was not found: {requested}")

    candidates = list(model_dir.glob("*.gguf")) if model_dir.exists() else []
    text_models = [path for path in candidates if is_qwen_text_model_file(path)]
    preferred_model = select_preferred_qwen_text_model(text_models)
    if preferred_model is not None:
        return preferred_model
    return model_dir / "Qwen3-8B-Q4_K_M.gguf"


def select_preferred_qwen_text_model(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    q4_35 = [path for path in candidates if "qwen3.5" in path.name.lower() and "q4_k_m" in path.name.lower()]
    if q4_35:
        return sorted(q4_35, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    q4 = [path for path in candidates if "q4_k_m" in path.name.lower()]
    if q4:
        return sorted(q4, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def is_qwen_text_model_file(path: Path) -> bool:
    name = path.name.lower()
    excluded_markers = ("mmproj", "vision", "visual", "projector", "encoder", "clip")
    return not any(marker in name for marker in excluded_markers)


def qwen_ollama_model_name(model_path: Path) -> str:
    stem = model_path.stem.lower()
    stem = stem.replace("qwen3.5", "qwen35")
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    stem = re.sub(r"-+", "-", stem)
    return f"manga-{stem or 'qwen'}"


def find_ollama_executable() -> str | None:
    candidates = [
        shutil.which("ollama"),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"),
        str(Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            logger.debug("Found Ollama executable: %s", candidate)
            return candidate
    return None


def ensure_ollama_model(ollama: str, model_name: str, gguf_path: Path) -> str:
    actual_model_name = model_name
    if ollama_model_exists(ollama, actual_model_name):
        if not ollama_model_has_projector(ollama, actual_model_name):
            return actual_model_name
        actual_model_name = f"{model_name}-textonly"
        logger.info(
            "Ollama model %s has an attached vision projector; using text-only model %s",
            model_name,
            actual_model_name,
        )
        if ollama_model_exists(ollama, actual_model_name):
            if not ollama_model_has_projector(ollama, actual_model_name):
                return actual_model_name
            raise RuntimeError(f"Ollama text-only Qwen model still has a vision projector: {actual_model_name}")

    modelfile = gguf_path.parent / f"{actual_model_name}.Modelfile"
    modelfile.write_text(
        "\n".join(
            [
                f"FROM {gguf_path.resolve()}",
                "PARAMETER num_ctx 8192",
                f"PARAMETER temperature {QWEN_TRANSLATION_SETTINGS.temperature}",
                f"PARAMETER top_p {QWEN_TRANSLATION_SETTINGS.top_p}",
                f"PARAMETER top_k {QWEN_TRANSLATION_SETTINGS.top_k}",
                f"PARAMETER min_p {QWEN_TRANSLATION_SETTINGS.min_p}",
                'SYSTEM """You are a professional Japanese-to-English manga dialogue translator. Return concise English only."""',
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Creating Ollama Qwen model: name=%s modelfile=%s", actual_model_name, modelfile)
    completed = subprocess.run(
        [ollama, "create", actual_model_name, "-f", str(modelfile)],
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
            "Could not create Ollama model from Qwen GGUF. "
            f"stdout={shorten(completed.stdout)} stderr={shorten(completed.stderr)}"
        )
    if ollama_model_has_projector(ollama, actual_model_name):
        raise RuntimeError(f"Ollama Qwen model unexpectedly has a vision projector: {actual_model_name}")
    logger.info("Ollama Qwen model is ready: %s", actual_model_name)
    return actual_model_name


def ollama_model_exists(ollama: str, model_name: str) -> bool:
    completed = subprocess.run(
        [ollama, "list"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        logger.warning("Could not list Ollama models: %s", shorten(completed.stderr))
        return False
    pattern = re.compile(rf"^{re.escape(model_name)}(?::|\s)", re.MULTILINE)
    exists = pattern.search(completed.stdout) is not None
    logger.debug("Ollama model exists check: model=%s exists=%s", model_name, exists)
    return exists


def ollama_model_has_projector(ollama: str, model_name: str) -> bool:
    completed = subprocess.run(
        [ollama, "show", model_name],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        logger.warning("Could not inspect Ollama model %s: %s", model_name, shorten(completed.stderr))
        return False
    has_projector = re.search(r"^\s*Projector\s*$", completed.stdout, flags=re.MULTILINE) is not None
    logger.debug("Ollama model projector check: model=%s has_projector=%s", model_name, has_projector)
    return has_projector


def run_ollama_prompt(
    ollama: str | None,
    model_name: str,
    prompt: str,
    *,
    settings: QwenGenerationSettings,
) -> str:
    if not ollama:
        return ""
    api_result = run_ollama_api_prompt(model_name, prompt, settings=settings)
    if api_result is not None:
        return api_result

    completed = subprocess.run(
        [ollama, "run", model_name, prompt],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        logger.error("Ollama Qwen translation failed: %s", shorten(completed.stderr))
        return ""
    return completed.stdout.strip()


def run_ollama_api_prompt(
    model_name: str,
    prompt: str,
    *,
    settings: QwenGenerationSettings,
) -> str | None:
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "top_k": settings.top_k,
                "min_p": settings.min_p,
                "num_predict": settings.num_predict,
                "num_ctx": 8192,
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        logger.debug("Ollama API prompt failed; falling back to CLI", exc_info=True)
        return None
    return str(data.get("response", "")).strip()
