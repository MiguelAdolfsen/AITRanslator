from __future__ import annotations

import base64
import contextlib
import io
import logging
import mimetypes
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .qwen_ollama import QWEN_MODEL_DIR, find_qwen_model_path, qwen_ollama_model_name
from .qwen_types import QwenGenerationSettings

logger = logging.getLogger(__name__)


VISION_SETTINGS = QwenGenerationSettings(temperature=0.1, top_p=0.8, top_k=20, min_p=0.0, num_predict=768)
VISION_JSON_REPAIR_SETTINGS = QwenGenerationSettings(temperature=0.1, top_p=0.8, top_k=20, min_p=0.0, num_predict=768)


def find_qwen_vision_projector(preferred: Path | None = None) -> Path:
    if preferred is not None:
        if preferred.exists():
            return preferred
        raise RuntimeError(f"Requested Qwen vision projector was not found: {preferred}")
    candidates = sorted(QWEN_MODEL_DIR.glob("*mmproj*.gguf")) if QWEN_MODEL_DIR.exists() else []
    if candidates:
        return candidates[0]
    return QWEN_MODEL_DIR / "mmproj-Qwen3.5-9B-BF16.gguf"


class QwenVisionClient:
    def __init__(
        self,
        *,
        model_path: Path | None = None,
        projector_path: Path | None = None,
        n_ctx: int = 8192,
    ) -> None:
        self.model_path = model_path or find_qwen_model_path()
        self.projector_path = find_qwen_vision_projector(projector_path)
        if not self.model_path.exists():
            raise RuntimeError(f"Qwen vision model was not found: {self.model_path}")
        if not self.projector_path.exists():
            raise RuntimeError(f"Qwen vision projector was not found: {self.projector_path}")
        logger.info("Initializing Qwen vision client: model=%s projector=%s", self.model_path, self.projector_path)
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        except ImportError as exc:
            raise RuntimeError("Vision repair needs llama-cpp-python with Qwen vision chat handler support.") from exc
        with suppress_native_output():
            chat_handler = Qwen25VLChatHandler(clip_model_path=str(self.projector_path), verbose=False)
            self._llm = Llama(
                model_path=str(self.model_path),
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                verbose=False,
            )
        self.model_name = qwen_ollama_model_name(self.model_path) + "-vision"

    def repair(self, prompt: str, image_path: Path) -> str:
        image_url = image_data_url(image_path)
        return self._completion(
            [
                {
                    "role": "system",
                    "content": "You are a manga translation QA assistant. Return JSON only. Do not OCR image text.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            settings=VISION_SETTINGS,
        )

    def extract_facts(self, prompt: str, image_path: Path) -> str:
        image_url = image_data_url(image_path)
        return self._completion(
            [
                {
                    "role": "system",
                    "content": "You extract grounded manga visual facts. Return JSON only. Do not translate and do not OCR image text.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            settings=VISION_SETTINGS,
        )

    def repair_json(self, prompt: str) -> str:
        return self._completion(
            [
                {
                    "role": "system",
                    "content": "You repair malformed JSON. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            settings=VISION_JSON_REPAIR_SETTINGS,
        )

    def _completion(self, messages, *, settings: QwenGenerationSettings) -> str:
        captured = io.StringIO()
        with suppress_native_output(), contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=settings.temperature,
                top_p=settings.top_p,
                top_k=settings.top_k,
                min_p=settings.min_p,
                max_tokens=settings.num_predict,
                response_format={"type": "json_object"},
                seed=12345,
            )
        if captured.getvalue().strip():
            logger.debug("Suppressed Qwen vision backend output: %s", captured.getvalue().strip()[:1000])
        return str(response["choices"][0]["message"]["content"]).strip()


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@contextmanager
def suppress_native_output():
    """Suppress native stdout/stderr writes from llama.cpp on Windows and POSIX."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        null_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
        finally:
            for fd in (stdout_fd, stderr_fd, null_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass
