from __future__ import annotations

import logging
import os

from .translate import MADLAD_MODEL_NAME

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Run: pip install -r requirements.txt") from exc

    logger.info("Downloading MADLAD translator model: %s", MADLAD_MODEL_NAME)
    path = snapshot_download(
        repo_id=MADLAD_MODEL_NAME,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "added_tokens.json",
            "model.safetensors",
            "spiece.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )
    logger.info("MADLAD translator model is ready at: %s", path)
    print(f"MADLAD translator model is ready at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
