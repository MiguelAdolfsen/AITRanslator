from __future__ import annotations

import argparse
import logging
import os

from .hf_translators import HY_MT2_MODEL_NAME

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Tencent Hy-MT2 into the local Hugging Face cache.")
    parser.add_argument("--model", default=HY_MT2_MODEL_NAME, help="Hugging Face model id to download.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Run: pip install -r requirements.txt") from exc

    logger.info("Downloading Tencent Hy-MT2 translator model: %s", args.model)
    path = snapshot_download(repo_id=args.model)
    logger.info("Tencent Hy-MT2 translator model is ready at: %s", path)
    print(f"Tencent Hy-MT2 translator model is ready at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
