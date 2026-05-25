from __future__ import annotations

import argparse

from .hf_translators import CAT_MODEL_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download CAT-Translate into the local Hugging Face cache.")
    parser.add_argument("--model", default=CAT_MODEL_NAME, help="Hugging Face model id to download.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing huggingface_hub. Run: pip install -r requirements.txt") from exc

    snapshot_download(args.model)
    print(f"CAT model downloaded: {args.model}")


if __name__ == "__main__":
    main()
