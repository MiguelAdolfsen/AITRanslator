from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stub for prompt variant experiments.")
    parser.add_argument("--variants", type=Path, default=Path("translation_quality_autoresearch/prompts/variants"))
    parser.add_argument("--output", type=Path, default=Path("translation_quality_autoresearch/runs/prompt_search"))
    args = parser.parse_args(argv)
    if not args.variants.exists():
        print(f"No prompt variants found at {args.variants}; nothing to run.")
        return 0
    variants = [path for path in args.variants.iterdir() if path.is_dir()]
    print(f"Found {len(variants)} prompt variant(s). Full live evaluation integration is intentionally deferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
