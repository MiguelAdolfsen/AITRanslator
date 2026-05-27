from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed benchmark is checked into this harness.")
    parser.add_argument("--output", default="translation_quality_autoresearch/benchmark")
    parser.parse_args(argv)
    print("Seed benchmark files already exist. Edit this script only when creating a new benchmark version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
