from __future__ import annotations

import argparse
import importlib.util


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional DSPy/MIPROv2 adapter stub.")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("dspy") is None:
        message = "DSPy is not installed. Install it manually to optimize instructions/few-shot examples against translation_quality_score."
        if args.strict:
            raise SystemExit(message)
        print(message)
        return 0
    print("DSPy is available, but MIPROv2 optimization is not implemented in the baseline harness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
