from __future__ import annotations

import argparse
import importlib.util


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional GEPA-style optimizer stub.")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    missing = [name for name in ("dspy",) if importlib.util.find_spec(name) is None]
    if missing:
        message = "DSPy/GEPA dependencies are not installed. Future GEPA-style optimization should use full traces and textual feedback."
        if args.strict:
            raise SystemExit(message)
        print(message)
        return 0
    print("GEPA adapter dependencies are available, but optimization is not implemented in the baseline harness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
