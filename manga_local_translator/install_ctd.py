from __future__ import annotations

import argparse
import logging

from .ctd_setup import CTD_MODEL_PATH, CTD_SOURCE_DIR, install_ctd

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install Comic Text Detector model files.")
    parser.parse_args(argv)

    logger.info("CTD install requested")
    install_ctd()
    print(f"CTD source: {CTD_SOURCE_DIR.resolve()}")
    print(f"CTD model:  {CTD_MODEL_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    from .logging_utils import configure_logging

    configure_logging(reset=True)
    raise SystemExit(main())

