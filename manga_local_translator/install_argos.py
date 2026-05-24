from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install a free Argos Translate model.")
    parser.add_argument("source", nargs="?", default="ja", help="Source language code.")
    parser.add_argument("target", nargs="?", default="en", help="Target language code.")
    args = parser.parse_args(argv)
    logger.info("Argos model install requested: %s -> %s", args.source, args.target)

    import argostranslate.package

    print("Updating Argos package index...")
    logger.info("Updating Argos package index")
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    logger.info("Argos package index returned %d package(s)", len(packages))
    package = next(
        (
            item
            for item in packages
            if item.from_code == args.source and item.to_code == args.target
        ),
        None,
    )
    if package is None:
        logger.error("No Argos package found for %s -> %s", args.source, args.target)
        raise SystemExit(f"No Argos package found for {args.source} -> {args.target}")

    print(f"Downloading {args.source} -> {args.target} model...")
    logger.info("Downloading Argos model: %s -> %s", args.source, args.target)
    package_path = package.download()
    logger.info("Downloaded Argos model to: %s", package_path)
    print("Installing model...")
    logger.info("Installing Argos model from: %s", package_path)
    argostranslate.package.install_from_path(package_path)
    print("Done.")
    logger.info("Argos model install complete")
    return 0


if __name__ == "__main__":
    from .logging_utils import configure_logging

    configure_logging(reset=True)
    raise SystemExit(main())
