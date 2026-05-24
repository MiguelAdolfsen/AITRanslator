from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOG_FILE_NAME = "manga_translator_debug.log"


def default_log_path() -> Path:
    return Path.cwd() / LOG_FILE_NAME


def configure_logging(*, reset: bool = True, log_path: Path | None = None) -> Path:
    path = (log_path or default_log_path()).resolve()
    if reset and path.exists():
        try:
            path.unlink()
        except OSError:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = path.with_name(f"{path.stem}-{timestamp}{path.suffix}")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )
    logging.getLogger(__name__).info("Debug logging started: %s", path)
    return path


def shorten(value: str, *, limit: int = 160) -> str:
    value = value.replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."
