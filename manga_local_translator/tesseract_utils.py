from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

COMMON_TESSERACT_PATHS = [
    Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
]


def find_tesseract() -> str | None:
    path_from_env = shutil.which("tesseract")
    if path_from_env:
        logger.info("Found tesseract on PATH: %s", path_from_env)
        return path_from_env

    for candidate in COMMON_TESSERACT_PATHS:
        if candidate.exists():
            logger.info("Found tesseract at common install path: %s", candidate)
            return str(candidate)

    logger.warning("Tesseract executable was not found on PATH or common Windows install paths")
    return None


def tesseract_missing_message() -> str:
    common_paths = "\n".join(f"- {path}" for path in COMMON_TESSERACT_PATHS)
    return (
        "Tesseract OCR is not installed or the app cannot find tesseract.exe.\n\n"
        "Install the Windows Tesseract OCR build, then restart this launcher:\n"
        "https://github.com/UB-Mannheim/tesseract/wiki\n\n"
        "During install, include Japanese language data if possible.\n\n"
        "After installing, the path is usually one of:\n"
        f"{common_paths}\n\n"
        "If it installs somewhere else, use the Tesseract path Browse button and select tesseract.exe."
    )

