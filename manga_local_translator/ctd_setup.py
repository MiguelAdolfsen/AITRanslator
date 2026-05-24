from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

CTD_ROOT = Path(".models") / "comic-text-detector"
CTD_SOURCE_DIR = CTD_ROOT / "source"
CTD_MODEL_PATH = CTD_ROOT / "comictextdetector.pt.onnx"
CTD_SOURCE_ZIP = CTD_ROOT / "comic-text-detector.zip"
CTD_SOURCE_URL = "https://github.com/dmMaze/comic-text-detector/archive/refs/heads/master.zip"
CTD_MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/"
    "beta-0.2.1/comictextdetector.pt.onnx"
)


def ctd_is_installed() -> bool:
    return CTD_SOURCE_DIR.exists() and CTD_MODEL_PATH.exists()


def require_ctd_installed() -> None:
    if ctd_is_installed():
        return
    raise RuntimeError(
        "Comic Text Detector is not installed.\n\n"
        "In the GUI, click 'Install CTD detector' first. "
        "From CLI, run: python -m manga_local_translator.install_ctd"
    )


def install_ctd() -> None:
    CTD_ROOT.mkdir(parents=True, exist_ok=True)
    if not CTD_SOURCE_DIR.exists():
        download_file(CTD_SOURCE_URL, CTD_SOURCE_ZIP)
        extract_source_zip(CTD_SOURCE_ZIP, CTD_SOURCE_DIR)
    else:
        logger.info("CTD source already installed: %s", CTD_SOURCE_DIR)

    if not CTD_MODEL_PATH.exists():
        download_file(CTD_MODEL_URL, CTD_MODEL_PATH)
    else:
        logger.info("CTD model already installed: %s", CTD_MODEL_PATH)


def download_file(url: str, destination: Path) -> None:
    logger.info("Downloading %s -> %s", url, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    logger.info("Downloaded: %s", destination)


def extract_source_zip(zip_path: Path, destination: Path) -> None:
    logger.info("Extracting CTD source: %s", zip_path)
    temp_dir = CTD_ROOT / "_extract"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(temp_dir)

    extracted_roots = [path for path in temp_dir.iterdir() if path.is_dir()]
    if len(extracted_roots) != 1:
        raise RuntimeError(f"Unexpected CTD source archive layout: {zip_path}")

    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(extracted_roots[0]), str(destination))
    shutil.rmtree(temp_dir)
    logger.info("CTD source installed: %s", destination)

