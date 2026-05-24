from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


REQUIRED_PACKAGES = [
    "pillow",
    "opencv-python",
    "numpy",
    "pytesseract",
    "tqdm",
    "manga-ocr",
    "transformers",
    "sentencepiece",
    "accelerate",
    "torchvision",
    "shapely",
    "pyclipper",
    "torchsummary",
]


def ensure_python_package(import_name: str, package_name: str) -> None:
    if importlib.util.find_spec(import_name) is not None:
        logger.debug("Python package already available: import=%s package=%s", import_name, package_name)
        return

    logger.info("Installing missing Python package: %s", package_name)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            package_name,
        ]
    )
    logger.info("Installed Python package: %s", package_name)


def ensure_runtime_dependencies() -> None:
    logger.info("Checking runtime Python dependencies")
    for package_name in REQUIRED_PACKAGES:
        import_name = package_name.replace("-", "_")
        if package_name == "opencv-python":
            import_name = "cv2"
        elif package_name == "pillow":
            import_name = "PIL"
        elif package_name == "sentencepiece":
            import_name = "sentencepiece"

        ensure_python_package(import_name, package_name)
    logger.info("Runtime Python dependency check complete")
