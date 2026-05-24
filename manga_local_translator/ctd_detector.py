from __future__ import annotations

import importlib
import importlib.machinery
import logging
import sys
import types
import warnings
from pathlib import Path

import numpy as np

from .ctd_setup import CTD_MODEL_PATH, CTD_SOURCE_DIR, require_ctd_installed
from .detect_types import TextBlock
from .logging_utils import configure_logging

logger = logging.getLogger(__name__)
_CTD_MODEL = None


def detect_ctd_text_blocks(image_path: Path) -> list[TextBlock]:
    import cv2

    require_ctd_installed()
    model = load_ctd_model()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image for CTD detection: {image_path}")

    inference = importlib.import_module("inference")
    _mask, _mask_refined, block_list = model(
        image_bgr,
        refine_mode=inference.REFINEMASK_ANNOTATION,
        keep_undetected_mask=False,
    )
    height, width = image_bgr.shape[:2]
    blocks: list[TextBlock] = []
    for block in block_list:
        box = clamp_box(tuple(int(value) for value in block.xyxy), width, height)
        if box_is_empty(box):
            continue
        lines = [np.array(line).astype(int).reshape(-1, 2).tolist() for line in getattr(block, "lines", [])]
        blocks.append(
            TextBlock(
                text="",
                box=box,
                confidence=float(getattr(block, "prob", 0.0)),
                detector="ctd",
                metadata={
                    "language": getattr(block, "language", "unknown"),
                    "vertical": bool(getattr(block, "vertical", False)),
                    "lines": lines,
                },
            )
        )

    logger.info("CTD detected %d text block(s)", len(blocks))
    return blocks


def load_ctd_model():
    global _CTD_MODEL
    if _CTD_MODEL is not None:
        return _CTD_MODEL

    source_path = str(CTD_SOURCE_DIR.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    install_ctd_import_stubs()
    install_numpy_compatibility_shims()
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    logger.info("Loading CTD model: %s", CTD_MODEL_PATH)
    inference = importlib.import_module("inference")
    configure_logging(reset=False)
    _CTD_MODEL = inference.TextDetector(
        model_path=str(CTD_MODEL_PATH.resolve()),
        input_size=1024,
        device="cpu",
        act="leaky",
    )
    logger.info("CTD model loaded")
    return _CTD_MODEL


def install_ctd_import_stubs() -> None:
    if "wandb" not in sys.modules:
        wandb = types.ModuleType("wandb")
        wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.log_model = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb


def install_numpy_compatibility_shims() -> None:
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64


def clamp_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(image_width, x1)),
        max(0, min(image_height, y1)),
        max(0, min(image_width, x2)),
        max(0, min(image_height, y2)),
    )


def box_is_empty(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x2 <= x1 or y2 <= y1
