from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .detect_types import TextBlock
from .logging_utils import shorten
from .tesseract_utils import find_tesseract, tesseract_missing_message
from .text_filter import is_ctd_horizontal_metadata, normalize_for_filter

logger = logging.getLogger(__name__)
_MANGA_OCR = None


def configure_tesseract(tesseract_cmd: str | None) -> None:
    import pytesseract

    resolved_cmd = tesseract_cmd or find_tesseract()
    if not resolved_cmd:
        logger.warning("Tesseract path is not configured and auto-detection failed")
        return

    pytesseract.pytesseract.tesseract_cmd = resolved_cmd
    logger.info("Using tesseract executable: %s", resolved_cmd)


def run_tesseract_ocr(
    image_path: Path,
    *,
    lang: str,
    psm: int,
    min_confidence: float,
) -> list[TextBlock]:
    import pytesseract
    from pytesseract import Output

    logger.info("Opening image for OCR: %s", image_path)
    image = Image.open(image_path).convert("RGB")
    config = f"--psm {psm} --oem 1"
    logger.debug("Tesseract config: lang=%s config=%s min_confidence=%.2f", lang, config, min_confidence)
    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            output_type=Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        logger.exception("Tesseract executable was not found")
        raise RuntimeError(tesseract_missing_message()) from exc

    raw_count = len(data["text"])
    accepted_count = 0
    rejected_empty = 0
    rejected_confidence = 0
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, raw_text in enumerate(data["text"]):
        text = normalize_ocr_text(raw_text)
        if not text:
            rejected_empty += 1
            continue
        confidence = parse_confidence(data["conf"][index])
        if confidence < min_confidence:
            rejected_confidence += 1
            logger.debug(
                "Rejected OCR item below confidence: index=%d confidence=%.2f text=%s",
                index,
                confidence,
                shorten(text),
            )
            continue

        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped[key].append(index)
        accepted_count += 1

    logger.info(
        "OCR words processed: raw=%d accepted=%d empty=%d low_confidence=%d groups=%d",
        raw_count,
        accepted_count,
        rejected_empty,
        rejected_confidence,
        len(grouped),
    )

    blocks: list[TextBlock] = []
    for indexes in grouped.values():
        texts = [normalize_ocr_text(data["text"][index]) for index in indexes]
        texts = [text for text in texts if text]
        if not texts:
            continue

        left = min(int(data["left"][index]) for index in indexes)
        top = min(int(data["top"][index]) for index in indexes)
        right = max(int(data["left"][index]) + int(data["width"][index]) for index in indexes)
        bottom = max(int(data["top"][index]) + int(data["height"][index]) for index in indexes)
        confidence = sum(parse_confidence(data["conf"][index]) for index in indexes) / len(indexes)
        text = "".join(texts)
        refined_box = shrink_box_to_dark_pixels(image, (left, top, right, bottom))
        block = TextBlock(text=text, box=refined_box, confidence=confidence, detector="tesseract")
        logger.debug(
            "OCR block: raw_box=%s refined_box=%s confidence=%.2f text=%s",
            (left, top, right, bottom),
            block.box,
            block.confidence,
            shorten(block.text),
        )
        blocks.append(block)

    merged = merge_nearby_vertical_blocks(blocks)
    logger.info("OCR block merge complete: before=%d after=%d", len(blocks), len(merged))
    return merged


def detect_candidate_blocks(
    image_path: Path,
    *,
    detector: str,
    lang: str,
    psm: int,
    min_confidence: float,
) -> list[TextBlock]:
    logger.info("Selected text detector: %s", detector)
    if detector == "ctd":
        from .ctd_detector import detect_ctd_text_blocks

        return detect_ctd_text_blocks(image_path)

    if detector == "tesseract":
        return run_tesseract_ocr(
            image_path,
            lang=lang,
            psm=psm,
            min_confidence=min_confidence,
        )

    if detector == "visual":
        image = Image.open(image_path).convert("RGB")
        visual_boxes = find_visual_text_candidates(image)
        logger.info("Visual text detector found %d candidate box(es)", len(visual_boxes))
        return [
            TextBlock(text="", box=box, confidence=0.0, detector="visual")
            for box in visual_boxes
        ]

    raise ValueError(f"Unknown text detector: {detector}")


def recognize_with_manga_ocr(
    image_path: Path,
    candidate_blocks: list[TextBlock],
    *,
    crop_padding: int = 10,
) -> list[TextBlock]:
    if not candidate_blocks:
        return []

    manga_ocr = load_manga_ocr()
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    refined: list[TextBlock] = []
    for block in candidate_blocks:
        x1, y1, x2, y2 = pad_box(block.box, width, height, crop_padding)
        crop = image.crop((x1, y1, x2, y2))
        if crop.width < 8 or crop.height < 8:
            logger.debug("Skipping tiny manga-ocr crop: box=%s size=%sx%s", block.box, crop.width, crop.height)
            continue

        crop = crop.resize((crop.width * 2, crop.height * 2))
        try:
            refined_text = normalize_ocr_text(manga_ocr(crop))
        except Exception:
            logger.exception("manga-ocr failed for crop box=%s; keeping detector text", block.box)
            refined_text = block.text

        if not refined_text:
            logger.debug("manga-ocr returned empty text for box=%s; keeping detector text=%s", block.box, shorten(block.text))
            refined_text = block.text
        if is_punctuation_only_ocr_text(refined_text):
            logger.debug("Skipping punctuation-only manga-ocr result: box=%s text=%s", block.box, shorten(refined_text))
            continue
        if not contains_japanese_ocr_text(refined_text):
            logger.debug("Skipping non-Japanese manga-ocr result: box=%s text=%s", block.box, shorten(refined_text))
            continue
        if len(normalize_for_filter(refined_text)) <= 1:
            logger.debug("Skipping single-character manga-ocr result: box=%s text=%s", block.box, shorten(refined_text))
            continue
        if is_ctd_horizontal_metadata(block, normalize_for_filter(refined_text)):
            logger.debug("Skipping horizontal metadata manga-ocr result: box=%s text=%s", block.box, shorten(refined_text))
            continue
        if is_bottom_left_short_horizontal_ctd_block(block, text=refined_text, width=width, height=height):
            logger.debug("Skipping bottom-left title-like manga-ocr result: box=%s text=%s", block.box, shorten(refined_text))
            continue
        if is_bottom_edge_short_horizontal_ctd_block(block, text=refined_text, width=width, height=height):
            logger.debug("Skipping bottom-edge short manga-ocr result: box=%s text=%s", block.box, shorten(refined_text))
            continue
        if is_ultra_tall_edge_ctd_block(block, width=width, height=height):
            logger.debug("Skipping ultra-tall edge manga-ocr result: box=%s text=%s", block.box, shorten(refined_text))
            continue

        logger.debug(
            "OCR refined: box=%s tesseract=%s manga_ocr=%s",
            block.box,
            shorten(block.text),
            shorten(refined_text),
        )
        refined.append(
            TextBlock(
                text=refined_text,
                box=block.box,
                confidence=block.confidence,
                detector=block.detector,
                metadata=block.metadata,
            )
        )

    deduped = deduplicate_similar_ocr_blocks(refined)
    deduped = remove_composite_overlap_ocr_blocks(deduped)
    logger.info("manga-ocr recognition complete: before=%d after=%d deduped=%d", len(candidate_blocks), len(refined), len(deduped))
    return deduped


def load_manga_ocr():
    global _MANGA_OCR

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")

    try:
        from manga_ocr import MangaOcr
    except ImportError as exc:
        logger.exception("manga-ocr import failed")
        raise RuntimeError(
            "manga-ocr is not installed. Re-run Launch-GUI.ps1 so dependencies are installed, "
            "or switch OCR engine to tesseract."
        ) from exc

    if _MANGA_OCR is None:
        logger.info("Loading manga-ocr model. First run can take a while while the model downloads.")
        _MANGA_OCR = instantiate_manga_ocr(
            MangaOcr,
            previous_transformers_offline=previous_transformers_offline,
            previous_hf_offline=previous_hf_offline,
        )
        logger.info("manga-ocr model loaded")
    return _MANGA_OCR


def instantiate_manga_ocr(
    manga_ocr_class,
    *,
    previous_transformers_offline: str | None,
    previous_hf_offline: str | None,
):
    cached_path = find_manga_ocr_snapshot()
    if cached_path:
        logger.info("Loading manga-ocr from cached snapshot: %s", cached_path)
        return manga_ocr_class(str(cached_path))

    logger.info("No complete manga-ocr cache found; allowing online model lookup")
    restore_env_var("TRANSFORMERS_OFFLINE", previous_transformers_offline)
    restore_env_var("HF_HUB_OFFLINE", previous_hf_offline)
    return manga_ocr_class()


def find_manga_ocr_snapshot() -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--kha-white--manga-ocr-base" / "snapshots"
    if not cache_root.exists():
        return None

    for snapshot in sorted(cache_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if not snapshot.is_dir():
            continue
        required = [
            snapshot / "config.json",
            snapshot / "preprocessor_config.json",
            snapshot / "tokenizer_config.json",
            snapshot / "vocab.txt",
        ]
        has_weights = (snapshot / "pytorch_model.bin").exists() or (snapshot / "model.safetensors").exists()
        if all(path.exists() for path in required) and has_weights:
            return snapshot
    return None


def restore_env_var(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def recognize_with_tesseract(
    image_path: Path,
    candidate_blocks: list[TextBlock],
    *,
    lang: str,
    psm: int,
) -> list[TextBlock]:
    import pytesseract

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    recognized: list[TextBlock] = []
    for block in candidate_blocks:
        if block.detector == "tesseract" and block.text:
            recognized.append(block)
            continue

        x1, y1, x2, y2 = pad_box(block.box, width, height, 10)
        crop = image.crop((x1, y1, x2, y2))
        config = f"--psm {psm} --oem 1"
        try:
            text = normalize_ocr_text(pytesseract.image_to_string(crop, lang=lang, config=config))
        except pytesseract.TesseractNotFoundError as exc:
            logger.exception("Tesseract executable was not found")
            raise RuntimeError(tesseract_missing_message()) from exc
        if is_punctuation_only_ocr_text(text):
            logger.debug("Skipping punctuation-only tesseract result: box=%s text=%s", block.box, shorten(text))
            continue
        if not contains_japanese_ocr_text(text):
            logger.debug("Skipping non-Japanese tesseract result: box=%s text=%s", block.box, shorten(text))
            continue

        recognized.append(
            TextBlock(
                text=text or block.text,
                box=block.box,
                confidence=block.confidence,
                detector=block.detector,
                metadata=block.metadata,
            )
        )

    logger.info("Tesseract recognition complete: before=%d after=%d", len(candidate_blocks), len(recognized))
    return recognized


def run_manga_ocr(
    image_path: Path,
    *,
    lang: str,
    psm: int,
    min_confidence: float,
    crop_padding: int = 10,
) -> list[TextBlock]:
    tesseract_blocks = run_tesseract_ocr(
        image_path,
        lang=lang,
        psm=psm,
        min_confidence=min_confidence,
    )
    image = Image.open(image_path).convert("RGB")
    visual_boxes = find_visual_text_candidates(image)
    candidate_blocks = add_visual_candidates(tesseract_blocks, visual_boxes)
    return recognize_with_manga_ocr(image_path, candidate_blocks, crop_padding=crop_padding)


def run_ocr(
    image_path: Path,
    *,
    detector: str,
    engine: str,
    lang: str,
    psm: int,
    min_confidence: float,
) -> list[TextBlock]:
    logger.info("Selected OCR engine: %s", engine)
    candidate_blocks = detect_candidate_blocks(
        image_path,
        detector=detector,
        lang=lang,
        psm=psm,
        min_confidence=min_confidence,
    )
    logger.info("Detector returned %d candidate block(s)", len(candidate_blocks))

    if engine == "manga-ocr":
        return recognize_with_manga_ocr(image_path, candidate_blocks)
    if engine == "tesseract":
        return recognize_with_tesseract(image_path, candidate_blocks, lang=lang, psm=psm)
    raise ValueError(f"Unknown OCR engine: {engine}")


def normalize_ocr_text(value: str) -> str:
    return "".join(str(value).split())


def parse_confidence(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def is_punctuation_only_ocr_text(value: str) -> bool:
    text = normalize_ocr_text(value)
    return bool(text) and not any(char.isalnum() for char in text)


def contains_japanese_ocr_text(value: str) -> bool:
    return any(is_japanese_ocr_char(char) for char in normalize_ocr_text(value))


def is_japanese_ocr_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
    )


def pad_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_width, x2 + padding),
        min(image_height, y2 + padding),
    )


def add_visual_candidates(
    blocks: list[TextBlock],
    visual_boxes: list[tuple[int, int, int, int]],
) -> list[TextBlock]:
    candidates = list(blocks)
    for box in visual_boxes:
        if any(box_iou(box, existing.box) > 0.55 for existing in candidates):
            logger.debug("Skipping duplicate visual candidate box=%s", box)
            continue
        candidates.append(TextBlock(text="", box=box, confidence=0.0, detector="visual"))
    return candidates


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / (a_area + b_area - intersection)


def deduplicate_similar_ocr_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    kept: list[TextBlock] = []
    for block in blocks:
        duplicate_index = None
        for index, existing in enumerate(kept):
            if normalize_ocr_text(block.text) == normalize_ocr_text(existing.text) and box_iou(block.box, existing.box) >= 0.82:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(block)
            continue

        existing = kept[duplicate_index]
        if box_area(block.box) > box_area(existing.box):
            logger.debug("Replacing duplicate OCR block with larger box: old=%s new=%s text=%s", existing.box, block.box, shorten(block.text))
            kept[duplicate_index] = block
        else:
            logger.debug("Skipping duplicate OCR block: kept=%s skipped=%s text=%s", existing.box, block.box, shorten(block.text))
    return kept


def remove_composite_overlap_ocr_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    kept: list[TextBlock] = []
    for block in blocks:
        if is_composite_overlap_ocr_block(block, blocks):
            logger.debug("Skipping composite-overlap OCR block: box=%s text=%s", block.box, shorten(block.text))
            continue
        kept.append(block)
    return kept


def is_composite_overlap_ocr_block(block: TextBlock, blocks: list[TextBlock]) -> bool:
    if getattr(block, "detector", "") != "ctd":
        return False
    if len(normalize_ocr_text(block.text)) < 12:
        return False

    block_vertical = bool(getattr(block, "metadata", {}).get("vertical"))
    block_area = box_area(block.box)
    smaller_overlaps = 0
    for other in blocks:
        if other is block:
            continue
        if getattr(other, "detector", "") != "ctd":
            continue
        if bool(getattr(other, "metadata", {}).get("vertical")) != block_vertical:
            continue
        if box_area(other.box) >= block_area * 0.75:
            continue
        if box_iou(block.box, other.box) >= 0.20:
            smaller_overlaps += 1
            if smaller_overlaps >= 2:
                return True
    return False


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def is_ultra_tall_edge_ctd_block(block: TextBlock, *, width: int, height: int) -> bool:
    if getattr(block, "detector", "") != "ctd":
        return False
    if not bool(getattr(block, "metadata", {}).get("vertical")):
        return False

    x1, y1, x2, y2 = block.box
    block_width = max(1, x2 - x1)
    block_height = max(1, y2 - y1)
    if block_height / max(1, height) < 0.72:
        return False
    if block_width / max(1, width) > 0.07:
        return False
    return x1 <= width * 0.08 or x2 >= width * 0.92


def is_bottom_left_short_horizontal_ctd_block(block: TextBlock, *, text: str, width: int, height: int) -> bool:
    if getattr(block, "detector", "") != "ctd":
        return False
    if bool(getattr(block, "metadata", {}).get("vertical")):
        return False
    if len(normalize_ocr_text(text)) > 4:
        return False

    x1, y1, x2, y2 = block.box
    block_width = max(1, x2 - x1)
    block_height = max(1, y2 - y1)
    return (
        x1 <= width * 0.04
        and y2 >= height * 0.96
        and block_width >= width * 0.12
        and block_height >= height * 0.12
    )


def is_bottom_edge_short_horizontal_ctd_block(block: TextBlock, *, text: str, width: int, height: int) -> bool:
    if getattr(block, "detector", "") != "ctd":
        return False
    if bool(getattr(block, "metadata", {}).get("vertical")):
        return False
    if len(normalize_ocr_text(text)) > 3:
        return False

    x1, y1, x2, y2 = block.box
    block_width = max(1, x2 - x1)
    block_height = max(1, y2 - y1)
    return (
        y1 >= height * 0.94
        and y2 >= height * 0.97
        and block_width <= width * 0.08
        and block_height <= height * 0.06
    )


def find_visual_text_candidates(image: Image.Image) -> list[tuple[int, int, int, int]]:
    import cv2

    gray = np.array(image.convert("L"))
    image_height, image_width = gray.shape
    dark_mask = (gray < 120).astype("uint8") * 255
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_mask, 8)

    glyph_mask = np.zeros_like(dark_mask)
    for index in range(1, component_count):
        x, y, width, height, area = stats[index]
        if not looks_like_glyph_component(width, height, area):
            continue
        glyph_mask[labels == index] = 255

    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (18, 22)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (10, 34)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (34, 10)),
    ]
    boxes: list[tuple[int, int, int, int]] = []
    for kernel in kernels:
        dilated = cv2.dilate(glyph_mask, kernel, iterations=1)
        contours, _hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            box = clamp_box((x, y, x + width, y + height), image_width, image_height)
            if looks_like_text_cluster(gray, box):
                boxes.append(box)

    merged = merge_overlapping_boxes(boxes)
    logger.debug("Visual text candidate boxes: %s", merged)
    return merged


def looks_like_glyph_component(width: int, height: int, area: int) -> bool:
    if width < 2 or height < 3:
        return False
    if width > 55 or height > 70:
        return False
    if area < 3 or area > 700:
        return False
    return area / max(1, width * height) > 0.04


def looks_like_text_cluster(gray: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    if width < 10 or height < 10:
        return False
    if width > 220 or height > 220:
        return False
    if area < 90 or area > 18000:
        return False

    crop = gray[y1:y2, x1:x2]
    white_ratio = float((crop > 180).mean())
    dark_ratio = float((crop < 120).mean())
    if white_ratio < 0.45:
        return False
    if dark_ratio < 0.015 or dark_ratio > 0.45:
        return False
    return True


def clamp_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1),
        max(0, y1),
        min(image_width, x2),
        min(image_height, y2),
    )


def merge_overlapping_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    ordered = sorted(set(boxes), key=lambda box: (box[1], box[0], box[3], box[2]))
    merged: list[tuple[int, int, int, int]] = []
    for box in ordered:
        merged_into_existing = False
        for index, existing in enumerate(merged):
            if box_iou(box, existing) > 0.25 or box_contains(existing, box) or box_contains(box, existing):
                merged[index] = (
                    min(existing[0], box[0]),
                    min(existing[1], box[1]),
                    max(existing[2], box[2]),
                    max(existing[3], box[3]),
                )
                merged_into_existing = True
                break
        if not merged_into_existing:
            merged.append(box)
    return merged


def box_contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2


def shrink_box_to_dark_pixels(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    margin: int = 6,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return box

    crop = image.crop((x1, y1, x2, y2)).convert("L")
    pixels = np.array(crop)
    dark_y, dark_x = np.where(pixels < 110)
    if len(dark_x) == 0 or len(dark_y) == 0:
        return box

    width = x2 - x1
    height = y2 - y1
    nx1 = max(0, int(dark_x.min()) - margin)
    ny1 = max(0, int(dark_y.min()) - margin)
    nx2 = min(width, int(dark_x.max()) + margin)
    ny2 = min(height, int(dark_y.max()) + margin)

    shrunk = (x1 + nx1, y1 + ny1, x1 + nx2, y1 + ny2)
    original_area = max(1, width * height)
    shrunk_area = max(1, (shrunk[2] - shrunk[0]) * (shrunk[3] - shrunk[1]))
    if shrunk_area / original_area < 0.08:
        logger.debug("Box shrink looked too aggressive; keeping original box=%s shrunk=%s", box, shrunk)
        return box

    return shrunk


def merge_nearby_vertical_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    if len(blocks) < 2:
        logger.debug("Skipping OCR merge because there are fewer than two blocks")
        return blocks

    ordered = sorted(blocks, key=lambda block: (block.box[1], block.box[0]))
    merged: list[TextBlock] = []

    for block in ordered:
        if not merged:
            merged.append(block)
            continue

        prev = merged[-1]
        if should_merge(prev.box, block.box):
            logger.debug("Merging nearby OCR blocks: prev=%s next=%s", prev.box, block.box)
            merged[-1] = merge_blocks(prev, block)
        else:
            merged.append(block)

    return merged


def should_merge(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    a_height = ay2 - ay1
    b_height = by2 - by1
    vertical_gap = max(0, by1 - ay2)
    horizontal_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    min_width = max(1, min(ax2 - ax1, bx2 - bx1))

    return (
        vertical_gap <= max(12, int(min(a_height, b_height) * 0.35))
        and horizontal_overlap / min_width >= 0.35
    )


def merge_blocks(a: TextBlock, b: TextBlock) -> TextBlock:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    return TextBlock(
        text=f"{a.text}{b.text}",
        box=(min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)),
        confidence=(a.confidence + b.confidence) / 2,
        detector=a.detector,
        metadata=a.metadata,
    )
