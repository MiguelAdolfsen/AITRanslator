from __future__ import annotations

import logging
import re
import unicodedata

from .detect_types import TextBlock
from .line_identity import block_id_for_block, line_id_for_block, source_hash
from .logging_utils import shorten

logger = logging.getLogger(__name__)

_CHAPTER_OR_DATE_PATTERN = re.compile(
    r"(?:\u7b2c[0-9]+\u8a71|[0-9]{4}\u5e74[0-9]{1,2}\u6708)"
)
_PROMO_METADATA_PATTERN = re.compile(
    r"(?:TV\u30a2\u30cb\u30e1|WEB|\u767a\u58f2\u4e2d|\u7d76\u8cdb\u767a\u58f2|\u2606|\u25bd|\u540c\u4eba)"
)
_CREDIT_METADATA_TERMS = (
    "\u6f2b\u753b",
    "\u539f\u4f5c",
    "\u30ad\u30e3\u30e9\u30af\u30bf\u30fc",
    "\u8868\u7d19",
    "\u5358\u884c\u672c",
)


def filter_text_blocks(
    blocks: list[TextBlock],
    *,
    width: int,
    height: int,
) -> tuple[list[TextBlock], list[dict[str, object]]]:
    kept: list[TextBlock] = []
    skipped: list[dict[str, object]] = []
    image_area = max(1, width * height)

    for block in blocks:
        reason = skip_reason(block, image_area=image_area, image_width=width, image_height=height)
        if reason:
            logger.info(
                "Skipping OCR block: reason=%s box=%s confidence=%.2f text=%s",
                reason,
                block.box,
                block.confidence,
                shorten(block.text),
            )
            skipped.append(
                block_to_debug_dict(block, reason=reason)
            )
            continue
        kept.append(block)

    return kept, skipped


def skip_reason(
    block: TextBlock,
    *,
    image_area: int,
    image_width: int | None = None,
    image_height: int | None = None,
) -> str | None:
    text = normalize_for_filter(block.text)
    if not text:
        return "empty"

    japanese_count = count_japanese_chars(text)
    kana_count = count_kana_chars(text)
    kanji_count = count_kanji_chars(text)
    alnum_count = sum(char.isalnum() for char in text)
    symbol_count = sum(not char.isspace() for char in text)

    if japanese_count == 0 and alnum_count == 0:
        return "punctuation_or_symbols_only"
    if japanese_count == 0:
        return "non_japanese_text"
    if symbol_count <= 1:
        return "single_character_fragment"
    if is_ctd_horizontal_metadata(block, text):
        return "horizontal_metadata_or_promo"
    if image_width is not None and image_height is not None:
        if is_ctd_horizontal_metadata_geometry(
            block,
            text,
            image_width=image_width,
            image_height=image_height,
            image_area=image_area,
        ):
            return "horizontal_metadata_geometry"
        if is_ctd_vertical_edge_metadata_geometry(
            block,
            text,
            image_width=image_width,
            image_height=image_height,
        ):
            return "vertical_edge_metadata_geometry"
    if kanji_count == 0 and kana_count <= 2 and japanese_count == kana_count:
        if getattr(block, "detector", "") == "ctd" and (has_dialogue_punctuation(text) or is_known_short_kana_reaction(text)):
            return None
        return "short_kana_fragment"

    x1, y1, x2, y2 = block.box
    area = max(1, (x2 - x1) * (y2 - y1))
    if area / image_area > 0.08 and japanese_count < 4:
        return "oversized_low_text_box"

    return None


def normalize_for_filter(text: str) -> str:
    return unicodedata.normalize("NFKC", "".join(str(text).split()))


def has_dialogue_punctuation(text: str) -> bool:
    return any(char in text for char in ("\u2026", ".", "\u3002", "?", "\uff1f", "!", "\uff01"))


def is_known_short_kana_reaction(text: str) -> bool:
    return normalize_for_filter(text) in {
        "\u3046\u3080",
        "\u3046\u3043",
        "\u304f\u3063",
        "\u304d\u3083",
        "\u3072\u3043",
        "\u3061\u3063",
        "\u3093",
    }


def is_ctd_horizontal_metadata(block: TextBlock, text: str) -> bool:
    if getattr(block, "detector", "") != "ctd":
        return False
    if bool(getattr(block, "metadata", {}).get("vertical")):
        return False
    if _CHAPTER_OR_DATE_PATTERN.search(text) or _PROMO_METADATA_PATTERN.search(text):
        return True
    return sum(term in text for term in _CREDIT_METADATA_TERMS) >= 2


def is_ctd_horizontal_metadata_geometry(
    block: TextBlock,
    text: str,
    *,
    image_width: int,
    image_height: int,
    image_area: int,
) -> bool:
    if getattr(block, "detector", "") != "ctd":
        return False
    metadata = getattr(block, "metadata", {}) or {}
    if bool(metadata.get("vertical")):
        return False

    x1, y1, x2, y2 = block.box
    block_width = max(1, x2 - x1)
    block_height = max(1, y2 - y1)
    area_ratio = (block_width * block_height) / max(1, image_area)
    text_len = len(text)
    has_punctuation = has_dialogue_punctuation(text)

    if metadata.get("language") == "unknown":
        if block_height <= max(14, int(image_height * 0.015)) and text_len <= 8 and not has_punctuation:
            return True
        if (
            area_ratio >= 0.018
            and text_len <= 6
            and (
                y1 >= image_height * 0.70
                or x1 <= image_width * 0.08
                or x2 >= image_width * 0.92
            )
            and not has_punctuation
        ):
            return True

    return (
        y1 >= image_height * 0.92
        and block_height <= image_height * 0.06
        and text_len <= 4
        and not has_punctuation
    )


def is_ctd_vertical_edge_metadata_geometry(
    block: TextBlock,
    text: str,
    *,
    image_width: int,
    image_height: int,
) -> bool:
    if getattr(block, "detector", "") != "ctd":
        return False
    metadata = getattr(block, "metadata", {}) or {}
    if metadata.get("vertical") is not True:
        return False

    x1, y1, x2, y2 = block.box
    block_width = max(1, x2 - x1)
    block_height = max(1, y2 - y1)
    touches_outer_edge = x1 <= image_width * 0.015 or x2 >= image_width * 0.985
    has_decorative_marker = any(marker in text for marker in ("\u266a", "\u2606", "\u2605"))
    return (
        touches_outer_edge
        and block_width <= max(30, image_width * 0.04)
        and block_height >= image_height * 0.12
        and has_decorative_marker
    )


def count_japanese_chars(text: str) -> int:
    return sum(is_kana(char) or is_kanji(char) for char in text)


def count_kana_chars(text: str) -> int:
    return sum(is_kana(char) for char in text)


def count_kanji_chars(text: str) -> int:
    return sum(is_kanji(char) for char in text)


def is_kana(char: str) -> bool:
    code = ord(char)
    return 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF


def is_kanji(char: str) -> bool:
    code = ord(char)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF


def unusable_translation_reason(source_text: str, translated_text: str, *, translator_name: str) -> str | None:
    text = translated_text.strip()
    if translator_name == "none":
        return None if text else "empty_translation"
    if not text:
        return "empty_translation"
    if any(marker in text for marker in ("\u00e3", "\u00ef", "\ufffd", "\u00e2\u2013")):
        return "mojibake_translation"
    if not any(char.isascii() and char.isalnum() for char in text):
        return "non_english_translation"
    if text == source_text:
        return "unchanged_translation"
    return None


def fallback_translation(source_text: str, *, reason: str) -> str:
    if reason in {"empty_translation", "non_english_translation"}:
        return "..."
    return source_text


def suspected_bad_translation(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped == "...":
        return True
    if any(marker in stripped for marker in ("\u00e3", "\u00ef", "\ufffd", "<think>", "{", "}")):
        return True
    if re.search(
        r"\b(please provide|could you please provide|i(?:'| a)m ready to translate|i do not see any japanese|i don't see any japanese|i don't understand what you|i can't provide that translation|return english translation only|return only the translation|translate the following japanese text|japanese translator who specializes|translation services|quality assurance|community management)\b",
        stripped,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"\b[A-Za-z]+(?:-[A-Za-z]+){4,}\b", stripped):
        return True
    return False


def is_box_like(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        [int(item) for item in value]
    except (TypeError, ValueError):
        return False
    return True


def block_to_debug_dict(
    block,
    *,
    status: str | None = None,
    reason: str | None = None,
    translated_text: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "detector": getattr(block, "detector", "unknown"),
        "box": block.box,
        "confidence": block.confidence,
        "block_id": block_id_for_block(block),
        "line_id": line_id_for_block(block),
        "source_hash": source_hash(block.text),
        "source_text": block.text,
    }
    if status is not None:
        payload["status"] = status
    if getattr(block, "metadata", None):
        payload["metadata"] = block.metadata
    if translated_text is not None:
        payload["translated_text"] = translated_text
    if reason is not None:
        payload["reason"] = reason
    return payload
