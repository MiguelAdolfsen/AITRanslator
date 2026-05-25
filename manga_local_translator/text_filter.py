from __future__ import annotations

import logging
import re
import unicodedata

from .detect_types import TextBlock
from .line_identity import block_id_for_block, line_id_for_block, source_hash
from .logging_utils import shorten

logger = logging.getLogger(__name__)


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
        reason = skip_reason(block, image_area=image_area)
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


def skip_reason(block: TextBlock, *, image_area: int) -> str | None:
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
