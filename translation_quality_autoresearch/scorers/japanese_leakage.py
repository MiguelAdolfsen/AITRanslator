from __future__ import annotations


def is_japanese_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
    )


def count_japanese_chars(text: str) -> int:
    return sum(1 for char in str(text) if is_japanese_char(char))


def japanese_leakage_ratio(text: str) -> float:
    stripped = str(text).strip()
    if not stripped:
        return 0.0
    return count_japanese_chars(stripped) / max(1, len(stripped))
