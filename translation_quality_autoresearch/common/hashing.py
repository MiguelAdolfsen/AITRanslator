from __future__ import annotations

import hashlib

from .text_normalize import normalize_source_text


def sha256_text(text: str) -> str:
    normalized = normalize_source_text(text)
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_matches(text: str, expected: str) -> bool:
    return sha256_text(text) == str(expected)
