from __future__ import annotations

import re
import unicodedata


def normalize_source_text(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def normalize_candidate_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def compact_for_copy_check(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(text))).lower()
