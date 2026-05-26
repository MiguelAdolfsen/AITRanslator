from __future__ import annotations

import re
import unicodedata


JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff]")
PUNCT_ONLY_RE = re.compile(r"^[\s\W_、。！？!?…・ー~〜]+$", re.UNICODE)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def contains_japanese(value: object) -> bool:
    return bool(JAPANESE_RE.search(str(value or "")))


def japanese_ratio(value: object) -> float:
    text = normalize_text(value)
    if not text:
        return 0.0
    japanese = sum(1 for char in text if contains_japanese(char))
    return japanese / max(1, len(text))


def is_empty_or_punctuation(value: object) -> bool:
    text = normalize_text(value)
    return not text or bool(PUNCT_ONLY_RE.fullmatch(text))


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def character_error_rate(reference: object, hypothesis: object, *, cap: float = 1.0) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return cap
    return min(cap, levenshtein(ref, hyp) / max(1, len(ref)))


def text_similarity(a: object, b: object) -> float:
    cer = character_error_rate(a, b, cap=1.0)
    return max(0.0, 1.0 - cer)


def normalized_region_text(region: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = region.get(key)
        if value is not None:
            text = normalize_text(value)
            if text:
                return text
    return ""

