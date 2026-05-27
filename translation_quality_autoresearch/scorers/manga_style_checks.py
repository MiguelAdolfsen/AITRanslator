from __future__ import annotations

from translation_quality_autoresearch.common.text_normalize import words


def style_metrics(case, candidate_text: str, *, max_target_words_default: int = 28) -> dict[str, object]:
    source_len = max(1, len(str(getattr(case, "source_text", ""))))
    target_words = words(candidate_text)
    max_words = int((getattr(case, "metadata", {}) or {}).get("max_target_words", max_target_words_default))
    length_ratio = len(target_words) / max(1.0, source_len / 3.0)
    punctuation_ok = True
    if "!" in getattr(case, "must_preserve", []) and "!" not in str(candidate_text):
        punctuation_ok = False
    naturalness = 1.0
    if len(target_words) > max_words:
        naturalness -= 0.3
    if any(marker in str(candidate_text).lower() for marker in ("the speaker", "this phrase", "it means")):
        naturalness -= 0.4
    return {
        "length_ratio": round(length_ratio, 4),
        "punctuation_ok": punctuation_ok,
        "dialogue_naturalness_proxy": max(0.0, round(naturalness, 4)),
        "word_count": len(target_words),
        "max_target_words": max_words,
    }
