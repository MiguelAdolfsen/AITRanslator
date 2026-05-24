from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from .text_filter import count_japanese_chars, suspected_bad_translation, unusable_translation_reason


FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
KANJI_NUMERALS = "一二三四五六七八九十百千万零〇"
HONORIFICS = ("さん", "様", "ちゃん", "くん", "先輩", "先生", "殿", "氏")
SOURCE_CREDIT_MARKERS = ("第", "話", "漫画", "原作", "キャラクター", "発売", "表紙", "次回", "号")
ENGLISH_NUMBER_WORDS = {
    "0": ("zero",),
    "1": ("one", "single"),
    "2": ("two", "both"),
    "3": ("three",),
    "4": ("four",),
    "5": ("five",),
    "6": ("six",),
    "7": ("seven",),
    "8": ("eight",),
    "9": ("nine",),
    "10": ("ten",),
    "11": ("eleven",),
    "12": ("twelve", "midnight", "noon"),
}
@dataclass
class ConsistencyEntry:
    source_term: str
    translations: dict[str, int] = field(default_factory=dict)
    count: int = 0

    def observe(self, translated_text: str) -> None:
        self.count += 1
        for candidate in english_term_candidates(translated_text):
            self.translations[candidate] = self.translations.get(candidate, 0) + 1

    def best_translation(self) -> str:
        if not self.translations:
            return ""
        return sorted(self.translations.items(), key=lambda item: (-item[1], item[0].lower()))[0][0]

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "source_term": self.source_term,
            "count": self.count,
            "translations": dict(sorted(self.translations.items(), key=lambda item: (-item[1], item[0]))),
            "best_translation": self.best_translation(),
        }


@dataclass
class ConsistencyMemory:
    entries: dict[str, ConsistencyEntry] = field(default_factory=dict)

    def observe(self, source_text: str, translated_text: str) -> None:
        if not translated_text.strip() or suspected_bad_translation(translated_text):
            return
        if unusable_translation_reason(source_text, translated_text, translator_name="qwen") is not None:
            return
        features = extract_source_features(source_text)
        if "ocr_noise" in features["risk_flags"]:
            return
        for term in consistency_source_terms(features):
            self.entries.setdefault(term, ConsistencyEntry(term)).observe(translated_text)

    def entries_for_source(self, source_text: str, translated_text: str = "") -> list[dict[str, object]]:
        features = extract_source_features(source_text)
        matches: list[dict[str, object]] = []
        normalized_translation = translated_text.lower()
        for term in consistency_source_terms(features):
            entry = self.entries.get(term)
            if entry is None:
                continue
            payload = entry.to_debug_dict()
            best = str(payload.get("best_translation") or "")
            payload["hit"] = bool(best and best.lower() in normalized_translation)
            matches.append(payload)
        return matches

    def to_debug_list(self) -> list[dict[str, object]]:
        return [entry.to_debug_dict() for entry in sorted(self.entries.values(), key=lambda item: item.source_term)]


def build_consistency_memory(items: Iterable[tuple[str, str]]) -> ConsistencyMemory:
    memory = ConsistencyMemory()
    for source_text, translated_text in items:
        memory.observe(source_text, translated_text)
    return memory


def extract_source_features(source_text: str) -> dict[str, object]:
    text = normalize_source(source_text)
    numbers = extract_numbers(text)
    bracket_terms = extract_bracket_terms(text)
    katakana_terms = extract_katakana_terms(text)
    honorific_names = extract_honorific_names(text)
    punctuation_intent = extract_punctuation_intent(text)
    risk_flags: list[str] = []
    if has_repeated_ocr_text(text):
        risk_flags.append("repeated_ocr")
    if looks_like_credit_or_promo_text(text):
        risk_flags.append("ocr_noise")
    if looks_like_mixed_credit_noise(text):
        risk_flags.append("mixed_credit_noise")
    if len(text) >= 50 and count_japanese_chars(text) < len(text) * 0.55:
        risk_flags.append("low_japanese_density")
    return {
        "normalized_source": text,
        "numbers": numbers,
        "bracket_terms": bracket_terms,
        "katakana_terms": katakana_terms,
        "honorific_names": honorific_names,
        "punctuation_intent": punctuation_intent,
        "source_length": len(text),
        "risk_flags": list(dict.fromkeys(risk_flags)),
    }


def analyze_translation_evidence(
    source_text: str,
    translated_text: str,
    *,
    memory_entries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    features = extract_source_features(source_text)
    failures: list[str] = []
    risk_flags = list(features["risk_flags"])
    if dropped_numbers(features, translated_text):
        failures.append("dropped_number")
    if dropped_bracket_terms(features, translated_text):
        failures.append("dropped_bracket_term")
    if invented_english_name_on_noisy_source(features, translated_text):
        failures.append("invented_english_name_on_noisy_source")
    if broken_english_pattern(translated_text):
        failures.append("broken_english")
    if count_japanese_chars(translated_text) > 0:
        failures.append("untranslated_japanese")
    memory_conflicts = consistency_conflicts(memory_entries or [], translated_text)
    if memory_conflicts:
        failures.append("consistency_conflict")
    if failures:
        risk_flags.append("translation_evidence_failure")
    return {
        "preservation_failures": list(dict.fromkeys(failures)),
        "risk_flags": list(dict.fromkeys(risk_flags)),
        "memory_conflicts": memory_conflicts,
        "broken_english": broken_english_pattern(translated_text),
        "invented_english_names": invented_english_names(translated_text),
    }


def build_evidence_context(source_text: str, translated_text: str, memory: ConsistencyMemory | None = None) -> dict[str, object]:
    memory_entries = memory.entries_for_source(source_text, translated_text) if memory is not None else []
    source_features = extract_source_features(source_text)
    translation_evidence = analyze_translation_evidence(source_text, translated_text, memory_entries=memory_entries)
    return {
        "source_features": source_features,
        "consistency_memory_entries": memory_entries,
        "translation_evidence": translation_evidence,
        "evidence_risk_flags": list(dict.fromkeys(source_features["risk_flags"] + translation_evidence["risk_flags"])),
        "evidence_repair_reasons": translation_evidence["preservation_failures"],
    }


def normalize_source(text: str) -> str:
    return unicodedata.normalize("NFKC", "".join(str(text).split()))


def extract_numbers(text: str) -> list[str]:
    normalized = text.translate(FULLWIDTH_DIGITS)
    values = re.findall(r"\d+", normalized)
    values.extend(re.findall(rf"[{KANJI_NUMERALS}]{{1,6}}(?=[番巻話号人年月日])", normalized))
    return list(dict.fromkeys(values))


def extract_bracket_terms(text: str) -> list[str]:
    terms = re.findall(r"[〈《「『〝](.*?)[〉》」』〟]", text)
    return [term for term in dict.fromkeys(term.strip() for term in terms) if term]


def extract_katakana_terms(text: str) -> list[str]:
    terms = re.findall(r"[\u30a1-\u30fa\u30fc]{2,}", text)
    return [term for term in dict.fromkeys(terms) if not is_probable_sfx_term(term)]


def extract_honorific_names(text: str) -> list[str]:
    suffixes = "|".join(re.escape(value) for value in HONORIFICS)
    pattern = rf"[\u3040-\u30ff\u4e00-\u9fff\u3400-\u4dbf]{{1,12}}(?:{suffixes})"
    return list(dict.fromkeys(re.findall(pattern, text)))


def extract_punctuation_intent(text: str) -> list[str]:
    intent: list[str] = []
    if any(char in text for char in ("?", "？")):
        intent.append("question")
    if any(char in text for char in ("!", "！")):
        intent.append("exclamation")
    if "..." in text or "…" in text or "．．" in text:
        intent.append("ellipsis")
    if text.endswith(("ぞ", "こい", "ろ", "なさい")):
        intent.append("imperative")
    return intent


def has_repeated_ocr_text(text: str) -> bool:
    if len(text) < 8:
        return False
    if len(text) % 2 == 0 and text[: len(text) // 2] == text[len(text) // 2 :]:
        return True
    for size in range(4, min(18, len(text) // 2 + 1)):
        chunk = text[:size]
        if text.startswith(chunk * 2):
            return True
    return False


def looks_like_credit_or_promo_text(text: str) -> bool:
    marker_count = sum(1 for marker in SOURCE_CREDIT_MARKERS if marker in text)
    return marker_count >= 3 or ("原作" in text and "漫画" in text)


def looks_like_mixed_credit_noise(text: str) -> bool:
    separators = sum(text.count(char) for char in "・『』〝〟<>＞＜")
    return len(text) >= 30 and separators >= 4


def is_probable_sfx_term(term: str) -> bool:
    compact = term.replace("ー", "")
    return len(term) <= 4 and len(set(compact)) <= 1


def consistency_source_terms(features: dict[str, object]) -> list[str]:
    terms: list[str] = []
    for key in ("honorific_names", "bracket_terms", "katakana_terms"):
        values = features.get(key, [])
        if isinstance(values, list):
            terms.extend(str(value) for value in values if str(value).strip())
    return list(dict.fromkeys(terms))


def english_term_candidates(translated_text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z]{2,}(?:[- ][A-Za-z]{2,})?\b", translated_text)
    honorific_candidates = re.findall(
        r"\b[A-Z][A-Za-z]{1,}(?:[- ][A-Z]?[A-Za-z]{1,})?\s+(?:san|sama|chan|kun|senpai|sensei|dono|shi)\b",
        translated_text,
        flags=re.IGNORECASE,
    )
    candidates.extend(honorific_candidates)
    blocked = {"The", "This", "That", "There", "Then", "When", "After", "Before", "And", "But", "Yes", "No"}
    return [candidate.strip() for candidate in dict.fromkeys(candidates) if candidate.split()[0] not in blocked]


def dropped_numbers(features: dict[str, object], translated_text: str) -> bool:
    numbers = [str(value) for value in features.get("numbers", [])]
    if not numbers:
        return False
    normalized = translated_text.translate(FULLWIDTH_DIGITS)
    lower = normalized.lower()
    for number in numbers:
        if not number.isdigit():
            continue
        if number in normalized:
            continue
        words = ENGLISH_NUMBER_WORDS.get(number, ())
        if words and any(re.search(rf"\b{re.escape(word)}\b", lower) for word in words):
            continue
        return True
    return False


def dropped_bracket_terms(features: dict[str, object], translated_text: str) -> bool:
    terms = [str(value) for value in features.get("bracket_terms", []) if str(value).strip()]
    if not terms:
        return False
    lower = translated_text.lower()
    if any(char in translated_text for char in ('"', "'", "“", "”")):
        return False
    if any(term.lower() in lower for term in terms if term.isascii()):
        return False
    return True


def invented_english_name_on_noisy_source(features: dict[str, object], translated_text: str) -> bool:
    risk_flags = set(str(value) for value in features.get("risk_flags", []))
    return bool(risk_flags & {"ocr_noise", "mixed_credit_noise"}) and bool(invented_english_names(translated_text))


def invented_english_names(translated_text: str) -> list[str]:
    candidates = english_term_candidates(translated_text)
    return [candidate for candidate in candidates if len(candidate.split()) >= 2]


def broken_english_pattern(translated_text: str) -> bool:
    lower = " ".join(translated_text.lower().split())
    patterns = (
        r"\bbeing the feeling of\b",
        r"\bone by three\b",
        r"\b[a-z]+ and [a-z]+ are hated\b",
        r"\bdo you even hold on to a hunchback\b",
        r"\bthere was a gas\b",
        r"\bcatch a bath\b",
        r"\b[a-z]+ dono\b",
    )
    return any(re.search(pattern, lower) for pattern in patterns)


def consistency_conflicts(memory_entries: list[dict[str, object]], translated_text: str) -> list[dict[str, object]]:
    lower = translated_text.lower()
    conflicts: list[dict[str, object]] = []
    for entry in memory_entries:
        best = str(entry.get("best_translation") or "").strip()
        if not best or int(entry.get("count") or 0) < 2:
            continue
        if best.lower() not in lower:
            conflicts.append(
                {
                    "source_term": entry.get("source_term", ""),
                    "expected": best,
                    "actual": translated_text,
                }
            )
    return conflicts
