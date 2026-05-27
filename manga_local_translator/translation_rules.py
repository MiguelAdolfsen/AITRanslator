from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .logging_utils import shorten
from .translation_phrasebook import PHRASEBOOK

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SourceReplacement:
    source: str
    target: str


@dataclass(frozen=True)
class TargetReplacement:
    source: str
    target: str
    when_source_contains: str | None = None


@dataclass(frozen=True)
class TranslationGlossary:
    source_replacements: tuple[SourceReplacement, ...]
    exact_phrases: dict[str, str]
    target_replacements: tuple[TargetReplacement, ...]
    source_path: Path | None = None


HONORIFIC_SUFFIXES = (
    ("\u3055\u3093", "san"),
    ("\u30b5\u30f3", "san"),
    ("\u3055\u307e", "sama"),
    ("\u30b5\u30de", "sama"),
    ("\u69d8", "sama"),
    ("\u3061\u3083\u3093", "chan"),
    ("\u30c1\u30e3\u30f3", "chan"),
    ("\u304f\u3093", "kun"),
    ("\u30af\u30f3", "kun"),
    ("\u541b", "kun"),
    ("\u5148\u8f29", "senpai"),
    ("\u305b\u3093\u3071\u3044", "senpai"),
    ("\u30bb\u30f3\u30d1\u30a4", "senpai"),
    ("\u5148\u751f", "sensei"),
    ("\u305b\u3093\u305b\u3044", "sensei"),
    ("\u30bb\u30f3\u30bb\u30a4", "sensei"),
    ("\u6bbf", "dono"),
    ("\u3069\u306e", "dono"),
    ("\u30c9\u30ce", "dono"),
    ("\u6c0f", "shi"),
)
HONORIFIC_FAMILY_BASES = {
    "\u7236",
    "\u6bcd",
    "\u5144",
    "\u59c9",
    "\u5f1f",
    "\u59b9",
    "\u304a\u7236",
    "\u304a\u6bcd",
    "\u304a\u5144",
    "\u304a\u59c9",
    "\u7686",
    "\u7686\u69d8",
    "\u5c4b",
    "\u83d3\u5b50\u5c4b",
    "\u304a\u83d3\u5b50\u5c4b",
    "\u304a\u3058",
    "\u304a\u3070",
    "\u3058\u3044",
    "\u3070\u3042",
    "\u3068\u3046",
    "\u304b\u3042",
    "\u306b\u3044",
    "\u306d\u3048",
}


def normalize_japanese_for_translation(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", "".join(str(text).split()))
    normalized = normalized.replace("\u30fb\u30fb\u30fb", "\u2026")
    normalized = normalized.replace("...", "\u2026").replace("\uff0e\uff0e\uff0e", "\u2026")
    normalized = normalized.replace("\uff01\uff01", "!").replace("\uff1f\uff1f", "?")
    return normalized


def split_japanese_translation_units(text: str) -> list[str]:
    units = re.findall(r"[^!?\u3002]+[!?\u3002]*", text)
    cleaned = [clean_translation_unit(unit) for unit in units]
    return [unit for unit in cleaned if unit]


def clean_translation_unit(text: str) -> str:
    return text.strip().strip("\u300c\u300d\u300e\u300f").rstrip(":;\uff1a\uff1b")


def should_use_context_translation(
    text: str,
    *,
    before: str | None,
    after: str | None,
) -> bool:
    if not before and not after:
        return False
    if len(text) < 10:
        return False
    if translate_known_phrase(text) is not None:
        return False
    return True


def choose_contextual_translation(
    *,
    focused: str,
    contextual: str,
    source_text: str,
    before_count: int,
    source_unit_count: int,
) -> str:
    if not contextual.strip():
        return focused
    extracted = extract_context_focus(
        contextual,
        before_count=before_count,
        source_unit_count=source_unit_count,
    )
    if extracted:
        contextual = extracted
    focused_words = focused.split()
    contextual_words = contextual.split()
    if len(contextual_words) <= max(5, len(focused_words) * 2):
        return contextual
    logger.debug(
        "Ignoring overlong contextual translation: source=%s focused=%s contextual=%s",
        shorten(source_text),
        shorten(focused),
        shorten(contextual),
    )
    return focused


def extract_context_focus(
    text: str,
    *,
    before_count: int,
    source_unit_count: int,
) -> str | None:
    sentences = split_english_sentences(text)
    if not sentences:
        return None
    start = before_count
    end = start + source_unit_count
    if end > len(sentences):
        return None
    return " ".join(sentences[start:end]).strip()


def split_english_sentences(text: str) -> list[str]:
    parts = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text)
    return [part.strip() for part in parts if part.strip()]


def normalize_phrase_key(text: str) -> str:
    return re.sub(r"[\u2026\u3002\uff0e.\u3001,!?\uff01\uff1f\u2013\u2014\u2015\u2500\u300c\u300d\u300e\u300f\s]+$", "", text)


def translate_known_phrase(text: str, glossary: TranslationGlossary | None = None) -> str | None:
    compact = normalize_phrase_key(text)
    if glossary is not None and compact in glossary.exact_phrases:
        return glossary.exact_phrases[compact]

    prepared = compact
    if glossary is not None:
        prepared, _replacements = prepare_source_for_translation(compact, glossary)
        if prepared in glossary.exact_phrases:
            return glossary.exact_phrases[prepared]

    if compact in PHRASEBOOK:
        return PHRASEBOOK[compact]
    if prepared in PHRASEBOOK:
        return PHRASEBOOK[prepared]
    if is_sasuga_senpai_phrase(prepared):
        return "A-as expected, senpai...!"
    if is_crunching_sfx_phrase(prepared):
        return "Crunch crunch."
    if is_watch_out_trap_senpai_phrase(prepared):
        return "Watch out, the enemy may have set a trap... As expected, senpai!"
    zannen_fragment = translate_short_zannen_fragment(prepared)
    if zannen_fragment is not None:
        return zannen_fragment
    if compact.startswith("\u3042\u3042") and len(compact) <= 4:
        return "Ah..."
    return None


def translate_short_zannen_fragment(text: str) -> str | None:
    compact = normalize_phrase_key(text)
    if not compact.startswith("\u6b8b\u5ff5"):
        return None
    tail = compact[len("\u6b8b\u5ff5") :]
    if tail in {"", "\u306a"}:
        return "What a shame."
    if len(compact) <= 8 and (
        tail in {"\u3060\u3051\u3069", "\u3060\u304c", "\u3067\u3059\u304c", "\u3051\u3069", "\u3067\u3082"}
        or tail.endswith(("\u3060\u3051\u3069", "\u3060\u304c", "\u3067\u3059\u304c", "\u3051\u3069", "\u3067\u3082"))
    ):
        return "Too bad, but..."
    return None


def is_watch_out_trap_senpai_phrase(text: str) -> bool:
    compact = normalize_phrase_key(text)
    required = ("\u6c17\u3092\u3064\u3051\u308d", "\u6575", "\u7f60", "\u5148\u8f29")
    return all(part in compact for part in required)


def is_sasuga_senpai_phrase(text: str) -> bool:
    compact = normalize_phrase_key(text)
    return "\u3055\u3059\u304c" in compact and ("\u5148\u8f29" in compact or "senpai" in compact.lower())


def is_crunching_sfx_phrase(text: str) -> bool:
    compact = normalize_phrase_key(text)
    normalized = compact.replace("\u30dd\u308a", "\u30dd\u30ea").replace("\u30dc\u308a", "\u30dc\u30ea")
    return bool(re.fullmatch(r"[\u30dd\u30ea\u30dc\u30ea\u300c\u300d\"\u3084\u308a]+", normalized)) and ("\u30dd\u30ea" in normalized or "\u30dc\u30ea" in normalized)


def prepare_source_for_translation(
    text: str,
    glossary: TranslationGlossary | None = None,
) -> tuple[str, list[dict[str, str]]]:
    active = glossary or load_translation_glossary(None)
    prepared = normalize_japanese_for_translation(text)
    replacements: list[dict[str, str]] = []
    for replacement in sorted(active.source_replacements, key=lambda item: len(item.source), reverse=True):
        if replacement.source not in prepared:
            continue
        prepared = prepared.replace(replacement.source, replacement.target)
        replacements.append({"source": replacement.source, "target": replacement.target})
    prepared, regex_replacements = apply_source_pattern_replacements(prepared)
    replacements.extend(regex_replacements)
    prepared, honorific_replacements = preserve_source_honorifics(prepared)
    replacements.extend(honorific_replacements)
    return prepared, replacements


def apply_source_pattern_replacements(text: str) -> tuple[str, list[dict[str, str]]]:
    replacements: list[dict[str, str]] = []

    def replace_numbered_district(match: re.Match[str]) -> str:
        original = match.group(0)
        target = f"{match.group(1)}\u756a\u8857"
        replacements.append({"source": original, "target": target})
        return target

    text = re.sub(r"([0-9\uff10-\uff19]+)\u3070\u3093\u304c\u3044", replace_numbered_district, text)
    return text, replacements


def preserve_source_honorifics(text: str) -> tuple[str, list[dict[str, str]]]:
    suffix_pattern = "|".join(re.escape(source) for source, _target in HONORIFIC_SUFFIXES)
    suffix_targets = dict(HONORIFIC_SUFFIXES)
    pattern = re.compile(
        rf"(?P<name>[\u3040-\u30ff\u30fc\u4e00-\u9fffA-Za-z][\u3040-\u30ff\u30fc\u4e00-\u9fffA-Za-z0-9_-]{{0,24}})"
        rf"(?P<suffix>{suffix_pattern})"
    )
    replacements: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        suffix = match.group("suffix")
        if name in HONORIFIC_FAMILY_BASES:
            return match.group(0)
        if name.endswith(("\u5c4b", "\u5e97", "\u4f1a\u793e", "\u5b66\u6821", "\u5bb6")):
            return match.group(0)
        target = suffix_targets[suffix]
        original = match.group(0)
        rendered = f"{name} {target}"
        replacements.append({"source": original, "target": rendered})
        return rendered

    return pattern.sub(replace, text), replacements


def postprocess_translation(
    source_text: str,
    prepared_text: str,
    translated_text: str,
    glossary: TranslationGlossary | None = None,
) -> str:
    translated, _changes = apply_target_replacements(source_text, prepared_text, translated_text, glossary)
    translated = re.sub(r"\s+", " ", translated).strip()
    translated = translated.replace("P2O", "P2")
    translated = translated.replace("P 2", "P2")
    return translated


def apply_target_replacements(
    source_text: str,
    prepared_text: str,
    translated_text: str,
    glossary: TranslationGlossary | None = None,
) -> tuple[str, list[dict[str, str]]]:
    active = glossary or load_translation_glossary(None)
    combined_source = f"{source_text}\n{prepared_text}"
    result = translated_text
    changes: list[dict[str, str]] = []
    result, honorific_changes = apply_honorific_title_replacements(combined_source, result)
    changes.extend(honorific_changes)
    for replacement in active.target_replacements:
        if replacement.when_source_contains and replacement.when_source_contains not in combined_source:
            continue
        pattern = re.compile(re.escape(replacement.source), flags=re.IGNORECASE)
        if pattern.search(result) is None:
            continue
        result = pattern.sub(replacement.target, result)
        changes.append(
            {
                "source": replacement.source,
                "target": replacement.target,
                "when_source_contains": replacement.when_source_contains or "",
            }
        )
    return result, changes


def apply_honorific_title_replacements(source_text: str, translated_text: str) -> tuple[str, list[dict[str, str]]]:
    result = translated_text
    changes: list[dict[str, str]] = []
    replacements = (
        ("sama", ("sama", "\u69d8", "\u3055\u307e", "\u30b5\u30de"), r"(?:Mr|Mrs|Ms|Miss|Lord|Lady|Sir|Madam|Master|Princess|Prince|Queen|King)"),
        ("sensei", ("sensei", "\u5148\u751f", "\u305b\u3093\u305b\u3044", "\u30bb\u30f3\u30bb\u30a4"), r"(?:Dr|Doctor|Professor|Prof|Teacher)"),
        ("senpai", ("senpai", "\u5148\u8f29", "\u305b\u3093\u3071\u3044", "\u30bb\u30f3\u30d1\u30a4"), r"(?:Senior|Upperclassman|Senpai)"),
        ("dono", ("dono", "\u6bbf", "\u3069\u306e", "\u30c9\u30ce"), r"(?:Lord|Lady|Sir|Madam)"),
        ("san", ("san", "\u3055\u3093", "\u30b5\u30f3"), r"(?:Mr|Mrs|Ms|Miss)"),
    )
    for honorific, source_markers, title_pattern in replacements:
        if not any(marker in source_text for marker in source_markers):
            continue
        suffix_pattern = re.compile(rf"\b(?P<name>[A-Z][A-Za-z0-9'_-]*)[-\s]+{honorific}\b", flags=re.IGNORECASE)

        def normalize_suffix(match: re.Match[str]) -> str:
            rendered = f"{match.group('name')} {honorific}"
            if match.group(0) != rendered:
                changes.append({"source": match.group(0), "target": rendered, "when_source_contains": honorific})
            return rendered

        result = suffix_pattern.sub(normalize_suffix, result)
        pattern = re.compile(rf"\b(?P<title>{title_pattern})\.?\s+(?P<name>[A-Z][A-Za-z0-9'_-]*)\b", flags=re.IGNORECASE)

        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            changes.append(
                {
                    "source": match.group(0),
                    "target": f"{name} {honorific}",
                    "when_source_contains": honorific,
                }
            )
            return f"{name} {honorific}"

        result = pattern.sub(replace, result)
    return result, changes


def translation_debug_info(
    source_text: str,
    translated_text: str,
    *,
    glossary_path: Path | None = None,
) -> dict[str, object]:
    glossary = load_translation_glossary(glossary_path)
    normalized = normalize_japanese_for_translation(source_text)
    prepared, source_replacements = prepare_source_for_translation(normalized, glossary)
    _postprocessed, target_replacements = apply_target_replacements(
        normalized,
        prepared,
        translated_text,
        glossary,
    )
    payload: dict[str, object] = {
        "normalized_source": normalized,
    }
    if prepared != normalized:
        payload["prepared_source"] = prepared
    if source_replacements:
        payload["glossary_source_replacements"] = source_replacements
    if target_replacements:
        payload["glossary_target_replacements"] = target_replacements
    if glossary.source_path is not None:
        payload["glossary_path"] = str(glossary.source_path)
    return payload


@lru_cache(maxsize=8)
def load_translation_glossary(path: Path | str | None = None) -> TranslationGlossary:
    source_replacements = list(default_source_replacements())
    exact_phrases: dict[str, str] = {}
    target_replacements = list(default_target_replacements())
    source_path = resolve_glossary_path(path)
    if source_path is not None:
        logger.info("Loading translation glossary: %s", source_path)
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"Could not read glossary file: {source_path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Glossary file is not valid JSON: {source_path}") from exc

        for item in payload.get("source_replacements", []):
            source = str(item.get("source", ""))
            target = str(item.get("target", ""))
            if source and target:
                source_replacements.append(SourceReplacement(source, target))

        custom_phrases = payload.get("exact_phrases", {})
        if isinstance(custom_phrases, dict):
            for source, target in custom_phrases.items():
                exact_phrases[normalize_phrase_key(normalize_japanese_for_translation(str(source)))] = str(target)
        else:
            for item in custom_phrases:
                source = str(item.get("source", ""))
                target = str(item.get("target", ""))
                if source and target:
                    exact_phrases[normalize_phrase_key(normalize_japanese_for_translation(source))] = target

        for item in payload.get("target_replacements", []):
            source = str(item.get("source", ""))
            target = str(item.get("target", ""))
            when = item.get("when_source_contains")
            if source and target:
                target_replacements.append(TargetReplacement(source, target, str(when) if when else None))

    return TranslationGlossary(
        source_replacements=tuple(source_replacements),
        exact_phrases=exact_phrases,
        target_replacements=tuple(target_replacements),
        source_path=source_path,
    )


def resolve_glossary_path(path: Path | str | None) -> Path | None:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        candidates.append(Path.cwd() / "translation_glossary.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if path:
        raise RuntimeError(f"Glossary file does not exist: {Path(path)}")
    return None


def default_source_replacements() -> tuple[SourceReplacement, ...]:
    pairs = (
        ("\u3082\u3046\u3059\u308b\u307e\u3059", "\u7533\u3057\u307e\u3059"),
        ("\u3088\u308d\u308d\u3059\u304a\u306d\u304c\u3044\u3059\u308b\u307e\u3059", "\u3088\u308d\u3057\u304f\u304a\u9858\u3044\u3057\u307e\u3059"),
        ("\u304a\u304b\u3057\u3084\u3055\u3093", "\u304a\u83d3\u5b50\u5c4b\u3055\u3093"),
        ("\u308f\u308b\u3082\u306e", "\u60aa\u8005"),
        ("\u306e\u3063\u3068\u3089\u308c\u305f", "\u4e57\u3063\u53d6\u3089\u308c\u305f"),
        ("\u305f\u3059\u3051\u306b\u3044\u304f", "\u52a9\u3051\u306b\u884c\u304f"),
        ("\u305f\u3059\u3051\u308c\u3044\u304f", "\u52a9\u3051\u306b\u884c\u304f"),
        ("\u304d\u3092\u3064\u3051\u308d", "\u6c17\u3092\u3064\u3051\u308d"),
        ("\u3066\u304d", "\u6575"),
        ("\u308f\u306a", "\u7f60"),
        ("\u3057\u304b\u3051\u3089\u308c\u3066\u308b", "\u4ed5\u639b\u3051\u3089\u308c\u3066\u308b"),
        ("\u3075\u305f\u308a", "\u4e8c\u4eba"),
        ("\u308a\u3063\u3071", "\u7acb\u6d3e"),
        ("\u3055\u3044\u3060\u3044", "\u6700\u5927"),
        ("\u3070\u3057\u3087", "\u5834\u6240"),
        ("\u30d8\u3084", "\u90e8\u5c4b"),
        ("\u3078\u3084", "\u90e8\u5c4b"),
        ("\u3061\u3061", "\u7236"),
        ("\u306f\u306f", "\u6bcd"),
    )
    return tuple(SourceReplacement(source, target) for source, target in pairs)


def default_target_replacements() -> tuple[TargetReplacement, ...]:
    return (
        TargetReplacement("boys", "you guys", "\u304a\u307e\u3048\u3089"),
        TargetReplacement("girls", "you guys", "\u304a\u307e\u3048\u3089"),
        TargetReplacement("men", "you guys", "\u304a\u307e\u3048\u3089"),
        TargetReplacement("women", "you guys", "\u304a\u307e\u3048\u3089"),
        TargetReplacement("will be able to", "can"),
        TargetReplacement("you'll be able to", "you can"),
        TargetReplacement("I'm going to go", "I'll go"),
    )
