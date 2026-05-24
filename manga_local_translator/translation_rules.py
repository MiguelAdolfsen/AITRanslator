from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .logging_utils import shorten

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
    return re.sub(r"[\u2026\u3002\uff0e.\u3001,!?\uff01\uff1f\u300c\u300d\u300e\u300f\s]+$", "", text)


def translate_known_phrase(text: str, glossary: TranslationGlossary | None = None) -> str | None:
    compact = normalize_phrase_key(text)
    if glossary is not None and compact in glossary.exact_phrases:
        return glossary.exact_phrases[compact]

    prepared = compact
    if glossary is not None:
        prepared, _replacements = prepare_source_for_translation(compact, glossary)
        if prepared in glossary.exact_phrases:
            return glossary.exact_phrases[prepared]

    phrasebook = {
        "\u3042": "Ah...",
        "\u3042\u3042": "Ah...",
        "\u3044\u3084": "No...",
        "\u3046\u3080": "Hmm.",
        "\u3046\u3093": "Yeah.",
        "\u3048": "Huh?",
        "\u3048\u3063": "Huh?",
        "\u306a\u3063": "Wha...!?",
        "\u304f\u3063": "Ngh.",
        "\u3046\u3043": "Whee!",
        "\u304a\u30fc": "Oh!",
        "\u304a\u304a": "Oh!",
        "\u304d\u3083": "Eek!",
        "\u304d\u3083\u3042": "Eek!",
        "\u3046\u308f": "Whoa!",
        "\u3046\u308f\u3042": "Whoa!",
        "\u3042\u308c": "Huh?",
        "\u3042\u3063": "Ah!",
        "\u3042\u3063\u3063": "Ah!",
        "\u306f\u3042": "Hah...",
        "\u306f\u3041": "Hah...",
        "\u306f\u3041\u306f\u3041": "Pant pant.",
        "\u3072\u3043": "Eek!",
        "\u3072\u3048": "Eek!",
        "\u3093": "Hm?",
        "\u3093\u3093": "Mm.",
        "\u3061\u3063": "Tch.",
        "\u304e\u3083": "Gah!",
        "\u304e\u3083\u3042": "Gyaa!",
        "\u3042\u306f": "Haha.",
        "\u3042\u306f\u306f": "Hahaha.",
        "\u30cf\u30cf": "Haha.",
        "\u30cf\u30cf\u30cf": "Hahaha.",
        "\u30d5\u30d5": "Heh heh.",
        "\u30d5\u30d5\u30d5": "Heh heh.",
        "\u30d8\u30d8": "Hehe.",
        "\u30d8\u30d8\u30d8": "Hehe.",
        "\u30cb\u30b3": "Smile.",
        "\u30cb\u30b3\u30c3": "Smile.",
        "\u30cb\u30e4": "Grin.",
        "\u30cb\u30e4\u30c3": "Grin.",
        "\u30ae\u30ed": "Glare.",
        "\u30b8\u30ed": "Stare.",
        "\u30b8\u30fc": "Stare...",
        "\u30b8\u30c3": "Stare.",
        "\u30b7\u30fc\u30f3": "Silence.",
        "\u30b7\u30f3": "Silence.",
        "\u30b7\u30fc\u30f3\u2026": "Silence...",
        "\u30ac\u30fc\u30f3": "Shock.",
        "\u30ac\u30fc\u30f3\u2026": "Shock...",
        "\u30c9\u30ad": "Thump.",
        "\u30c9\u30ad\u30c9\u30ad": "Thump thump.",
        "\u30c9\u30af\u30f3": "Thump.",
        "\u30d0\u30af\u30d0\u30af": "Thump thump.",
        "\u30c9\u30f3": "Thud.",
        "\u30c9\u30f3\u30c3": "Thud!",
        "\u30c9\u30ab": "Wham!",
        "\u30d0\u30f3": "Bang!",
        "\u30d0\u30f3\u30c3": "Bang!",
        "\u30d0\u30bf\u30f3": "Slam!",
        "\u30d0\u30bf": "Flop.",
        "\u30c9\u30b5": "Thump.",
        "\u30ac\u30c1\u30e3": "Clack.",
        "\u30ac\u30c1\u30e3\u30f3": "Clatter.",
        "\u30ab\u30c1\u30e3": "Click.",
        "\u30ab\u30c1": "Click.",
        "\u30b3\u30c4": "Tap.",
        "\u30b3\u30c4\u30b3\u30c4": "Tap tap.",
        "\u30c8\u30f3": "Tap.",
        "\u30c8\u30f3\u30c8\u30f3": "Tap tap.",
        "\u30c6\u30af\u30c6\u30af": "Step step.",
        "\u30bf\u30c3": "Dash.",
        "\u30c0\u30c3": "Dash.",
        "\u30c0\u30c0\u30c3": "Dash dash.",
        "\u30b9\u30bf": "Step.",
        "\u30b9\u30bf\u30b9\u30bf": "Step step.",
        "\u30b9\u30c3": "Whoosh.",
        "\u30b9\u30a5": "Whoosh.",
        "\u30b5\u30c3": "Whoosh.",
        "\u30d2\u30e5\u30fc": "Whoosh.",
        "\u30d2\u30e5\u30f3": "Whizz.",
        "\u30d3\u30e5\u30fc": "Whoosh.",
        "\u30b4\u30b4\u30b4": "Rumble...",
        "\u30b4\u30b4\u30b4\u30b4": "Rumble...",
        "\u30b6\u30ef": "Murmur.",
        "\u30b6\u30ef\u30b6\u30ef": "Murmur murmur.",
        "\u30ac\u30e4\u30ac\u30e4": "Chatter chatter.",
        "\u30ef\u30a4\u30ef\u30a4": "Chatter chatter.",
        "\u30b6\u30fc": "Pour.",
        "\u30b6\u30fc\u30b6\u30fc": "Pouring.",
        "\u30dd\u30c4": "Drip.",
        "\u30dd\u30c4\u30dd\u30c4": "Drip drip.",
        "\u30d4\u30c1\u30e7\u30f3": "Drip.",
        "\u30ad\u30e9": "Sparkle.",
        "\u30ad\u30e9\u30ad\u30e9": "Sparkle sparkle.",
        "\u30d4\u30ab": "Flash.",
        "\u30d4\u30ab\u30c3": "Flash!",
        "\u30d4\u30ab\u30d4\u30ab": "Shiny.",
        "\u30dd\u30ea\u30dd\u30ea": "Crunch crunch.",
        "\u30dc\u30ea\u30dc\u30ea": "Crunch crunch.",
        "\u30d1\u30af": "Munch.",
        "\u30d1\u30af\u30d1\u30af": "Munch munch.",
        "\u30e2\u30b0\u30e2\u30b0": "Munch munch.",
        "\u30b4\u30af": "Gulp.",
        "\u30b4\u30af\u30b4\u30af": "Gulp gulp.",
        "\u30b9\u30e4": "Sleep.",
        "\u30b9\u30e4\u30b9\u30e4": "Zzz.",
        "\u30b0\u30fc": "Zzz.",
        "\u30b0\u30fc\u30b0\u30fc": "Zzz.",
        "\u30c9\u30ad\u30c3": "Thump!",
        "\u30d3\u30af": "Flinch.",
        "\u30d3\u30af\u30c3": "Flinch!",
        "\u30d6\u30eb": "Shiver.",
        "\u30d6\u30eb\u30d6\u30eb": "Shiver shiver.",
        "\u30aa\u30ed\u30aa\u30ed": "Fluster fluster.",
        "\u30a2\u30bf\u30d5\u30bf": "Flustered.",
        "\u30ac\u30af": "Drop.",
        "\u30ac\u30af\u30ac\u30af": "Shake shake.",
        "\u30d0\u30c3": "Wham!",
        "\u30d0\u30d0\u30c3": "Wham wham!",
        "\u30d0\u30ad": "Crack!",
        "\u30d0\u30ad\u30c3": "Crack!",
        "\u30dc\u30ad": "Snap!",
        "\u30dc\u30ad\u30c3": "Snap!",
        "\u30d1\u30f3": "Clap!",
        "\u30d1\u30f3\u30c3": "Clap!",
        "\u30d1\u30c1": "Snap.",
        "\u30d1\u30c1\u30f3": "Snap!",
        "\u30d1\u30c1\u30d1\u30c1": "Clap clap.",
        "\u30d1\u30bf": "Pat.",
        "\u30d1\u30bf\u30d1\u30bf": "Pat pat.",
        "\u30d1\u30bf\u30f3": "Shut.",
        "\u30d0\u30b5": "Rustle.",
        "\u30d0\u30b5\u30c3": "Rustle!",
        "\u30ac\u30b5": "Rustle.",
        "\u30ac\u30b5\u30ac\u30b5": "Rustle rustle.",
        "\u30ab\u30b5": "Rustle.",
        "\u30ab\u30b5\u30ab\u30b5": "Rustle rustle.",
        "\u30d2\u30bd": "Whisper.",
        "\u30d2\u30bd\u30d2\u30bd": "Whisper whisper.",
        "\u30b3\u30bd": "Sneak.",
        "\u30b3\u30bd\u30b3\u30bd": "Sneak sneak.",
        "\u30bd\u30ed": "Tiptoe.",
        "\u30bd\u30ed\u30bd\u30ed": "Tiptoe tiptoe.",
        "\u30ce\u30bd": "Lumber.",
        "\u30ce\u30bd\u30ce\u30bd": "Lumber lumber.",
        "\u30c0\u30e9": "Drip.",
        "\u30c0\u30e9\u30c0\u30e9": "Sweat sweat.",
        "\u30bf\u30e9": "Sweat.",
        "\u30bf\u30e9\u30fc": "Sweat...",
        "\u30c4\u30fc": "Drip...",
        "\u30dd\u30bf": "Drop.",
        "\u30dd\u30bf\u30dd\u30bf": "Drop drop.",
        "\u30d3\u30b7\u30e7": "Soaked.",
        "\u30d3\u30b7\u30e7\u30d3\u30b7\u30e7": "Soaked.",
        "\u30e0\u30ab": "Irritated.",
        "\u30e0\u30ab\u30e0\u30ab": "Irritated.",
        "\u30a4\u30e9": "Irritated.",
        "\u30a4\u30e9\u30a4\u30e9": "Irritated.",
        "\u30d7\u30f3": "Hmph.",
        "\u30d7\u30f3\u30d7\u30f3": "Fume fume.",
        "\u30ab\u30a1": "Blush.",
        "\u30ab\u30fc": "Blush.",
        "\u30dd\u30c3": "Blush.",
        "\u30dd\u30c3\u30c3": "Blush.",
        "\u30e2\u30b8": "Fidget.",
        "\u30e2\u30b8\u30e2\u30b8": "Fidget fidget.",
        "\u30ef\u30af": "Excited.",
        "\u30ef\u30af\u30ef\u30af": "Excited!",
        "\u30cb\u30b3\u30cb\u30b3": "Smile smile.",
        "\u30cb\u30e4\u30cb\u30e4": "Grin grin.",
        "\u30e1\u30e9": "Flare.",
        "\u30e1\u30e9\u30e1\u30e9": "Flare flare.",
        "\u30dc\u30fc": "Dazed.",
        "\u30dc\u30fc\u30c3": "Dazed.",
        "\u30dd\u30ab\u30f3": "Blank stare.",
        "\u30ad\u30e7\u30c8\u30f3": "Blank stare.",
        "\u30d4\u30bf": "Stop.",
        "\u30d4\u30bf\u30c3": "Stop.",
        "\u30d4\u30bf\u30ea": "Stop.",
        "\u30b0\u30a4": "Yank.",
        "\u30b0\u30a4\u30c3": "Yank!",
        "\u30ae\u30e5": "Squeeze.",
        "\u30ae\u30e5\u30c3": "Squeeze!",
        "\u30ae\u30e5\u30a6": "Squeeze.",
        "\u30ca\u30c7": "Pat.",
        "\u30ca\u30c7\u30ca\u30c7": "Pat pat.",
        "\u30b5\u30ef": "Touch.",
        "\u30b5\u30ef\u30b5\u30ef": "Touch touch.",
        "\u30d0\u30bf\u30d0\u30bf": "Flap flap.",
        "\u30d1\u30bf\u30d1\u30bf": "Flutter flutter.",
        "\u30d2\u30e9": "Flutter.",
        "\u30d2\u30e9\u30d2\u30e9": "Flutter flutter.",
        "\u30af\u30eb": "Turn.",
        "\u30af\u30eb\u30c3": "Turn.",
        "\u30b0\u30eb": "Spin.",
        "\u30b0\u30eb\u30b0\u30eb": "Spin spin.",
        "\u30d5\u30e9": "Wobble.",
        "\u30d5\u30e9\u30d5\u30e9": "Wobble wobble.",
        "\u30e8\u30ed": "Stagger.",
        "\u30e8\u30ed\u30e8\u30ed": "Stagger stagger.",
        "\u30b4\u30ed": "Roll.",
        "\u30b4\u30ed\u30b4\u30ed": "Roll roll.",
        "\u30b3\u30ed": "Roll.",
        "\u30b3\u30ed\u30b3\u30ed": "Roll roll.",
        "\u30ba\u30eb": "Drag.",
        "\u30ba\u30eb\u30ba\u30eb": "Drag drag.",
        "\u30ba\u30b7": "Thud.",
        "\u30ba\u30b7\u30f3": "Thoom.",
        "\u30ba\u30c9\u30f3": "Boom!",
        "\u30c9\u30fc\u30f3": "Boom!",
        "\u30c9\u30ab\u30fc\u30f3": "Kaboom!",
        "\u30d0\u30fc\u30f3": "Bam!",
        "\u30ac\u30fc": "Roar.",
        "\u30ac\u30aa": "Roar!",
        "\u30ac\u30aa\u30fc": "Roar!",
        "\u30ad\u30fc\u30f3": "Ring...",
        "\u30ad\u30fc\u30f3\u30b3\u30fc\u30f3": "Ding dong.",
        "\u30d4\u30f3\u30dd\u30f3": "Ding dong.",
        "\u30d4\u30ed\u30f3": "Ding.",
        "\u30d4\u30d4": "Beep beep.",
        "\u30d4\u30d4\u30c3": "Beep!",
        "\u30d4\u30fc": "Beep.",
        "\u30d6\u30fc": "Bzz.",
        "\u30d6\u30fc\u30f3": "Buzz.",
        "\u30d6\u30f3": "Whoosh.",
        "\u30d6\u30f3\u30d6\u30f3": "Swing swing.",
        "\u30d0\u30b7": "Smack.",
        "\u30d0\u30b7\u30c3": "Smack!",
        "\u30d3\u30b7": "Whack.",
        "\u30d3\u30b7\u30c3": "Whack!",
        "\u30da\u30c1": "Smack.",
        "\u30da\u30c1\u30f3": "Smack!",
        "\u30da\u30bf": "Plop.",
        "\u30da\u30bf\u30da\u30bf": "Plop plop.",
        "\u30d9\u30bf": "Splat.",
        "\u30d9\u30bf\u30d9\u30bf": "Sticky.",
        "\u30d9\u30c1\u30e3": "Splat.",
        "\u30b0\u30c3": "Grip.",
        "\u30b0\u30c3\u30b0\u30c3": "Grip grip.",
        "\u30ae\u30ea": "Grit.",
        "\u30ae\u30ea\u30ae\u30ea": "Grit grit.",
        "\u30ab\u30ea": "Scratch.",
        "\u30ab\u30ea\u30ab\u30ea": "Scratch scratch.",
        "\u30dd\u30ea": "Crunch.",
        "\u30dc\u30ea": "Crunch.",
        "\u30b5\u30af": "Crisp.",
        "\u30b5\u30af\u30b5\u30af": "Crunch crunch.",
        "\u305f\u305f\u305f": "Tap tap tap.",
        "\u30bf\u30bf\u30bf": "Tap tap tap.",
        "\u304a\u30fc\u304d\u3069\u30fc\u304d\u30fc": "Okey dokey!",
        "\u304a\u30fc\u304d\uff01\u3069\u30fc\u304d\u30fc": "Okey dokey!",
        "\u304a\u30fc\u304d!\u3069\u30fc\u304d\u30fc": "Okey dokey!",
        "\u30ba\u30ba": "Sip.",
        "\u30ba\u30ba\u30c3": "Slurp.",
        "\u30ba\u30eb\u30ba\u30eb": "Slurp slurp.",
        "\u30b8\u30e5\u30fc": "Sizzle.",
        "\u30b0\u30c4\u30b0\u30c4": "Simmer simmer.",
        "\u30e0\u30b7\u30e3": "Chomp.",
        "\u30e0\u30b7\u30e3\u30e0\u30b7\u30e3": "Chomp chomp.",
        "\u6bcd\u3055\u3093": "Mom",
        "\u304a\u6bcd\u3055\u3093": "Mom",
        "\u3088\u304b\u3063\u305f": "I'm glad.",
        "\u3061\u3087\u3063\u3068": "Wait.",
        "\u3054\u3081\u3093": "Sorry.",
        "\u3059\u307f\u307e\u305b\u3093": "Sorry.",
        "\u3060\u3044\u3058\u3087\u3046\u3076": "Are you okay?",
        "\u3088\u308d\u3057\u304f\u304a\u9858\u3044\u3057\u307e\u3059": "Nice to meet you.",
        "\u30af\u30b9\u30af\u30b9": "Hehe.",
    }
    if compact in phrasebook:
        return phrasebook[compact]
    if prepared in phrasebook:
        return phrasebook[prepared]
    if is_sasuga_senpai_phrase(prepared):
        return "A-as expected, senpai...!"
    if is_crunching_sfx_phrase(prepared):
        return "Crunch crunch."
    if is_watch_out_trap_senpai_phrase(prepared):
        return "Watch out, the enemy may have set a trap... As expected, senpai!"
    if compact.startswith("\u3042\u3042") and len(compact) <= 4:
        return "Ah..."
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
        rf"(?P<name>[\u30a0-\u30ff\u30fc\u4e00-\u9fffA-Za-z][\u30a0-\u30ff\u30fc\u4e00-\u9fffA-Za-z0-9_-]{{0,24}})"
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
        ("sama", ("sama", "\u69d8", "\u3055\u307e", "\u30b5\u30de"), r"(?:Mr|Mrs|Ms|Miss|Lord|Lady|Sir|Madam|Master)"),
        ("sensei", ("sensei", "\u5148\u751f", "\u305b\u3093\u305b\u3044", "\u30bb\u30f3\u30bb\u30a4"), r"(?:Dr|Professor|Prof|Teacher)"),
        ("senpai", ("senpai", "\u5148\u8f29", "\u305b\u3093\u3071\u3044", "\u30bb\u30f3\u30d1\u30a4"), r"(?:Senior|Senpai)"),
        ("dono", ("dono", "\u6bbf", "\u3069\u306e", "\u30c9\u30ce"), r"(?:Lord|Lady|Sir|Madam)"),
        ("san", ("san", "\u3055\u3093", "\u30b5\u30f3"), r"(?:Mr|Mrs|Ms|Miss)"),
    )
    for honorific, source_markers, title_pattern in replacements:
        if not any(marker in source_text for marker in source_markers):
            continue
        pattern = re.compile(rf"\b(?P<title>{title_pattern})\.?\s+(?P<name>[A-Z][A-Za-z0-9'_-]*)\b")

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
