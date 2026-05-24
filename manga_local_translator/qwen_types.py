from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class QwenGenerationSettings:
    temperature: float
    top_p: float
    top_k: int = 20
    min_p: float = 0.0
    num_predict: int = 96


@dataclass(frozen=True)
class QwenVerificationDecision:
    ok: bool
    choice: str
    reason: str
    translation: str | None = None
    unsupported_terms: tuple[str, ...] = ()
    raw_response: str = ""


@dataclass(frozen=True)
class QwenCriticDecision:
    ok: bool
    severity: str
    issues: tuple[str, ...] = ()
    reason: str = ""
    raw_response: str = ""
    source_evidence: tuple[str, ...] = ()
    translation_evidence: tuple[str, ...] = ()
    repair_recommended: bool | None = None


@dataclass(frozen=True)
class QwenPageTranslation:
    page_id: int
    translation: str
    confidence: float | None = None
    reason: str = ""


def qwen_settings_from_env(prefix: str, default: QwenGenerationSettings) -> QwenGenerationSettings:
    return QwenGenerationSettings(
        temperature=float(os.environ.get(f"{prefix}_TEMPERATURE", default.temperature)),
        top_p=float(os.environ.get(f"{prefix}_TOP_P", default.top_p)),
        top_k=int(os.environ.get(f"{prefix}_TOP_K", default.top_k)),
        min_p=float(os.environ.get(f"{prefix}_MIN_P", default.min_p)),
        num_predict=int(os.environ.get(f"{prefix}_NUM_PREDICT", default.num_predict)),
    )


def qwen_settings_to_debug_dict(settings: QwenGenerationSettings) -> dict[str, int | float]:
    return {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "min_p": settings.min_p,
        "num_predict": settings.num_predict,
    }


QWEN_TRANSLATION_SETTINGS = qwen_settings_from_env(
    "MANGA_QWEN_TRANSLATION",
    QwenGenerationSettings(temperature=0.2, top_p=0.8, num_predict=96),
)
QWEN_PAGE_SETTINGS = qwen_settings_from_env(
    "MANGA_QWEN_PAGE",
    QwenGenerationSettings(temperature=0.2, top_p=0.8, num_predict=768),
)
QWEN_VERIFICATION_SETTINGS = qwen_settings_from_env(
    "MANGA_QWEN_VERIFICATION",
    QwenGenerationSettings(temperature=0.1, top_p=0.8, num_predict=192),
)
QWEN_CRITIC_SETTINGS = qwen_settings_from_env(
    "MANGA_QWEN_CRITIC",
    QwenGenerationSettings(temperature=0.1, top_p=0.8, num_predict=192),
)
QWEN_REPAIR_SETTINGS = qwen_settings_from_env(
    "MANGA_QWEN_REPAIR",
    QwenGenerationSettings(temperature=0.4, top_p=0.9, num_predict=256),
)
