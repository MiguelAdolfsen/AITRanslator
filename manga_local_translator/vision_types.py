from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VisionNumberMapEntry:
    number: int
    line_id: str
    order_index: int
    source_text: str
    translated_text: str
    box: tuple[int, int, int, int]
    state_key: str = ""
    block_id: str = ""
    source_hash: str = ""


@dataclass(frozen=True)
class VisionArtifact:
    mode: str
    image_path: Path
    number_map: tuple[VisionNumberMapEntry, ...]
    preprocessing_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisionRepairRequest:
    line_id: str
    number: int
    source_text: str
    current_translation: str
    issue: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class VisionFactsRequest:
    line_id: str
    number: int
    source_text: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class VisionFactsResult:
    line_id: str
    number: int
    source_text: str
    bubble_type: str = "unknown"
    line_role: str = "unknown"
    speaker_position: str = "unknown"
    speaker_anchor: str = "unknown"
    visible_emotion: str = "unknown"
    tone_hint: str = "unknown"
    observable_action: str = "unknown"
    mapping_confidence: str = "medium"
    facts: tuple[str, ...] = ()
    context_hints: tuple[str, ...] = ()
    needs_review: bool = False
    risk_flags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionRepairResult:
    line_id: str
    number: int
    action: str
    translation: str
    source_text: str = ""
    speaker: str | None = None
    speaker_confidence: str = "medium"
    situation: str | None = None
    visual_evidence_type: str = "weak"
    visual_evidence: str = ""
    confidence: str = "medium"
    needs_review: bool = False
    risk_flags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    raw: dict[str, object] = field(default_factory=dict)
