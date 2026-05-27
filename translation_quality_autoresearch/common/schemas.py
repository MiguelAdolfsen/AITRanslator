from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hashing import sha256_text


CASE_REQUIRED = {
    "case_id",
    "source_text",
    "source_hash",
    "source_type",
    "ocr_risk",
    "context_before",
    "context_after",
    "forbidden_patterns",
    "must_preserve",
    "tags",
}


@dataclass(frozen=True)
class TranslationCase:
    case_id: str
    source_text: str
    source_hash: str
    source_type: str
    ocr_risk: str
    context_before: list[str]
    context_after: list[str]
    forbidden_patterns: list[str]
    must_preserve: list[str]
    tags: list[str]
    page_id: str | None = None
    group_id: str | None = None
    normalized_source: str | None = None
    speaker_hint: str | None = None
    orientation: str | None = None
    box: list[int] | None = None
    page_order: int | None = None
    glossary_terms: list[str] = field(default_factory=list)
    style_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranslationCase":
        missing = sorted(CASE_REQUIRED - set(payload))
        if missing:
            raise ValueError(f"case {payload.get('case_id', '<missing>')} missing required fields: {', '.join(missing)}")
        return cls(
            case_id=str(payload["case_id"]),
            source_text=str(payload["source_text"]),
            source_hash=str(payload["source_hash"]),
            source_type=str(payload["source_type"]),
            ocr_risk=str(payload["ocr_risk"]),
            context_before=list(payload["context_before"]),
            context_after=list(payload["context_after"]),
            forbidden_patterns=list(payload["forbidden_patterns"]),
            must_preserve=list(payload["must_preserve"]),
            tags=list(payload["tags"]),
            page_id=optional_str(payload.get("page_id")),
            group_id=optional_str(payload.get("group_id")),
            normalized_source=optional_str(payload.get("normalized_source")),
            speaker_hint=optional_str(payload.get("speaker_hint")),
            orientation=optional_str(payload.get("orientation")),
            box=list(payload["box"]) if isinstance(payload.get("box"), list) else None,
            page_order=int(payload["page_order"]) if payload.get("page_order") is not None else None,
            glossary_terms=list(payload.get("glossary_terms") or []),
            style_notes=list(payload.get("style_notes") or []),
            metadata=dict(payload.get("metadata") or {}),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.case_id:
            errors.append("case_id is empty")
        if not self.source_text.strip():
            errors.append(f"{self.case_id}: source_text is empty")
        source_for_hash = self.normalized_source or self.source_text
        if sha256_text(source_for_hash) != self.source_hash:
            errors.append(f"{self.case_id}: source_hash mismatch")
        for field_name in ("context_before", "context_after", "forbidden_patterns", "must_preserve", "tags"):
            if not isinstance(getattr(self, field_name), list):
                errors.append(f"{self.case_id}: {field_name} must be a list")
        return errors


@dataclass(frozen=True)
class ReferenceRecord:
    case_id: str
    reference_translations: list[str]
    acceptable_meaning_notes: list[str] = field(default_factory=list)
    known_bad_translations: list[str] = field(default_factory=list)
    required_meaning_units: list[str] = field(default_factory=list)
    forbidden_additions: list[str] = field(default_factory=list)
    human_priority: str = "normal"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceRecord":
        return cls(
            case_id=str(payload["case_id"]),
            reference_translations=list(payload.get("reference_translations") or []),
            acceptable_meaning_notes=list(payload.get("acceptable_meaning_notes") or []),
            known_bad_translations=list(payload.get("known_bad_translations") or []),
            required_meaning_units=list(payload.get("required_meaning_units") or []),
            forbidden_additions=list(payload.get("forbidden_additions") or []),
            human_priority=str(payload.get("human_priority") or "normal"),
        )


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    agent: str
    text: str
    raw_output: str = ""
    prompt_name: str | None = None
    prompt_hash: str | None = None
    model_name: str = ""
    latency_ms: float = 0.0
    cost_proxy: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Candidate":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            agent=str(payload["agent"]),
            text=str(payload.get("text") or ""),
            raw_output=str(payload.get("raw_output") or payload.get("text") or ""),
            prompt_name=optional_str(payload.get("prompt_name")),
            prompt_hash=optional_str(payload.get("prompt_hash")),
            model_name=str(payload.get("model_name") or ""),
            latency_ms=float(payload.get("latency_ms") or 0.0),
            cost_proxy=float(payload.get("cost_proxy") or 0.0),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "agent": self.agent,
            "text": self.text,
            "raw_output": self.raw_output,
            "prompt_name": self.prompt_name,
            "prompt_hash": self.prompt_hash,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "cost_proxy": self.cost_proxy,
            "metadata": self.metadata,
        }


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
