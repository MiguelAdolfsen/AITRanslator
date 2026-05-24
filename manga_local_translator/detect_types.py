from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextBlock:
    text: str
    box: tuple[int, int, int, int]
    confidence: float
    detector: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)

