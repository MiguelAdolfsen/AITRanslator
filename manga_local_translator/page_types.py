from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detect_types import TextBlock
from .vision_types import VisionArtifact


@dataclass
class PreparedPage:
    image_path: Path
    output_path: Path
    image_bgr: Any
    width: int
    height: int
    raw_blocks: list[TextBlock]
    render_blocks: list[TextBlock]
    skipped_blocks: list[dict[str, object]]
    grouping_report: list[dict[str, object]]
    page_order_report: list[dict[str, object]]
    translations: dict[str, str] = field(default_factory=dict)
    translation_contexts: dict[str, dict[str, object]] = field(default_factory=dict)
    translation_fallback_blocks: list[dict[str, object]] = field(default_factory=list)
    vision_artifact: VisionArtifact | None = None
    vision_facts_artifact: VisionArtifact | None = None

    @property
    def chapter_context_after(self) -> str | None:
        if not self.page_order_report:
            return None
        return str(self.page_order_report[-1]["source_text"])
