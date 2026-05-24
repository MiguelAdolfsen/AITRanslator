from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    detector: str = "ctd"
    ocr_engine: str = "manga-ocr"
    translator: str = "opus"
    qwen_mode: str = "block"
    erase_mode: str = "white"
    tesseract_cmd: str | None = None
    tesseract_lang: str = "jpn+jpn_vert"
    tesseract_psm: int = 11
    min_confidence: float = 20.0
    padding: int = 8
    render_expand: float = 2.2
    font_path: Path | None = None
    glossary_path: Path | None = None
    qwen_model_path: Path | None = None
    qwen_fallback_model_path: Path | None = None
    qwen_critic_model_path: Path | None = None
    vision_enabled: bool = False
    vision_facts_enabled: bool = False
    vision_mode: str = "numbered_page"
    vision_trigger: str = "suspicious"
    vision_model_path: Path | None = None
    vision_projector_path: Path | None = None
    base_font_size: int = 28
    recursive: bool = True
    overwrite: bool = False
    debug: bool = False
    resume: bool = False
    work_dir: Path | None = None
