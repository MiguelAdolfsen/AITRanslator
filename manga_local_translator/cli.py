from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import PipelineConfig
from .logging_utils import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manga-local-translator",
        description="Translate Japanese manga page images to English locally.",
    )
    parser.add_argument("input", type=Path, help="Input image file or folder.")
    parser.add_argument("output", type=Path, help="Output file or folder.")
    parser.add_argument(
        "--detector",
        choices=["ctd", "tesseract", "visual"],
        default="ctd",
        help="Text region detector. ctd is best quality but needs model install.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["manga-ocr", "tesseract"],
        default="manga-ocr",
        help="OCR reader. manga-ocr is slower but usually better for manga text.",
    )
    parser.add_argument(
        "--translator",
        choices=["opus", "qwen", "madlad", "argos", "none"],
        default="opus",
        help="Translation backend. opus is the default; qwen uses a local GGUF model through Ollama/llama.cpp.",
    )
    parser.add_argument(
        "--erase-mode",
        choices=["white", "inpaint"],
        default="white",
        help="How to remove the original Japanese text.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Path to tesseract.exe if it is not on PATH.",
    )
    parser.add_argument(
        "--tesseract-lang",
        default="jpn+jpn_vert",
        help="Tesseract language pack combination.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=11,
        help="Tesseract page segmentation mode. 11 is sparse text.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=20.0,
        help="Discard OCR text below this confidence.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Pixels added around detected text when erasing.",
    )
    parser.add_argument(
        "--render-expand",
        type=float,
        default=2.2,
        help="How much to expand OCR boxes before drawing English text.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Optional TrueType font path for English text.",
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=None,
        help="Optional translation_glossary.json path for source cleanup, exact phrases, and name fixes.",
    )
    parser.add_argument(
        "--qwen-model",
        type=Path,
        default=None,
        help="Optional Qwen text GGUF path. Defaults to the preferred local Q4 model.",
    )
    parser.add_argument(
        "--qwen-fallback-model",
        type=Path,
        default=None,
        help="Optional second Qwen text GGUF used only for rejected/repaired/suspicious blocks.",
    )
    parser.add_argument(
        "--qwen-mode",
        choices=["block", "page"],
        default="block",
        help="Qwen translation strategy. block is current stable behavior; page translates ordered bubbles together.",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable local Qwen vision repair for suspicious translated blocks.",
    )
    parser.add_argument(
        "--vision-facts",
        action="store_true",
        help="Use local Qwen vision before translation to extract visual facts only.",
    )
    parser.add_argument(
        "--vision-mode",
        choices=["numbered_page", "page_image"],
        default="numbered_page",
        help="Vision artifact mode. numbered_page masks text regions and labels them for the vision model.",
    )
    parser.add_argument(
        "--vision-trigger",
        choices=["suspicious", "layout", "all"],
        default="suspicious",
        help="Which blocks should be sent to vision repair.",
    )
    parser.add_argument(
        "--vision-model",
        type=Path,
        default=None,
        help="Optional Qwen vision GGUF path. Defaults to the selected Qwen text model.",
    )
    parser.add_argument(
        "--vision-projector",
        type=Path,
        default=None,
        help="Optional Qwen mmproj GGUF path. Defaults to .models/qwen/mmproj-Qwen3.5-9B-BF16.gguf.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=28,
        help="Starting font size for rendered English text.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process nested folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing translated files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write debug images with OCR boxes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse batched hybrid intermediate files when available.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional folder for batched hybrid intermediate files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = configure_logging(reset=True)
    logger.info("CLI started")
    logger.debug("CLI args: %s", args)
    config = PipelineConfig(
        detector=args.detector,
        ocr_engine=args.ocr_engine,
        translator=args.translator,
        erase_mode=args.erase_mode,
        tesseract_cmd=args.tesseract_cmd,
        tesseract_lang=args.tesseract_lang,
        tesseract_psm=args.psm,
        min_confidence=args.min_confidence,
        padding=args.padding,
        render_expand=args.render_expand,
        font_path=args.font,
        glossary_path=args.glossary,
        qwen_model_path=args.qwen_model,
        qwen_fallback_model_path=args.qwen_fallback_model,
        qwen_mode=args.qwen_mode,
        vision_enabled=args.vision,
        vision_facts_enabled=args.vision_facts,
        vision_mode=args.vision_mode,
        vision_trigger=args.vision_trigger,
        vision_model_path=args.vision_model,
        vision_projector_path=args.vision_projector,
        base_font_size=args.font_size,
        recursive=args.recursive,
        overwrite=args.overwrite,
        debug=args.debug,
        resume=args.resume,
        work_dir=args.work_dir,
    )
    from .pipeline import process_folder

    logger.info("Debug log: %s", log_path)
    process_folder(args.input, args.output, config)
    logger.info("CLI finished")
    return 0
