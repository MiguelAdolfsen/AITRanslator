from __future__ import annotations

import io
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from collections.abc import Callable
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__
from . import gui_widgets as ui
from .config import PipelineConfig
from .ctd_setup import ctd_is_installed
from .image_io import iter_images
from .logging_utils import configure_logging, shorten
from .tesseract_utils import find_tesseract, tesseract_missing_message

logger = logging.getLogger(__name__)

APP_NAME = "Manga Translator"
SETTINGS_PATH = Path.home() / ".manga_local_translator" / "gui_settings.json"
MAX_LOG_LINES = 4000
LOG_LINE_LIMIT = 320
HYBRID_TRANSLATORS = {"qwen", "hy-mt2", "cat"}
GGUF_FILETYPES = (("GGUF model", "*.gguf"), ("All files", "*.*"))


@dataclass(frozen=True)
class Choice:
    value: str
    label: str
    description: str
    tag: str | None = None
    tag_level: str = "neutral"


TRANSLATORS = (
    Choice("opus", "OPUS-MT", "Fast Helsinki-NLP Japanese to English model. A dependable default.", "Default", "info"),
    Choice("hy-mt2", "Hy-MT2", "Tencent Hy-MT2 1.8B translation model. Works with optional Qwen review."),
    Choice("qwen", "Qwen", "Local Qwen GGUF language model with per-block or full-page context.", "LLM"),
    Choice("cat", "CAT-Translate", "CyberAgent 7B GGUF model with batched retry and review passes.", "7B"),
    Choice("madlad", "MADLAD-400", "Google's large multilingual translation model.", "Large"),
    Choice("argos", "Argos Translate", "Small offline OpenNMT language package that installs quickly."),
    Choice("none", "No translation", "Keeps the OCR text as-is to inspect detection and OCR results.", "Debug", "warning"),
)
DETECTORS = (
    Choice("ctd", "CTD", "Comic Text Detector finds speech bubbles and captions most reliably. Needs a one-time install."),
    Choice("tesseract", "Tesseract", "Uses Tesseract's layout analysis to locate text. Requires Tesseract OCR."),
    Choice("visual", "Visual", "Lightweight image heuristic with no extra install. Less precise."),
)
OCR_ENGINES = (
    Choice("manga-ocr", "manga-ocr", "Purpose-built for manga lettering. Slower, but usually the most accurate."),
    Choice("tesseract", "Tesseract", "General-purpose OCR. Requires Tesseract with Japanese language data."),
)
ERASE_MODES = (
    Choice("white", "White fill", "Paints detected text regions white. Clean and fast for speech bubbles."),
    Choice("inpaint", "Inpaint", "Reconstructs the background behind the text. Better for text over artwork."),
)
QWEN_MODES = (
    Choice("block", "Per block", "Translates each text block on its own. The stable default."),
    Choice("page", "Full page", "Translates the ordered bubbles of a page together for more context."),
)
VISION_MODES = (
    Choice("numbered_page", "Numbered page", "Masks text regions and labels them with numbers so the model can reference each block."),
    Choice("page_image", "Page image", "Sends the page image without numbered labels."),
)
VISION_TRIGGERS = (
    Choice("suspicious", "Suspicious", "Only blocks that the translation checks flagged as suspicious."),
    Choice("layout", "Layout issues", "Only blocks with layout warnings, such as text that does not fit its bubble."),
    Choice("all", "All blocks", "Every translated block. The most thorough option, and the slowest."),
)

SETTING_DEFAULTS: dict[str, object] = {
    "input": "",
    "output": "",
    "detector": "ctd",
    "ocr_engine": "manga-ocr",
    "erase_mode": "white",
    "translator": "opus",
    "qwen_mode": "block",
    "qwen_model": "",
    "qwen_fallback": "",
    "qwen_critic": "",
    "cat_model": "",
    "hy_mt2_model": "",
    "vision": False,
    "vision_facts": False,
    "vision_mode": "numbered_page",
    "vision_trigger": "suspicious",
    "glossary": "",
    "font": "",
    "font_size": 28,
    "render_expand": 2.2,
    "work_dir": "",
    "tesseract": "",
    "debug": True,
    "overwrite": True,
    "resume": False,
}
SETTING_CHOICES = {
    "detector": DETECTORS,
    "ocr_engine": OCR_ENGINES,
    "erase_mode": ERASE_MODES,
    "translator": TRANSLATORS,
    "qwen_mode": QWEN_MODES,
    "vision_mode": VISION_MODES,
    "vision_trigger": VISION_TRIGGERS,
}
PRESERVED_ON_RESET = {"input", "output", "tesseract"}

PAGES = (
    ("project", "Project", "Pages & output", "Choose the manga pages to translate and where the results are saved."),
    ("translation", "Translation", "Translation engine", "Pick the model that turns Japanese text into English."),
    ("detection", "Detection & OCR", "Detection & OCR", "Control how text is found, read, and removed from each page."),
    ("vision", "Vision", "Vision assistance", "Optional Qwen vision passes that look at the page image itself."),
    ("typesetting", "Typesetting", "Typesetting", "Adjust how the English text is drawn back into each bubble."),
    ("setup", "Setup", "Models & tools", "Install the detector and translation models the pipeline uses."),
)


@dataclass(frozen=True)
class Component:
    key: str
    title: str
    description: str
    size_note: str
    job_label: str


COMPONENTS = (
    Component("ctd", "Comic Text Detector", "Speech bubble and text region detector used by the CTD option.", "Source + ONNX model", "Installing CTD detector"),
    Component("opus", "OPUS-MT", "Default Japanese to English translation model.", "Hugging Face download", "Installing OPUS translator"),
    Component("hy-mt2", "Hy-MT2", "Tencent Hy-MT2 1.8B translation model.", "Large download", "Installing Hy-MT2 translator"),
    Component("cat", "CAT-Translate", "CyberAgent CAT-Translate 7B GGUF model.", "Large download", "Installing CAT translator"),
    Component("madlad", "MADLAD-400", "Google MADLAD-400 multilingual translation model.", "Large download", "Installing MADLAD translator"),
    Component("argos", "Argos JA to EN", "Argos Translate offline language package.", "Small download", "Installing Argos model"),
)


@dataclass(frozen=True)
class Check:
    level: str
    text: str
    action: tuple[str, Callable[[], None]] | None = None

    def key(self) -> tuple[str, str, str | None]:
        return (self.level, self.text, self.action[0] if self.action else None)


def choice_label(choices: tuple[Choice, ...], value: str) -> str:
    return next((choice.label for choice in choices if choice.value == value), value)


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class _QueueLogHandler(logging.Handler):
    """Forwards pipeline log records to the GUI event queue."""

    def __init__(self, sink: Callable[[tuple], None]) -> None:
        super().__init__(logging.INFO)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        if record.name in {__name__, "__main__"}:
            return
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed log arguments.
            return
        self._sink(("log", record.levelno, message, record.created))


class _QueueWriter(io.TextIOBase):
    """File-like object that forwards printed lines to the GUI event queue."""

    def __init__(self, sink: Callable[[tuple], None]) -> None:
        self._sink = sink
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._buffer += text.replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._sink(("line", line.rstrip(), "info"))
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._sink(("line", self._buffer.rstrip(), "info"))
        self._buffer = ""


class MangaTranslatorGui(tk.Tk):
    def __init__(self, *, log_path: Path | None = None) -> None:
        super().__init__()
        self.withdraw()
        self.title(APP_NAME)
        self.log_path = log_path
        self.scale = ui.ui_scale(self)
        self.fonts = ui.create_fonts(self)
        ui.apply_styles(self, self.fonts, self.scale)
        self.assets = ui.Assets(self, self.scale)
        self.configure(bg=ui.BG)

        self.vars: dict[str, tk.Variable] = {}
        for key, default in SETTING_DEFAULTS.items():
            if isinstance(default, bool):
                self.vars[key] = tk.BooleanVar(self, value=default)
            elif isinstance(default, int):
                self.vars[key] = tk.IntVar(self, value=default)
            elif isinstance(default, float):
                self.vars[key] = tk.DoubleVar(self, value=default)
            else:
                self.vars[key] = tk.StringVar(self, value=default)
        self._load_settings()

        self.events: queue.Queue[tuple] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._job: str | None = None
        self._job_label = ""
        self._job_started = 0.0
        self._progress_stage = ""
        self._progress_detail = ""
        self._blocked = True
        self._checks_signature: list[tuple] | None = None
        self._checks_after: str | None = None
        self._image_counts: dict[str, int] = {}
        self._auto_output: str | None = None
        self._tesseract_detected: str | None = None
        self.current_page = ""
        self.nav_items: dict[str, ui.NavItem] = {}
        self.pages: dict[str, ui.ScrollFrame] = {}
        self.install_buttons: dict[str, ttk.Button] = {}
        self._context_updaters: list[Callable[[], None]] = []

        self._build_ui()
        self._set_window_icon()
        self._prefill_tesseract_path()
        self._install_log_handler()
        self._bind_events()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._place_window()
        self._show_page("project")
        self._refresh_context()
        self._refresh_checks()
        self._set_idle_progress()
        self.deiconify()
        self.after(100, self._drain_events)
        logger.info("GUI initialized")

    # ------------------------------------------------------------------ settings

    def _text(self, key: str) -> str:
        return str(self.vars[key].get()).strip()

    def _path(self, key: str) -> Path | None:
        value = self._text(key)
        return Path(value) if value else None

    def _load_settings(self) -> None:
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning("Could not read GUI settings from %s", SETTINGS_PATH, exc_info=True)
            return
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            if key not in self.vars:
                continue
            default = SETTING_DEFAULTS[key]
            if key in SETTING_CHOICES and value not in {choice.value for choice in SETTING_CHOICES[key]}:
                continue
            if isinstance(default, bool) != isinstance(value, bool):
                continue
            if isinstance(default, (int, float)) and not isinstance(value, (int, float)):
                continue
            if isinstance(default, str) and not isinstance(value, str):
                continue
            self.vars[key].set(value)
        logger.info("Loaded GUI settings from %s", SETTINGS_PATH)

    def _save_settings(self) -> None:
        data = {}
        for key, variable in self.vars.items():
            try:
                data[key] = variable.get()
            except tk.TclError:
                data[key] = SETTING_DEFAULTS[key]
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug("Saved GUI settings to %s", SETTINGS_PATH)
        except OSError:
            logger.warning("Could not save GUI settings to %s", SETTINGS_PATH, exc_info=True)

    def _restore_defaults(self) -> None:
        if not messagebox.askyesno(
            "Restore defaults",
            "Reset all settings to their defaults?\n\nYour input, output, and Tesseract paths are kept.",
            parent=self,
        ):
            return
        for key, default in SETTING_DEFAULTS.items():
            if key not in PRESERVED_ON_RESET:
                self.vars[key].set(default)
        self._save_settings()
        self.log_message("Settings restored to defaults.", "muted")

    def _reset_typesetting(self) -> None:
        for key in ("font", "font_size", "render_expand"):
            self.vars[key].set(SETTING_DEFAULTS[key])

    # ------------------------------------------------------------------ layout

    def _px(self, value: float) -> int:
        return int(round(value * self.scale))

    def _place_window(self) -> None:
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        width = min(self._px(1320), int(screen_w * 0.94))
        height = min(self._px(840), int(screen_h * 0.88))
        self.minsize(min(self._px(1080), int(screen_w * 0.9)), min(self._px(640), int(screen_h * 0.8)))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _set_window_icon(self) -> None:
        icons = [self.assets.logo(size, unscaled=True) for size in (64, 32)]
        icons = [icon for icon in icons if icon is not None]
        if icons:
            try:
                self.iconphoto(True, *icons)
            except tk.TclError:
                logger.debug("Window icon could not be set", exc_info=True)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar().grid(row=0, column=0, sticky="nsw")

        main = tk.Frame(self, bg=ui.BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        builders = {
            "project": self._build_project_page,
            "translation": self._build_translation_page,
            "detection": self._build_detection_page,
            "vision": self._build_vision_page,
            "typesetting": self._build_typesetting_page,
            "setup": self._build_setup_page,
        }
        for key, nav_label, title, subtitle in PAGES:
            page = ui.ScrollFrame(main, bg=ui.BG, max_width=self._px(820), padding=(self._px(36), self._px(30)))
            page.grid(row=0, column=0, sticky="nsew")
            self._page_header(page.content, nav_label, title, subtitle)
            builders[key](page.content)
            self.pages[key] = page

        tk.Frame(self, bg=ui.BORDER, width=1).grid(row=0, column=2, sticky="ns")
        self._build_run_panel().grid(row=0, column=3, sticky="nse")

    def _build_sidebar(self) -> tk.Frame:
        bg = ui.SIDEBAR_BG
        sidebar = tk.Frame(self, bg=bg, width=self._px(236))
        sidebar.pack_propagate(False)

        brand = tk.Frame(sidebar, bg=bg, padx=20, pady=24)
        brand.pack(fill="x")
        logo = self.assets.logo(38)
        if logo is not None:
            tk.Label(brand, image=logo, bg=bg, bd=0).pack(side="left")
        brand_text = tk.Frame(brand, bg=bg)
        brand_text.pack(side="left", padx=(12, 0))
        tk.Label(brand_text, text=APP_NAME, font=self.fonts.brand, bg=bg, fg=ui.SIDEBAR_TEXT_ACTIVE).pack(anchor="w")
        tk.Label(brand_text, text="Japanese → English", font=self.fonts.small, bg=bg, fg=ui.SIDEBAR_TEXT).pack(anchor="w")

        tk.Label(sidebar, text="SETTINGS", font=self.fonts.caption, bg=bg, fg=ui.LOG_MUTED, anchor="w").pack(
            fill="x", padx=24, pady=(6, 8)
        )
        nav = tk.Frame(sidebar, bg=bg)
        nav.pack(fill="x", padx=10)
        for key, nav_label, _title, _subtitle in PAGES:
            item = ui.NavItem(nav, self.fonts, icon=ui.ICONS[key], text=nav_label, command=lambda k=key: self._show_page(k))
            item.pack(fill="x", pady=1)
            self.nav_items[key] = item

        footer = tk.Frame(sidebar, bg=bg, padx=22, pady=20)
        footer.pack(side="bottom", fill="x")
        status_row = tk.Frame(footer, bg=bg)
        status_row.pack(fill="x")
        self.sidebar_dot = tk.Label(status_row, text="●", font=self.fonts.small, bg=bg, fg=ui.LOG_SUCCESS)
        self.sidebar_dot.pack(side="left")
        self.sidebar_status = tk.Label(status_row, text="Ready", font=self.fonts.small_strong, bg=bg, fg=ui.SIDEBAR_TEXT_ACTIVE)
        self.sidebar_status.pack(side="left", padx=(6, 0))
        tk.Label(footer, text=f"Version {__version__} · Runs locally", font=self.fonts.small, bg=bg, fg=ui.LOG_MUTED).pack(
            anchor="w", pady=(4, 0)
        )
        return sidebar

    def _page_header(self, parent: tk.Frame, eyebrow: str, title: str, subtitle: str) -> None:
        header = tk.Frame(parent, bg=ui.BG)
        header.pack(fill="x", pady=(0, 20))
        tk.Label(header, text=eyebrow.upper(), font=self.fonts.caption, bg=ui.BG, fg=ui.ACCENT).pack(anchor="w")
        tk.Label(header, text=title, font=self.fonts.title, bg=ui.BG, fg=ui.TEXT).pack(anchor="w", pady=(2, 0))
        ui.WrapLabel(header, text=subtitle, font=self.fonts.body, bg=ui.BG, fg=ui.MUTED).pack(fill="x", pady=(4, 0))

    def _card(self, parent: tk.Frame, title: str | None = None, description: str | None = None, **kwargs) -> ui.Card:
        card = ui.Card(parent, self.fonts, title=title, description=description, **kwargs)
        card.pack(fill="x", pady=(0, 16))
        return card

    def _choice_field(
        self,
        parent: tk.Frame,
        *,
        title: str,
        key: str,
        choices: tuple[Choice, ...],
    ) -> tuple[tk.Frame, ui.SegmentedControl]:
        field = tk.Frame(parent, bg=ui.SURFACE)
        field.pack(fill="x")
        label = tk.Label(field, text=title, font=self.fonts.body_strong, bg=ui.SURFACE, fg=ui.TEXT, anchor="w")
        label.pack(fill="x", pady=(0, 8))
        variable = self.vars[key]
        control = ui.SegmentedControl(field, self.fonts, variable=variable, options=[(c.value, c.label) for c in choices])
        control.pack(anchor="w")
        hint = ui.WrapLabel(field, font=self.fonts.small, bg=ui.SURFACE, fg=ui.MUTED)
        hint.pack(fill="x", pady=(8, 0))

        def update_hint(*_args) -> None:
            value = variable.get()
            hint.configure(text=next((c.description for c in choices if c.value == value), ""))

        variable.trace_add("write", update_hint)
        update_hint()
        field.label = label  # type: ignore[attr-defined]
        return field, control

    # ------------------------------------------------------------------ pages

    def _build_project_page(self, parent: tk.Frame) -> None:
        card = self._card(parent, "Source & destination", "Folders are scanned recursively, and the output mirrors their structure.")
        ui.PathField(
            card.body,
            self.fonts,
            label="Manga pages",
            variable=self.vars["input"],
            buttons=(("Folder…", self._choose_input), ("Image…", self._choose_input_file)),
            hint="A folder of pages or a single image. Supports JPG, PNG, WebP, BMP, and TIFF.",
        ).pack(fill="x")
        ui.separator(card.body, pady=16)
        ui.PathField(
            card.body,
            self.fonts,
            label="Output folder",
            variable=self.vars["output"],
            buttons=(("Browse…", self._choose_output),),
            hint="Translated pages are written here. Filled in automatically when you pick the input.",
        ).pack(fill="x")

        card = self._card(parent, "Run options")
        rows = (
            ("overwrite", "Overwrite existing pages", "Translate pages again even if they already exist in the output folder."),
            ("resume", "Resume cached work", "Reuse intermediate results from an earlier CAT or Qwen review run."),
            ("debug", "Debug output", "Save box overlays and JSON reports next to each page for troubleshooting."),
        )
        for index, (key, title, description) in enumerate(rows):
            if index:
                ui.separator(card.body, pady=14)
            row, _switch = ui.switch_row(card.body, self.fonts, self.assets, title=title, description=description, variable=self.vars[key])
            row.pack(fill="x")

        card = self._card(parent, "Work cache", "Batched runs keep intermediate files here so they can resume later.")
        ui.PathField(
            card.body,
            self.fonts,
            label="Cache folder",
            variable=self.vars["work_dir"],
            buttons=(("Browse…", lambda: self._browse_dir("work_dir", "Choose work cache folder")),),
            hint="Leave empty to use a .manga-work folder inside the output folder.",
            optional=True,
        ).pack(fill="x")

    def _build_translation_page(self, parent: tk.Frame) -> None:
        card = self._card(parent, "Engine", "Models download once from the Setup page and then run fully offline.")
        grid = tk.Frame(card.body, bg=ui.SURFACE)
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1, uniform="engine")
        grid.columnconfigure(1, weight=1, uniform="engine")
        for index, choice in enumerate(TRANSLATORS):
            option = ui.OptionCard(
                grid,
                self.fonts,
                self.assets,
                variable=self.vars["translator"],
                value=choice.value,
                title=choice.label,
                description=choice.description,
                tag=choice.tag,
                tag_level=choice.tag_level,
            )
            row, column = divmod(index, 2)
            option.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 6 if column == 0 else 0), pady=6)

        dynamic = tk.Frame(parent, bg=ui.BG)
        dynamic.pack(fill="x")

        model_card = ui.Card(dynamic, self.fonts, title="Model", description="Override where the selected engine loads its model from.")
        qwen_mode_field, _ = self._choice_field(model_card.body, title="Qwen strategy", key="qwen_mode", choices=QWEN_MODES)
        qwen_model = ui.PathField(
            model_card.body,
            self.fonts,
            label="Qwen model",
            variable=self.vars["qwen_model"],
            buttons=(("Browse…", lambda: self._browse_file("qwen_model", "Choose primary Qwen GGUF model", GGUF_FILETYPES)),),
            hint="GGUF file. Leave empty to use the preferred local Q4 model.",
            optional=True,
        )
        hy_mt2_model = ui.PathField(
            model_card.body,
            self.fonts,
            label="Hy-MT2 model",
            variable=self.vars["hy_mt2_model"],
            buttons=(("Browse…", lambda: self._browse_dir("hy_mt2_model", "Choose local Hy-MT2 model folder")),),
            hint="Local model folder or Hugging Face id. Defaults to tencent/Hy-MT2-1.8B.",
            optional=True,
        )
        cat_model = ui.PathField(
            model_card.body,
            self.fonts,
            label="CAT model",
            variable=self.vars["cat_model"],
            buttons=(("Browse…", lambda: self._browse_file("cat_model", "Choose CAT model", GGUF_FILETYPES)),),
            hint="GGUF file or Hugging Face id. Defaults to .models/CAT-Translate, then cyberagent/CAT-Translate-7b.",
            optional=True,
        )

        review_card = ui.Card(
            dynamic,
            self.fonts,
            title="Quality review",
            description="Optional Qwen models that double-check suspicious blocks and repair rejected translations.",
        )
        ui.PathField(
            review_card.body,
            self.fonts,
            label="Critic model",
            variable=self.vars["qwen_critic"],
            buttons=(("Browse…", lambda: self._browse_file("qwen_critic", "Choose critic Qwen GGUF model", GGUF_FILETYPES)),),
            hint="Verifies suspicious accepted blocks.",
            optional=True,
        ).pack(fill="x")
        ui.separator(review_card.body, pady=16)
        ui.PathField(
            review_card.body,
            self.fonts,
            label="Fallback model",
            variable=self.vars["qwen_fallback"],
            buttons=(("Browse…", lambda: self._browse_file("qwen_fallback", "Choose fallback Qwen GGUF model", GGUF_FILETYPES)),),
            hint="Re-translates blocks that were rejected, repaired, or flagged by the critic.",
            optional=True,
        ).pack(fill="x")

        card = self._card(parent, "Glossary", "Fix names, recurring phrases, and common OCR mistakes before translation.")
        ui.PathField(
            card.body,
            self.fonts,
            label="Glossary file",
            variable=self.vars["glossary"],
            buttons=(("Browse…", lambda: self._browse_file("glossary", "Choose translation glossary", (("JSON glossary", "*.json"), ("All files", "*.*")))),),
            hint="JSON file. See translation_glossary.example.json for the format.",
            optional=True,
        ).pack(fill="x")

        def update() -> None:
            translator = self._text("translator")
            fields = []
            if translator == "qwen":
                fields = [qwen_mode_field, qwen_model]
            elif translator == "hy-mt2":
                fields = [hy_mt2_model]
            elif translator == "cat":
                fields = [cat_model]
            for widget in (qwen_mode_field, qwen_model, hy_mt2_model, cat_model):
                widget.pack_forget()
            for separator_line in getattr(model_card, "_separators", []):
                separator_line.destroy()
            model_card._separators = []  # type: ignore[attr-defined]
            for index, widget in enumerate(fields):
                if index:
                    model_card._separators.append(ui.separator(model_card.body, pady=16))  # type: ignore[attr-defined]
                widget.pack(fill="x")
            model_card.pack_forget()
            review_card.pack_forget()
            if fields:
                model_card.pack(fill="x", pady=(0, 16))
            if translator in HYBRID_TRANSLATORS:
                review_card.pack(fill="x", pady=(0, 16))

        self._context_updaters.append(update)

    def _build_detection_page(self, parent: tk.Frame) -> None:
        card = self._card(parent, "Text detection", "Finds speech bubbles, captions, and other lettering on the page.")
        self._choice_field(card.body, title="Detector", key="detector", choices=DETECTORS)
        ctd_notice = ui.Callout(
            card.body,
            self.fonts,
            text="Comic Text Detector is not installed yet. Install it once before translating.",
            level="warning",
            action=("Install now", lambda: self._install("ctd")),
        )

        card = self._card(parent, "Reading text")
        self._choice_field(card.body, title="OCR engine", key="ocr_engine", choices=OCR_ENGINES)
        ui.separator(card.body, pady=18)
        self._choice_field(card.body, title="Text removal", key="erase_mode", choices=ERASE_MODES)

        card = self._card(
            parent,
            "Tesseract OCR",
            "Used only when Tesseract is the detector or OCR engine.",
            trailing=lambda header: self._make_tesseract_badge(header),
        )
        ui.PathField(
            card.body,
            self.fonts,
            label="tesseract.exe",
            variable=self.vars["tesseract"],
            buttons=(("Browse…", self._choose_tesseract), ("Setup help", self._show_tesseract_help)),
            hint="Detected automatically when Tesseract is on PATH or in Program Files.",
        ).pack(fill="x")

        def update() -> None:
            if self._text("detector") == "ctd" and not ctd_is_installed():
                ctd_notice.pack(fill="x", pady=(14, 0))
            else:
                ctd_notice.pack_forget()
            needed = self._tesseract_needed()
            available = self._tesseract_available()
            if available:
                ui.set_badge(self.tesseract_badge, "Found", "ok")
            elif needed:
                ui.set_badge(self.tesseract_badge, "Missing", "error")
            else:
                ui.set_badge(self.tesseract_badge, "Not found", "neutral")

        self._context_updaters.append(update)

    def _make_tesseract_badge(self, header: tk.Frame) -> tk.Label:
        self.tesseract_badge = ui.badge(header, self.fonts, "Checking", "neutral")
        return self.tesseract_badge

    def _build_vision_page(self, parent: tk.Frame) -> None:
        ui.Callout(
            parent,
            self.fonts,
            text="Vision passes run a local Qwen vision model with its mmproj projector from .models/qwen. "
            "They add noticeable time to every page.",
            level="info",
        ).pack(fill="x", pady=(0, 16))

        card = self._card(parent, "Vision passes")
        facts_row, _ = ui.switch_row(
            card.body,
            self.fonts,
            self.assets,
            title="Visual facts",
            description="Before translating, describe each page so the translator has visual context.",
            variable=self.vars["vision_facts"],
        )
        facts_row.pack(fill="x")
        ui.separator(card.body, pady=14)
        repair_row, _ = ui.switch_row(
            card.body,
            self.fonts,
            self.assets,
            title="Vision repair",
            description="After translating, re-check flagged blocks against the page image and repair them.",
            variable=self.vars["vision"],
        )
        repair_row.pack(fill="x")

        card = self._card(parent, "Repair settings", "Applies when vision repair is on.")
        mode_field, mode_control = self._choice_field(card.body, title="Page artifact", key="vision_mode", choices=VISION_MODES)
        ui.separator(card.body, pady=18)
        trigger_field, trigger_control = self._choice_field(card.body, title="Blocks to repair", key="vision_trigger", choices=VISION_TRIGGERS)

        def update() -> None:
            enabled = bool(self.vars["vision"].get())
            for control in (mode_control, trigger_control):
                control.set_enabled(enabled)
            for field in (mode_field, trigger_field):
                field.label.configure(fg=ui.TEXT if enabled else ui.SUBTLE)

        self._context_updaters.append(update)

    def _build_typesetting_page(self, parent: tk.Frame) -> None:
        card = self._card(parent, "Font")
        ui.PathField(
            card.body,
            self.fonts,
            label="Font file",
            variable=self.vars["font"],
            buttons=(("Browse…", lambda: self._browse_file("font", "Choose text font", (("Font files", "*.ttf *.otf"), ("All files", "*.*")))),),
            hint="TrueType or OpenType font for the English text. Leave empty to use the default font.",
            optional=True,
        ).pack(fill="x")

        card = self._card(
            parent,
            "Sizing",
            "How English text is fitted into the space left by the original lettering.",
            trailing=lambda header: ui.LinkLabel(header, self.fonts, text="Reset", command=self._reset_typesetting, bg=ui.SURFACE),
        )
        ui.SliderField(
            card.body,
            self.fonts,
            self.assets,
            label="Starting font size",
            description="Size the renderer starts from for each block of English text.",
            variable=self.vars["font_size"],
            minimum=10,
            maximum=72,
            step=1,
            formatter=lambda value: f"{int(value)} pt",
        ).pack(fill="x")
        ui.separator(card.body, pady=18)
        ui.SliderField(
            card.body,
            self.fonts,
            self.assets,
            label="Box expansion",
            description="How much detected text boxes may grow before the English text is drawn.",
            variable=self.vars["render_expand"],
            minimum=1.0,
            maximum=4.0,
            step=0.1,
            formatter=lambda value: f"{value:.1f}×",
        ).pack(fill="x")

    def _build_setup_page(self, parent: tk.Frame) -> None:
        card = self._card(parent, "Components", "Each component downloads once. Large models can take a while on the first install.")
        self.ctd_badge: tk.Label | None = None
        for index, component in enumerate(COMPONENTS):
            if index:
                ui.separator(card.body, pady=12)
            row = ui.SettingRow(card.body, self.fonts, title=component.title, description=component.description)
            row.pack(fill="x")
            meta = tk.Frame(row, bg=ui.SURFACE)
            meta.grid(row=2, column=0, sticky="w", pady=(6, 0))
            if component.key == "ctd":
                self.ctd_badge = ui.badge(meta, self.fonts, "Checking", "neutral")
                self.ctd_badge.pack(side="left", padx=(0, 8))
            tk.Label(meta, text=component.size_note, font=self.fonts.small, bg=ui.SURFACE, fg=ui.SUBTLE).pack(side="left")
            button = ttk.Button(row, text="Install", style="Small.TButton", command=lambda key=component.key: self._install(key))
            row.place_control(button)
            button.grid_configure(rowspan=3, sticky="e")
            self.install_buttons[component.key] = button

        card = self._card(parent, "Tesseract OCR", "Optional system OCR used by the Tesseract detector and OCR options.")
        row = ui.SettingRow(card.body, self.fonts, title="Tesseract", description="Install the Windows build with Japanese language data.")
        row.pack(fill="x")
        row.place_control(ttk.Button(row, text="Setup guide", style="Small.TButton", command=self._show_tesseract_help))

        card = self._card(parent, "Files")
        for title, path, opener in (
            ("Debug log", self.log_path, self._open_log_file),
            ("Settings file", SETTINGS_PATH, lambda: self._open_path(SETTINGS_PATH.parent)),
        ):
            row = ui.SettingRow(card.body, self.fonts, title=title, description=str(path) if path else "Not available")
            row.pack(fill="x", pady=(0, 12))
            row.place_control(ttk.Button(row, text="Open", style="Small.TButton", command=opener))
        ui.separator(card.body, pady=(0, 12))
        row = ui.SettingRow(card.body, self.fonts, title="Restore defaults", description="Reset every option except your input, output, and Tesseract paths.")
        row.pack(fill="x")
        row.place_control(ttk.Button(row, text="Restore…", style="Small.TButton", command=self._restore_defaults))

        def update() -> None:
            installed = ctd_is_installed()
            if self.ctd_badge is not None:
                ui.set_badge(self.ctd_badge, "Installed" if installed else "Not installed", "ok" if installed else "warning")
            self.install_buttons["ctd"].configure(text="Reinstall" if installed else "Install")

        self._context_updaters.append(update)

    def _build_run_panel(self) -> tk.Frame:
        panel = tk.Frame(self, bg=ui.SURFACE, width=self._px(392))
        panel.pack_propagate(False)
        body = tk.Frame(panel, bg=ui.SURFACE, padx=self._px(22), pady=self._px(24))
        body.pack(fill="both", expand=True)

        header = tk.Frame(body, bg=ui.SURFACE)
        header.pack(fill="x")
        tk.Label(header, text="Run", font=self.fonts.heading, bg=ui.SURFACE, fg=ui.TEXT).pack(side="left")
        self.state_badge = ui.badge(header, self.fonts, "Ready", "neutral")
        self.state_badge.pack(side="right")

        tk.Label(body, text="CHECKLIST", font=self.fonts.caption, bg=ui.SURFACE, fg=ui.SUBTLE).pack(anchor="w", pady=(18, 6))
        self.checklist = tk.Frame(body, bg=ui.SURFACE)
        self.checklist.pack(fill="x")
        self.pipeline_summary = ui.WrapLabel(body, font=self.fonts.small, bg=ui.SURFACE, fg=ui.MUTED, wraplength=self._px(340))
        self.pipeline_summary.pack(fill="x", pady=(10, 0))

        progress = tk.Frame(body, bg=ui.SURFACE_ALT, highlightthickness=1, highlightbackground=ui.BORDER, padx=14, pady=12)
        progress.pack(fill="x", pady=(18, 14))
        top = tk.Frame(progress, bg=ui.SURFACE_ALT)
        top.pack(fill="x")
        self.progress_stage_label = tk.Label(top, font=self.fonts.body_strong, bg=ui.SURFACE_ALT, fg=ui.TEXT, anchor="w")
        self.progress_stage_label.pack(side="left")
        self.progress_percent = tk.Label(top, font=self.fonts.small_strong, bg=ui.SURFACE_ALT, fg=ui.MUTED)
        self.progress_percent.pack(side="right")
        self.progressbar = ttk.Progressbar(progress, style="Accent.Horizontal.TProgressbar", mode="determinate", maximum=100)
        self.progressbar.pack(fill="x", pady=(10, 8))
        self.progress_detail_label = tk.Label(progress, font=self.fonts.small, bg=ui.SURFACE_ALT, fg=ui.MUTED, anchor="w")
        self.progress_detail_label.pack(fill="x")

        self.start_button = ttk.Button(body, text="Start translation", style="Accent.TButton", command=self._start)
        self.start_button.pack(fill="x")
        self.start_hint = tk.Label(body, font=self.fonts.small, bg=ui.SURFACE, fg=ui.MUTED)
        self.start_hint.pack(pady=(6, 0))

        actions = tk.Frame(body, bg=ui.SURFACE)
        actions.pack(fill="x", pady=(10, 0))
        actions.columnconfigure(0, weight=1, uniform="actions")
        actions.columnconfigure(1, weight=1, uniform="actions")
        self.open_output_button = ttk.Button(actions, text="Open output", style="Small.TButton", command=self._open_output)
        self.open_output_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="Open debug log", style="Small.TButton", command=self._open_log_file).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        ui.separator(body, pady=(20, 10))
        log_header = tk.Frame(body, bg=ui.SURFACE)
        log_header.pack(fill="x", pady=(0, 8))
        tk.Label(log_header, text="Activity", font=self.fonts.body_strong, bg=ui.SURFACE, fg=ui.TEXT).pack(side="left")
        ttk.Button(log_header, text="Clear", style="Ghost.TButton", command=self._clear_log).pack(side="right")
        ttk.Button(log_header, text="Copy", style="Ghost.TButton", command=self._copy_log).pack(side="right", padx=(0, 2))

        log_frame = tk.Frame(body, bg=ui.LOG_BG)
        log_frame.pack(fill="both", expand=True)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            bg=ui.LOG_BG,
            fg=ui.LOG_TEXT,
            insertbackground=ui.LOG_TEXT,
            selectbackground="#1E293B",
            selectforeground=ui.LOG_TEXT,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=12,
            pady=10,
            wrap="word",
            font=self.fonts.mono,
            spacing1=1,
            spacing3=3,
            state="disabled",
            cursor="arrow",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview, style="Log.Vertical.TScrollbar")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.tag_configure("line", lmargin2=self.fonts.mono.measure("00:00:00  "))
        self.log.tag_configure("time", foreground=ui.LOG_MUTED)
        self.log.tag_configure("info", foreground=ui.LOG_TEXT)
        self.log.tag_configure("muted", foreground=ui.LOG_MUTED)
        self.log.tag_configure("stage", foreground=ui.LOG_ACCENT)
        self.log.tag_configure("success", foreground=ui.LOG_SUCCESS)
        self.log.tag_configure("warning", foreground=ui.LOG_WARNING)
        self.log.tag_configure("error", foreground=ui.LOG_DANGER)
        return panel

    # ------------------------------------------------------------------ navigation & context

    def _show_page(self, key: str) -> None:
        if key not in self.pages:
            return
        self.current_page = key
        self.pages[key].tkraise()
        self.pages[key].scroll_to_top()
        for page_key, item in self.nav_items.items():
            item.set_active(page_key == key)

    def _bind_events(self) -> None:
        traced = ("input", "output", "detector", "ocr_engine", "tesseract", "translator", "overwrite", "vision", "vision_facts", "erase_mode")
        for key in traced:
            self.vars[key].trace_add("write", lambda *_args: self._schedule_context_refresh())
        self.vars["input"].trace_add("write", lambda *_args: self._maybe_update_auto_output())
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", lambda event: self._on_mousewheel(event, 120), add="+")
        self.bind_all("<Button-5>", lambda event: self._on_mousewheel(event, -120), add="+")
        self.bind("<Control-Return>", lambda _e: self._start())
        self.bind("<Control-o>", lambda _e: self._choose_input())
        for index, (key, *_rest) in enumerate(PAGES, start=1):
            self.bind(f"<Control-Key-{index}>", lambda _e, k=key: self._show_page(k))

    def _on_mousewheel(self, event: tk.Event, delta: int | None = None) -> None:
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            return
        while widget is not None:
            if isinstance(widget, tk.Text):
                return
            if isinstance(widget, ui.ScrollFrame):
                widget.scroll_wheel(delta if delta is not None else event.delta)
                return
            widget = widget.master

    def _schedule_context_refresh(self) -> None:
        if self._checks_after is not None:
            self.after_cancel(self._checks_after)
        self._checks_after = self.after(200, self._refresh_all)

    def _refresh_all(self) -> None:
        self._checks_after = None
        self._refresh_context()
        self._refresh_checks()

    def _refresh_context(self) -> None:
        for update in self._context_updaters:
            update()

    def _tesseract_needed(self) -> bool:
        return self._text("detector") == "tesseract" or self._text("ocr_engine") == "tesseract"

    def _tesseract_available(self) -> bool:
        value = self._text("tesseract")
        if not value:
            return False
        return Path(value).is_file() or shutil.which(value) is not None

    def _image_count(self, path: Path) -> int:
        key = str(path)
        if key not in self._image_counts:
            try:
                self._image_counts[key] = len(iter_images(path, recursive=True))
            except OSError:
                logger.debug("Image count failed for %s", path, exc_info=True)
                self._image_counts[key] = 0
        return self._image_counts[key]

    def _collect_checks(self) -> list[Check]:
        checks: list[Check] = []
        input_text = self._text("input")
        if not input_text:
            checks.append(Check("error", "Choose the pages to translate.", ("Browse", self._choose_input)))
        elif not Path(input_text).exists():
            checks.append(Check("error", "The input path does not exist.", ("Browse", self._choose_input)))
        else:
            count = self._image_count(Path(input_text))
            if count == 0:
                checks.append(Check("error", "No supported images found in the input.", ("Browse", self._choose_input)))
            else:
                name = Path(input_text).name or input_text
                checks.append(Check("ok", f"{count} page{'s' if count != 1 else ''} found in {name}"))

        output_text = self._text("output")
        if not output_text:
            checks.append(Check("error", "Choose an output folder.", ("Browse", self._choose_output)))
        else:
            output_path = Path(output_text)
            same_as_input = False
            if input_text:
                try:
                    same_as_input = output_path.resolve() == Path(input_text).resolve()
                except OSError:
                    same_as_input = False
            if same_as_input:
                checks.append(Check("warning", "Output is the same folder as the input.", ("Change", self._choose_output)))
            else:
                checks.append(Check("ok", f"Saving to {output_path.name or output_text}"))
            if not self.vars["overwrite"].get() and output_path.is_dir() and any(output_path.iterdir()):
                checks.append(Check("info", "Pages already in the output will be skipped.", ("Options", lambda: self._show_page("project"))))

        if self._text("detector") == "ctd":
            if ctd_is_installed():
                checks.append(Check("ok", "Comic Text Detector installed"))
            else:
                checks.append(Check("error", "Comic Text Detector is not installed.", ("Install", lambda: self._install("ctd"))))

        if self._tesseract_needed():
            if self._tesseract_available():
                checks.append(Check("ok", "Tesseract OCR found"))
            else:
                checks.append(Check("error", "Tesseract OCR was not found.", ("Fix", lambda: self._show_page("detection"))))
        return checks

    def _refresh_checks(self) -> None:
        checks = self._collect_checks()
        signature = [check.key() for check in checks]
        if signature != self._checks_signature:
            self._checks_signature = signature
            for child in self.checklist.winfo_children():
                child.destroy()
            for check in checks:
                row = tk.Frame(self.checklist, bg=ui.SURFACE)
                row.pack(fill="x", pady=3)
                icon = self.assets.status(check.level)
                if icon is not None:
                    tk.Label(row, image=icon, bg=ui.SURFACE, bd=0).pack(side="left", anchor="n", pady=(1, 0))
                else:
                    color = ui.LEVEL_COLORS.get(check.level, ui.LEVEL_COLORS["neutral"])[0]
                    tk.Label(row, text="●", font=self.fonts.small, bg=ui.SURFACE, fg=color).pack(side="left", anchor="n")
                if check.action:
                    ui.LinkLabel(row, self.fonts, text=check.action[0], command=check.action[1], bg=ui.SURFACE).pack(
                        side="right", anchor="n", padx=(8, 0)
                    )
                ui.WrapLabel(row, text=check.text, font=self.fonts.small, bg=ui.SURFACE, fg=ui.TEXT_SECONDARY, wraplength=self._px(260)).pack(
                    side="left", fill="x", expand=True, padx=(10, 0)
                )
        self._blocked = any(check.level == "error" for check in checks)

        summary = [
            choice_label(TRANSLATORS, self._text("translator")),
            f"{choice_label(DETECTORS, self._text('detector'))} detector",
            f"{choice_label(OCR_ENGINES, self._text('ocr_engine'))} OCR",
            choice_label(ERASE_MODES, self._text("erase_mode")),
        ]
        if self.vars["vision_facts"].get() or self.vars["vision"].get():
            summary.append("Vision")
        self.pipeline_summary.configure(text="  ·  ".join(summary))

        self.nav_items["project"].set_attention("error" if any(c.level == "error" and c.action and c.action[0] == "Browse" for c in checks) else None)
        ctd_missing = self._text("detector") == "ctd" and not ctd_is_installed()
        self.nav_items["setup"].set_attention("warning" if ctd_missing else None)
        tesseract_missing = self._tesseract_needed() and not self._tesseract_available()
        self.nav_items["detection"].set_attention("error" if tesseract_missing else None)
        self._update_controls()

    def _update_controls(self) -> None:
        running = self._is_running()
        if running:
            self.start_button.configure(text="Translating…" if self._job == "translate" else "Working…")
            self.start_button.state(["disabled"])
            self.start_hint.configure(text="Settings changes apply to the next run.", fg=ui.MUTED)
        else:
            self.start_button.configure(text="Start translation")
            self.start_button.state(["disabled"] if self._blocked else ["!disabled"])
            if self._blocked:
                self.start_hint.configure(text="Resolve the checklist items to start.", fg=ui.WARNING)
            else:
                self.start_hint.configure(text="Ready  ·  Ctrl+Enter", fg=ui.MUTED)
        for button in self.install_buttons.values():
            button.state(["disabled"] if running else ["!disabled"])
        output = self._path("output")
        self.open_output_button.state(["!disabled"] if output and output.exists() else ["disabled"])

    # ------------------------------------------------------------------ file choosers

    def _initial_dir(self, key: str) -> str | None:
        value = self._path(key)
        if value is None:
            return None
        candidate = value if value.is_dir() else value.parent
        return str(candidate) if candidate.exists() else None

    def _browse_file(self, key: str, title: str, filetypes: tuple[tuple[str, str], ...]) -> None:
        selected = filedialog.askopenfilename(parent=self, title=title, filetypes=filetypes, initialdir=self._initial_dir(key))
        if selected:
            self.vars[key].set(selected)
            logger.info("Selected %s: %s", key, selected)
        else:
            logger.debug("%s selection cancelled", key)

    def _browse_dir(self, key: str, title: str) -> None:
        selected = filedialog.askdirectory(parent=self, title=title, initialdir=self._initial_dir(key))
        if selected:
            self.vars[key].set(selected)
            logger.info("Selected %s: %s", key, selected)
        else:
            logger.debug("%s selection cancelled", key)

    def _choose_input(self) -> None:
        self._browse_dir("input", "Choose folder with manga images")

    def _choose_input_file(self) -> None:
        self._browse_file(
            "input",
            "Choose a manga page",
            (("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"), ("All files", "*.*")),
        )

    def _maybe_update_auto_output(self) -> None:
        input_text = self._text("input")
        output_text = self._text("output")
        if not input_text or (output_text and output_text != self._auto_output):
            return
        source = Path(input_text)
        folder = source.parent if source.suffix else source
        if not folder.name:
            return
        output = str(folder.with_name(f"{folder.name}-translated"))
        self._auto_output = output
        self.vars["output"].set(output)
        logger.info("Auto-selected output folder: %s", output)

    def _choose_output(self) -> None:
        self._browse_dir("output", "Choose output folder")

    def _choose_tesseract(self) -> None:
        self._browse_file(
            "tesseract",
            "Choose tesseract.exe",
            (("Tesseract executable", "tesseract.exe"), ("Executables", "*.exe"), ("All files", "*.*")),
        )

    def _prefill_tesseract_path(self) -> None:
        if self._tesseract_available():
            return
        found = find_tesseract()
        if found:
            self.vars["tesseract"].set(found)
            self.log_message(f"Tesseract found: {found}", "muted")
        elif self._tesseract_needed():
            self.log_message("Tesseract was not found. Open Detection & OCR to set its path.", "warning")

    def _show_tesseract_help(self) -> None:
        logger.info("Showing Tesseract help dialog")
        messagebox.showinfo("Tesseract setup", tesseract_missing_message(), parent=self)

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showinfo("Not found", f"{path} does not exist yet.", parent=self)
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # noqa: S606 - opens a local folder or file chosen by the user.
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            logger.warning("Could not open %s", path, exc_info=True)
            messagebox.showerror("Could not open", str(exc), parent=self)

    def _open_output(self) -> None:
        output = self._path("output")
        if output is not None:
            self._open_path(output)

    def _open_log_file(self) -> None:
        if self.log_path is not None:
            self._open_path(self.log_path)

    # ------------------------------------------------------------------ jobs

    def _is_running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _start(self) -> None:
        if self._is_running():
            logger.warning("Start ignored because a worker is already running")
            return
        self._refresh_checks()
        if self._blocked:
            logger.warning("Start blocked by checklist: %s", self._checks_signature)
            return

        input_path = Path(self._text("input"))
        output_path = Path(self._text("output"))
        try:
            font_size = int(self.vars["font_size"].get())
            render_expand = float(self.vars["render_expand"].get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid typesetting value", "Check the font size and box expansion values.", parent=self)
            return

        config = PipelineConfig(
            detector=self._text("detector"),
            ocr_engine=self._text("ocr_engine"),
            translator=self._text("translator"),
            qwen_mode=self._text("qwen_mode"),
            erase_mode=self._text("erase_mode"),
            tesseract_cmd=self._text("tesseract") or None,
            render_expand=render_expand,
            font_path=self._path("font"),
            glossary_path=self._path("glossary"),
            qwen_model_path=self._path("qwen_model"),
            qwen_fallback_model_path=self._path("qwen_fallback"),
            qwen_critic_model_path=self._path("qwen_critic"),
            cat_model_name=self._text("cat_model") or None,
            hy_mt2_model_name=self._text("hy_mt2_model") or None,
            vision_enabled=bool(self.vars["vision"].get()),
            vision_facts_enabled=bool(self.vars["vision_facts"].get()),
            vision_mode=self._text("vision_mode"),
            vision_trigger=self._text("vision_trigger"),
            base_font_size=font_size,
            work_dir=self._path("work_dir"),
            debug=bool(self.vars["debug"].get()),
            overwrite=bool(self.vars["overwrite"].get()),
            resume=bool(self.vars["resume"].get()),
        )
        self._save_settings()
        logger.info("Starting translation from GUI")
        logger.debug("GUI pipeline config: %s", config)
        self._run_worker("translate", "Translating", lambda: self._translate(input_path, output_path, config))

    def _install(self, key: str) -> None:
        if self._is_running():
            logger.warning("%s install ignored because a worker is already running", key)
            return
        component = next(c for c in COMPONENTS if c.key == key)
        workers = {
            "ctd": self._install_ctd_worker,
            "opus": self._install_opus_worker,
            "hy-mt2": self._install_hy_mt2_worker,
            "cat": self._install_cat_worker,
            "madlad": self._install_madlad_worker,
            "argos": self._install_argos_worker,
        }
        logger.info("Starting %s install from GUI", key)
        self._run_worker(f"install:{key}", component.job_label, workers[key])

    def _run_worker(self, job: str, label: str, target: Callable[[], None]) -> None:
        self._job = job
        self._job_label = label
        self._job_started = time.monotonic()
        self._progress_stage = ""
        self.log_message(f"{label}…", "stage")
        logger.info("Worker starting: %s", label)
        self._set_progress_style("Accent")
        self.progressbar.configure(mode="indeterminate")
        self.progressbar.start(14)
        self.progress_stage_label.configure(text=f"{label}…")
        self.progress_percent.configure(text="")
        self._progress_detail = "Starting up"
        self._set_run_state("running")
        self.worker = threading.Thread(target=self._worker_wrapper, args=(target,), daemon=True)
        self.worker.start()
        self._update_controls()
        self.after(1000, self._tick)

    def _worker_wrapper(self, target: Callable[[], None]) -> None:
        writer = _QueueWriter(self.events.put)
        try:
            with redirect_stdout(writer), self._progress_tracking():
                target()
        except (Exception, SystemExit) as exc:
            logger.exception("Worker failed")
            writer.flush()
            self.events.put(("line", traceback.format_exc().rstrip(), "error"))
            summary = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            self.events.put(("done", False, summary))
        else:
            writer.flush()
            logger.info("Worker finished successfully")
            self.events.put(("done", True, ""))

    @contextmanager
    def _progress_tracking(self):
        from . import pipeline, translation_review_pass

        modules = (pipeline, translation_review_pass)
        originals = [module.tqdm for module in modules]
        for module, original in zip(modules, originals):
            module.tqdm = self._tracked_tqdm(original)
        try:
            yield
        finally:
            for module, original in zip(modules, originals):
                module.tqdm = original

    def _tracked_tqdm(self, original: Callable):
        post = self.events.put

        def tracked(iterable, *args, **kwargs):
            stage = str(kwargs.get("desc") or "Working")
            unit = str(kwargs.get("unit") or "item")
            total = kwargs.get("total")
            if total is None:
                try:
                    total = len(iterable)
                except TypeError:
                    total = None
            source = original(iterable, *args, **kwargs) if sys.stderr is not None else iterable
            index = 0
            for index, item in enumerate(source, start=1):
                post(("progress", stage, index - 1, total, unit, index))
                yield item
            post(("progress", stage, total if total is not None else index, total, unit, None))

        return tracked

    def _translate(self, input_path: Path, output_path: Path, config: PipelineConfig) -> None:
        from .dependencies import ensure_runtime_dependencies
        from .pipeline import process_folder

        self.events.put(("line", "Checking Python dependencies…", "muted"))
        logger.info("Checking runtime dependencies before translation")
        ensure_runtime_dependencies()
        self.events.put(("line", f"Input: {input_path}", "muted"))
        self.events.put(("line", f"Output: {output_path}", "muted"))
        logger.info("Processing input=%s output=%s", input_path, output_path)
        process_folder(input_path, output_path, config)

    def _install_argos_worker(self) -> None:
        from .dependencies import ensure_python_package
        from .install_argos import main

        self.events.put(("line", "Checking argostranslate package…", "muted"))
        logger.info("Checking argostranslate before model install")
        ensure_python_package("argostranslate", "argostranslate")
        main(["ja", "en"])

    def _install_ctd_worker(self) -> None:
        from .dependencies import ensure_python_package
        from .install_ctd import main

        self.events.put(("line", "Checking CTD Python dependencies…", "muted"))
        logger.info("Checking CTD dependencies before detector install")
        for import_name, package_name in (
            ("shapely", "shapely"),
            ("pyclipper", "pyclipper"),
            ("torchsummary", "torchsummary"),
        ):
            ensure_python_package(import_name, package_name)
        main([])

    def _install_madlad_worker(self) -> None:
        from .dependencies import ensure_python_package
        from .install_madlad import main

        self.events.put(("line", "Checking MADLAD Python dependencies…", "muted"))
        logger.info("Checking MADLAD dependencies before translator install")
        for import_name, package_name in (
            ("transformers", "transformers"),
            ("sentencepiece", "sentencepiece"),
            ("accelerate", "accelerate"),
            ("huggingface_hub", "huggingface-hub"),
        ):
            ensure_python_package(import_name, package_name)
        self.events.put(("line", "Downloading MADLAD model. This is large and can take a while the first time.", "info"))
        main([])

    def _install_opus_worker(self) -> None:
        from .dependencies import ensure_python_package
        from .install_opus import main

        self.events.put(("line", "Checking OPUS Python dependencies…", "muted"))
        logger.info("Checking OPUS dependencies before translator install")
        for import_name, package_name in (
            ("transformers", "transformers"),
            ("sentencepiece", "sentencepiece"),
            ("huggingface_hub", "huggingface-hub"),
        ):
            ensure_python_package(import_name, package_name)
        main([])

    def _install_cat_worker(self) -> None:
        from .dependencies import ensure_python_package
        from .install_cat import main

        self.events.put(("line", "Checking CAT Python dependencies…", "muted"))
        logger.info("Checking CAT dependencies before translator install")
        ensure_python_package("huggingface_hub", "huggingface-hub")
        self.events.put(("line", "Downloading CAT model. This is large and can take a while the first time.", "info"))
        main([])

    def _install_hy_mt2_worker(self) -> None:
        from .dependencies import ensure_python_package
        from .install_hy_mt2 import main

        self.events.put(("line", "Checking Hy-MT2 Python dependencies…", "muted"))
        logger.info("Checking Hy-MT2 dependencies before translator install")
        for import_name, package_name in (
            ("transformers", "transformers"),
            ("accelerate", "accelerate"),
            ("huggingface_hub", "huggingface-hub"),
        ):
            ensure_python_package(import_name, package_name)
        self.events.put(("line", "Downloading Hy-MT2 model. This is large and can take a while the first time.", "info"))
        main([])

    # ------------------------------------------------------------------ events & progress

    def _install_log_handler(self) -> None:
        handler = _QueueLogHandler(self.events.put)
        package_logger = logging.getLogger(__package__ or "manga_local_translator")
        package_logger.addHandler(handler)
        self.bind("<Destroy>", lambda event: package_logger.removeHandler(handler) if event.widget is self else None, add="+")

    def _drain_events(self) -> None:
        try:
            for _ in range(500):
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    _kind, level, message, created = event
                    tag = "error" if level >= logging.ERROR else "warning" if level >= logging.WARNING else "info"
                    self.log_message(shorten(message, limit=LOG_LINE_LIMIT), tag, when=created)
                elif kind == "line":
                    self.log_message(event[1], event[2])
                elif kind == "progress":
                    self._on_progress(*event[1:])
                elif kind == "done":
                    self._on_job_done(event[1], event[2])
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _set_progress_style(self, name: str) -> None:
        self.progressbar.configure(style=f"{name}.Horizontal.TProgressbar")

    def _set_run_state(self, state: str) -> None:
        badge_text, level, dot = {
            "ready": ("Ready", "neutral", ui.LOG_SUCCESS),
            "running": ("Running", "info", ui.LOG_ACCENT),
            "done": ("Complete", "ok", ui.LOG_SUCCESS),
            "failed": ("Failed", "error", ui.LOG_DANGER),
        }[state]
        ui.set_badge(self.state_badge, badge_text, level)
        self.sidebar_dot.configure(fg=dot)
        self.sidebar_status.configure(text={"running": self._job_label or "Running", "failed": "Last run failed"}.get(state, "Ready"))

    def _set_idle_progress(self) -> None:
        self.progressbar.stop()
        self.progressbar.configure(mode="determinate", value=0)
        self._set_progress_style("Accent")
        self.progress_stage_label.configure(text="Ready to translate")
        self.progress_percent.configure(text="")
        self.progress_detail_label.configure(text="Progress for each stage appears here.")

    def _on_progress(self, stage: str, completed: int, total: int | None, unit: str, current: int | None) -> None:
        if stage != self._progress_stage:
            self._progress_stage = stage
            self.log_message(f"▸ {stage}", "stage")
        self.progress_stage_label.configure(text=stage)
        if total:
            if str(self.progressbar.cget("mode")) != "determinate":
                self.progressbar.stop()
                self.progressbar.configure(mode="determinate")
            percent = max(0.0, min(100.0, completed / total * 100))
            self.progressbar.configure(value=percent)
            self.progress_percent.configure(text=f"{int(percent)}%")
        unit_label = unit.capitalize()
        if current is not None:
            self._progress_detail = f"{unit_label} {current} of {total}" if total else f"{unit_label} {current}"
        else:
            self._progress_detail = f"{completed} {unit}{'s' if completed != 1 else ''} done"
        self._render_progress_detail()

    def _render_progress_detail(self) -> None:
        if self._is_running():
            elapsed = format_duration(time.monotonic() - self._job_started)
            self.progress_detail_label.configure(text=f"{self._progress_detail}  ·  {elapsed} elapsed")

    def _tick(self) -> None:
        if self._is_running():
            self._render_progress_detail()
            self.after(1000, self._tick)

    def _on_job_done(self, ok: bool, summary: str) -> None:
        elapsed = format_duration(time.monotonic() - self._job_started)
        job, label = self._job, self._job_label
        self.worker = None
        self._job = None
        self.progressbar.stop()
        self.progressbar.configure(mode="determinate", value=100)
        self.progress_percent.configure(text="")
        if ok:
            self._set_progress_style("Success")
            title = "Translation complete" if job == "translate" else f"{label.replace('Installing', '').strip()} installed"
            self.progress_stage_label.configure(text=title)
            self.progress_detail_label.configure(text=f"Finished in {elapsed}")
            self.log_message(f"{title} in {elapsed}.", "success")
            self._set_run_state("done")
        else:
            self._set_progress_style("Danger")
            self.progress_stage_label.configure(text="Translation failed" if job == "translate" else f"{label} failed")
            self.progress_detail_label.configure(text=summary or "See the activity log for details.")
            self.log_message(f"Failed after {elapsed}: {summary}", "error")
            self._set_run_state("failed")
        self._image_counts.clear()
        self._checks_signature = None
        self._refresh_context()
        self._refresh_checks()

    # ------------------------------------------------------------------ activity log

    def log_message(self, message: str, tag: str = "info", *, when: float | None = None) -> None:
        stamp = datetime.fromtimestamp(when if when is not None else time.time()).strftime("%H:%M:%S")
        self.log.configure(state="normal")
        at_end = self.log.yview()[1] >= 0.999
        self.log.insert(tk.END, f"{stamp}  ", ("time", "line"), f"{message.rstrip()}\n", (tag, "line"))
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self.log.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
        if at_end:
            self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")

    def _copy_log(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.log.get("1.0", "end-1c"))
        self.log_message("Activity log copied to the clipboard.", "muted")

    # ------------------------------------------------------------------ shutdown

    def _on_close(self) -> None:
        if self._is_running() and not messagebox.askyesno(
            "Job still running",
            f"{self._job_label} is still running.\n\nQuit anyway? Work on the current page will be lost.",
            icon="warning",
            parent=self,
        ):
            return
        self._save_settings()
        self.destroy()


def _prepare_windows_process() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        logger.debug("DPI awareness could not be enabled", exc_info=True)
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MangaLocalTranslator.GUI")
    except (AttributeError, OSError):
        logger.debug("App user model id could not be set", exc_info=True)


def main() -> int:
    log_path = configure_logging(reset=True)
    logger.info("GUI main started")
    _prepare_windows_process()
    app = MangaTranslatorGui(log_path=log_path)
    app.log_message(f"Debug log: {log_path}", "muted")
    app.mainloop()
    logger.info("GUI main exited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
