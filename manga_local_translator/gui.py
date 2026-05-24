from __future__ import annotations

import queue
import logging
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import PipelineConfig
from .logging_utils import configure_logging
from .tesseract_utils import find_tesseract, tesseract_missing_message

logger = logging.getLogger(__name__)


class MangaTranslatorGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Local Manga Translator")
        self.geometry("980x680")
        self.minsize(900, 620)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.detector_var = tk.StringVar(value="ctd")
        self.ocr_engine_var = tk.StringVar(value="manga-ocr")
        self.erase_mode_var = tk.StringVar(value="white")
        self.translator_var = tk.StringVar(value="opus")
        self.qwen_mode_var = tk.StringVar(value="block")
        self.qwen_model_var = tk.StringVar()
        self.qwen_fallback_var = tk.StringVar()
        self.vision_var = tk.BooleanVar(value=False)
        self.vision_facts_var = tk.BooleanVar(value=False)
        self.vision_mode_var = tk.StringVar(value="numbered_page")
        self.vision_trigger_var = tk.StringVar(value="suspicious")
        self.glossary_var = tk.StringVar()
        self.font_var = tk.StringVar()
        self.font_size_var = tk.IntVar(value=28)
        self.render_expand_var = tk.DoubleVar(value=2.2)
        self.work_dir_var = tk.StringVar()
        self.tesseract_var = tk.StringVar()
        self.debug_var = tk.BooleanVar(value=True)
        self.overwrite_var = tk.BooleanVar(value=True)
        self.resume_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        self._build_ui()
        self._prefill_tesseract_path()
        self.after(100, self._drain_log_queue)
        logger.info("GUI initialized")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(18, weight=1)

        ttk.Label(root, text="Image folder").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(root, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(root, text="Browse", command=self._choose_input).grid(row=0, column=2, pady=(0, 8))

        ttk.Label(root, text="Output folder").grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(root, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(root, text="Browse", command=self._choose_output).grid(row=1, column=2, pady=(0, 8))

        ttk.Label(root, text="Text detector").grid(row=2, column=0, sticky="w", pady=(0, 8))
        ttk.Combobox(
            root,
            textvariable=self.detector_var,
            values=("ctd", "tesseract", "visual"),
            state="readonly",
            width=16,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=(0, 8))

        ttk.Label(root, text="OCR engine").grid(row=3, column=0, sticky="w", pady=(0, 8))
        ttk.Combobox(
            root,
            textvariable=self.ocr_engine_var,
            values=("manga-ocr", "tesseract"),
            state="readonly",
            width=16,
        ).grid(row=3, column=1, sticky="w", padx=8, pady=(0, 8))

        ttk.Label(root, text="Translator").grid(row=4, column=0, sticky="w", pady=(0, 8))
        ttk.Combobox(
            root,
            textvariable=self.translator_var,
            values=("opus", "qwen", "madlad", "argos", "none"),
            state="readonly",
            width=16,
        ).grid(row=4, column=1, sticky="w", padx=8, pady=(0, 8))

        ttk.Label(root, text="Qwen mode").grid(row=5, column=0, sticky="w", pady=(0, 8))
        ttk.Combobox(
            root,
            textvariable=self.qwen_mode_var,
            values=("block", "page"),
            state="readonly",
            width=16,
        ).grid(row=5, column=1, sticky="w", padx=8, pady=(0, 8))

        ttk.Label(root, text="Erase mode").grid(row=6, column=0, sticky="w", pady=(0, 8))
        ttk.Combobox(
            root,
            textvariable=self.erase_mode_var,
            values=("white", "inpaint"),
            state="readonly",
            width=16,
        ).grid(row=6, column=1, sticky="w", padx=8, pady=(0, 8))

        ttk.Label(root, text="Qwen primary model").grid(row=7, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(root, textvariable=self.qwen_model_var).grid(row=7, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(root, text="Browse", command=self._choose_qwen_model).grid(row=7, column=2, pady=(0, 8))

        ttk.Label(root, text="Qwen fallback model").grid(row=8, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(root, textvariable=self.qwen_fallback_var).grid(row=8, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(root, text="Browse", command=self._choose_qwen_fallback).grid(row=8, column=2, pady=(0, 8))

        ttk.Label(root, text="Vision repair").grid(row=9, column=0, sticky="w", pady=(0, 8))
        vision_options = ttk.Frame(root)
        vision_options.grid(row=9, column=1, sticky="w", padx=8, pady=(0, 8))
        ttk.Checkbutton(vision_options, text="Enable", variable=self.vision_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(vision_options, text="Facts", variable=self.vision_facts_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Combobox(
            vision_options,
            textvariable=self.vision_mode_var,
            values=("numbered_page", "page_image"),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Combobox(
            vision_options,
            textvariable=self.vision_trigger_var,
            values=("suspicious", "layout", "all"),
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT)

        ttk.Label(root, text="Glossary").grid(row=10, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(root, textvariable=self.glossary_var).grid(row=10, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(root, text="Browse", command=self._choose_glossary).grid(row=10, column=2, pady=(0, 8))

        ttk.Label(root, text="Font").grid(row=11, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(root, textvariable=self.font_var).grid(row=11, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(root, text="Browse", command=self._choose_font).grid(row=11, column=2, pady=(0, 8))

        ttk.Label(root, text="Typesetting").grid(row=12, column=0, sticky="w", pady=(0, 8))
        type_options = ttk.Frame(root)
        type_options.grid(row=12, column=1, sticky="w", padx=8, pady=(0, 8))
        ttk.Label(type_options, text="Font size").pack(side=tk.LEFT)
        ttk.Spinbox(type_options, from_=10, to=72, textvariable=self.font_size_var, width=6).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(type_options, text="Box expand").pack(side=tk.LEFT)
        ttk.Spinbox(type_options, from_=1.0, to=4.0, increment=0.1, textvariable=self.render_expand_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(root, text="Work cache folder").grid(row=13, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(root, textvariable=self.work_dir_var).grid(row=13, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(root, text="Browse", command=self._choose_work_dir).grid(row=13, column=2, pady=(0, 8))

        ttk.Label(root, text="Tesseract path").grid(row=14, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(root, textvariable=self.tesseract_var).grid(row=14, column=1, sticky="ew", padx=8, pady=(0, 8))
        tesseract_buttons = ttk.Frame(root)
        tesseract_buttons.grid(row=14, column=2, pady=(0, 8))
        ttk.Button(tesseract_buttons, text="Browse", command=self._choose_tesseract).pack(side=tk.LEFT)
        ttk.Button(tesseract_buttons, text="Help", command=self._show_tesseract_help).pack(side=tk.LEFT, padx=(6, 0))

        options = ttk.Frame(root)
        options.grid(row=15, column=1, sticky="w", padx=8, pady=(0, 10))
        ttk.Checkbutton(options, text="Debug boxes", variable=self.debug_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(options, text="Overwrite existing", variable=self.overwrite_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(options, text="Resume cached work", variable=self.resume_var).pack(side=tk.LEFT)

        actions = ttk.Frame(root)
        actions.grid(row=16, column=0, columnspan=3, sticky="ew", pady=(2, 10))
        self.start_button = ttk.Button(actions, text="Start", command=self._start)
        self.start_button.pack(side=tk.LEFT)
        ttk.Button(actions, text="Install CTD detector", command=self._install_ctd).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Install OPUS translator", command=self._install_opus).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Install MADLAD translator", command=self._install_madlad).pack(side=tk.LEFT)
        ttk.Button(actions, text="Install Argos JA->EN", command=self._install_argos).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(actions, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        ttk.Label(root, text="Log").grid(row=17, column=0, sticky="w")
        self.log = tk.Text(root, height=10, wrap="word", state="disabled")
        self.log.grid(row=18, column=0, columnspan=3, sticky="nsew")

    def _choose_input(self) -> None:
        selected = filedialog.askdirectory(title="Choose folder with manga images")
        if not selected:
            logger.debug("Input folder selection cancelled")
            return
        self.input_var.set(selected)
        logger.info("Selected input folder: %s", selected)
        if not self.output_var.get():
            output = str(Path(selected).with_name(f"{Path(selected).name}-translated"))
            self.output_var.set(output)
            logger.info("Auto-selected output folder: %s", output)

    def _prefill_tesseract_path(self) -> None:
        found = find_tesseract()
        if found:
            self.tesseract_var.set(found)
            self._append_log(f"Tesseract found: {found}\n")
        else:
            self._append_log("Tesseract was not found. Click Tesseract path Help for install instructions.\n")

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Choose output folder")
        if selected:
            self.output_var.set(selected)
            logger.info("Selected output folder: %s", selected)
        else:
            logger.debug("Output folder selection cancelled")

    def _choose_tesseract(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose tesseract.exe",
            filetypes=(("Tesseract executable", "tesseract.exe"), ("Executables", "*.exe"), ("All files", "*.*")),
        )
        if selected:
            self.tesseract_var.set(selected)
            logger.info("Selected tesseract executable: %s", selected)
        else:
            logger.debug("Tesseract executable selection cancelled")

    def _choose_qwen_fallback(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose fallback Qwen GGUF model",
            filetypes=(("GGUF model", "*.gguf"), ("All files", "*.*")),
        )
        if selected:
            self.qwen_fallback_var.set(selected)
            logger.info("Selected Qwen fallback model: %s", selected)
        else:
            logger.debug("Qwen fallback model selection cancelled")

    def _choose_qwen_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose primary Qwen GGUF model",
            filetypes=(("GGUF model", "*.gguf"), ("All files", "*.*")),
        )
        if selected:
            self.qwen_model_var.set(selected)
            logger.info("Selected Qwen primary model: %s", selected)
        else:
            logger.debug("Qwen primary model selection cancelled")

    def _choose_glossary(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose translation glossary",
            filetypes=(("JSON glossary", "*.json"), ("All files", "*.*")),
        )
        if selected:
            self.glossary_var.set(selected)
            logger.info("Selected glossary: %s", selected)
        else:
            logger.debug("Glossary selection cancelled")

    def _choose_font(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose text font",
            filetypes=(("Font files", "*.ttf *.otf"), ("All files", "*.*")),
        )
        if selected:
            self.font_var.set(selected)
            logger.info("Selected font: %s", selected)
        else:
            logger.debug("Font selection cancelled")

    def _choose_work_dir(self) -> None:
        selected = filedialog.askdirectory(title="Choose work cache folder")
        if selected:
            self.work_dir_var.set(selected)
            logger.info("Selected work cache folder: %s", selected)
        else:
            logger.debug("Work cache folder selection cancelled")

    def _show_tesseract_help(self) -> None:
        logger.info("Showing Tesseract help dialog")
        messagebox.showinfo("Tesseract setup", tesseract_missing_message())

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            logger.warning("Start ignored because a worker is already running")
            return

        input_path = Path(self.input_var.get().strip())
        output_path = Path(self.output_var.get().strip())
        if not input_path.exists():
            logger.warning("Start blocked: input path does not exist: %s", input_path)
            messagebox.showerror("Missing input", "Choose an existing image folder first.")
            return
        if not output_path:
            logger.warning("Start blocked: output path is empty")
            messagebox.showerror("Missing output", "Choose an output folder first.")
            return

        config = PipelineConfig(
            detector=self.detector_var.get(),
            ocr_engine=self.ocr_engine_var.get(),
            translator=self.translator_var.get(),
            qwen_mode=self.qwen_mode_var.get(),
            erase_mode=self.erase_mode_var.get(),
            tesseract_cmd=self.tesseract_var.get().strip() or None,
            render_expand=float(self.render_expand_var.get()),
            font_path=Path(self.font_var.get().strip()) if self.font_var.get().strip() else None,
            glossary_path=Path(self.glossary_var.get().strip()) if self.glossary_var.get().strip() else None,
            qwen_model_path=Path(self.qwen_model_var.get().strip()) if self.qwen_model_var.get().strip() else None,
            qwen_fallback_model_path=Path(self.qwen_fallback_var.get().strip()) if self.qwen_fallback_var.get().strip() else None,
            vision_enabled=self.vision_var.get(),
            vision_facts_enabled=self.vision_facts_var.get(),
            vision_mode=self.vision_mode_var.get(),
            vision_trigger=self.vision_trigger_var.get(),
            base_font_size=int(self.font_size_var.get()),
            work_dir=Path(self.work_dir_var.get().strip()) if self.work_dir_var.get().strip() else None,
            debug=self.debug_var.get(),
            overwrite=self.overwrite_var.get(),
            resume=self.resume_var.get(),
        )
        logger.info("Starting translation from GUI")
        logger.debug("GUI pipeline config: %s", config)
        self._run_worker("Translating", lambda: self._translate(input_path, output_path, config))

    def _install_argos(self) -> None:
        if self.worker and self.worker.is_alive():
            logger.warning("Argos install ignored because a worker is already running")
            return
        logger.info("Starting Argos model install from GUI")
        self._run_worker("Installing Argos model", self._install_argos_worker)

    def _install_ctd(self) -> None:
        if self.worker and self.worker.is_alive():
            logger.warning("CTD install ignored because a worker is already running")
            return
        logger.info("Starting CTD install from GUI")
        self._run_worker("Installing CTD detector", self._install_ctd_worker)

    def _install_madlad(self) -> None:
        if self.worker and self.worker.is_alive():
            logger.warning("MADLAD install ignored because a worker is already running")
            return
        logger.info("Starting MADLAD model install from GUI")
        self._run_worker("Installing MADLAD translator", self._install_madlad_worker)

    def _install_opus(self) -> None:
        if self.worker and self.worker.is_alive():
            logger.warning("OPUS install ignored because a worker is already running")
            return
        logger.info("Starting OPUS model install from GUI")
        self._run_worker("Installing OPUS translator", self._install_opus_worker)

    def _run_worker(self, label: str, target) -> None:
        self.status_var.set(label)
        self.start_button.configure(state="disabled")
        self._append_log(f"{label}...\n")
        logger.info("Worker starting: %s", label)
        self.worker = threading.Thread(target=self._worker_wrapper, args=(target,), daemon=True)
        self.worker.start()

    def _worker_wrapper(self, target) -> None:
        try:
            target()
        except Exception:
            logger.exception("Worker failed")
            self.log_queue.put(traceback.format_exc())
            self.log_queue.put("FAILED\n")
        else:
            logger.info("Worker finished successfully")
            self.log_queue.put("Done.\n")
        finally:
            self.log_queue.put("__WORKER_DONE__")

    def _translate(self, input_path: Path, output_path: Path, config: PipelineConfig) -> None:
        from .dependencies import ensure_runtime_dependencies
        from .pipeline import process_folder

        self.log_queue.put("Checking Python dependencies...\n")
        logger.info("Checking runtime dependencies before translation")
        ensure_runtime_dependencies()
        self.log_queue.put(f"Input: {input_path}\n")
        self.log_queue.put(f"Output: {output_path}\n")
        logger.info("Processing input=%s output=%s", input_path, output_path)
        process_folder(input_path, output_path, config)

    def _install_argos_worker(self) -> None:
        from .dependencies import ensure_python_package
        from .install_argos import main

        self.log_queue.put("Checking argostranslate package...\n")
        logger.info("Checking argostranslate before model install")
        ensure_python_package("argostranslate", "argostranslate")
        main(["ja", "en"])

    def _install_ctd_worker(self) -> None:
        from .dependencies import ensure_python_package
        from .install_ctd import main

        self.log_queue.put("Checking CTD Python dependencies...\n")
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

        self.log_queue.put("Checking MADLAD Python dependencies...\n")
        logger.info("Checking MADLAD dependencies before translator install")
        for import_name, package_name in (
            ("transformers", "transformers"),
            ("sentencepiece", "sentencepiece"),
            ("accelerate", "accelerate"),
            ("huggingface_hub", "huggingface-hub"),
        ):
            ensure_python_package(import_name, package_name)
        self.log_queue.put("Downloading MADLAD model. This is large and can take a while the first time.\n")
        main([])

    def _install_opus_worker(self) -> None:
        from .dependencies import ensure_python_package
        from .install_opus import main

        self.log_queue.put("Checking OPUS Python dependencies...\n")
        logger.info("Checking OPUS dependencies before translator install")
        for import_name, package_name in (
            ("transformers", "transformers"),
            ("sentencepiece", "sentencepiece"),
            ("huggingface_hub", "huggingface-hub"),
        ):
            ensure_python_package(import_name, package_name)
        main([])

    def _drain_log_queue(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message == "__WORKER_DONE__":
                    self.start_button.configure(state="normal")
                    self.status_var.set("Ready")
                    logger.debug("Worker completion message handled")
                else:
                    self._append_log(message)
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, message)
        self.log.see(tk.END)
        self.log.configure(state="disabled")


def main() -> int:
    log_path = configure_logging(reset=True)
    logger.info("GUI main started")
    app = MangaTranslatorGui()
    app._append_log(f"Debug log: {log_path}\n")
    app.mainloop()
    logger.info("GUI main exited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
