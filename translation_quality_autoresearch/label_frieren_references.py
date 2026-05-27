from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
REFERENCE_ROLES = ("speech", "narration", "thought", "sfx", "sign", "metadata", "unknown")
ALIGNMENT_CONFIDENCES = ("high", "medium", "low")


@dataclass(frozen=True)
class PageCase:
    page_number: int
    page_id: str
    prepared_path: Path
    japanese_image: Path
    english_image: Path
    lines: tuple[dict[str, Any], ...]
    image_size: tuple[int, int]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_image(folder: Path, page_number: int) -> Path:
    stems = [f"{page_number:03d}", str(page_number)]
    for stem in stems:
        for suffix in IMAGE_SUFFIXES:
            path = folder / f"{stem}{suffix}"
            if path.exists():
                return path
    raise FileNotFoundError(f"No image found for page {page_number} in {folder}")


def find_prepared(prepared_dir: Path, page_number: int) -> Path:
    patterns = (
        f"*-{page_number:03d}.prepared.json",
        f"*-{page_number}.prepared.json",
    )
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(prepared_dir.glob(pattern)))
    if not matches:
        raise FileNotFoundError(f"No prepared cache found for page {page_number} in {prepared_dir}")
    return matches[0]


def load_page_case(
    *,
    page_number: int,
    prepared_dir: Path,
    japanese_dir: Path,
    english_dir: Path,
) -> PageCase:
    prepared_path = find_prepared(prepared_dir, page_number)
    data = load_json(prepared_path)
    image_size = (int(data.get("width") or 0), int(data.get("height") or 0))
    if image_size[0] <= 0 or image_size[1] <= 0:
        with Image.open(japanese_dir / f"{page_number:03d}.jpg") as image:
            image_size = image.size
    lines = []
    for item in sorted(data.get("page_order_report", []), key=lambda value: int(value.get("page_order") or 0)):
        if not isinstance(item, dict):
            continue
        line_id = str(item.get("line_id") or "").strip()
        box = item.get("box")
        if not line_id or not isinstance(box, list) or len(box) != 4:
            continue
        lines.append(
            {
                "line_id": line_id,
                "page_order": int(item.get("page_order") or 0),
                "source_text": str(item.get("source_text") or ""),
                "source_box": [int(value) for value in box],
                "block_id": str(item.get("block_id") or ""),
                "source_hash": str(item.get("source_hash") or ""),
            }
        )
    if not lines:
        raise RuntimeError(f"Prepared cache has no labelable page_order_report lines: {prepared_path}")
    return PageCase(
        page_number=page_number,
        page_id=f"frieren_ch26_{page_number:03d}",
        prepared_path=prepared_path,
        japanese_image=find_image(japanese_dir, page_number),
        english_image=find_image(english_dir, page_number),
        lines=tuple(lines),
        image_size=image_size,
    )


def default_label_payload(case: PageCase) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "frieren_ch26_pages_002_010",
        "page_id": case.page_id,
        "page_number": case.page_number,
        "japanese_image": str(case.japanese_image),
        "english_image": str(case.english_image),
        "prepared_cache": str(case.prepared_path),
        "image_size": [case.image_size[0], case.image_size[1]],
        "labels": [
            {
                "page_id": case.page_id,
                "line_id": line["line_id"],
                "page_order": line["page_order"],
                "source_text": line["source_text"],
                "source_box": line["source_box"],
                "block_id": line["block_id"],
                "source_hash": line["source_hash"],
                "reference_text": "",
                "alignment_confidence": "high",
                "reference_role": "speech",
                "skip_reference": False,
                "skip_reason": "",
                "notes": "",
            }
            for line in case.lines
        ],
    }


class ReferenceLabeler(tk.Tk):
    def __init__(self, cases: list[PageCase], output_dir: Path) -> None:
        super().__init__()
        self.title("Frieren English Reference Labeler")
        self.geometry("1680x980")
        self.minsize(1250, 760)

        self.cases = cases
        self.output_dir = output_dir
        self.labels_dir = output_dir / "labels"
        self.page_index = 0
        self.line_index = 0
        self.page_payloads: dict[str, dict[str, Any]] = {}

        self.jp_image: Image.Image | None = None
        self.en_image: Image.Image | None = None
        self.jp_photo: ImageTk.PhotoImage | None = None
        self.en_photo: ImageTk.PhotoImage | None = None
        self.zoom = 0.55

        self.reference_text: tk.Text
        self.notes_text: tk.Text
        self.skip_reason_var = tk.StringVar()
        self.confidence_var = tk.StringVar(value="high")
        self.role_var = tk.StringVar(value="speech")
        self.skip_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="")
        self.line_info_var = tk.StringVar(value="")

        self._build_ui()
        self._load_all_labels()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._load_page(0)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=6)
        toolbar.grid(row=0, column=0, columnspan=4, sticky="ew")
        ttk.Button(toolbar, text="Save", command=self.save_current).pack(side="left")
        ttk.Button(toolbar, text="Export JSONL", command=self.export_jsonl).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Prev Page", command=lambda: self.goto_page(self.page_index - 1)).pack(side="left", padx=(20, 0))
        ttk.Button(toolbar, text="Next Page", command=lambda: self.goto_page(self.page_index + 1)).pack(side="left")
        ttk.Button(toolbar, text="Prev Line", command=lambda: self.goto_line(self.line_index - 1)).pack(side="left", padx=(20, 0))
        ttk.Button(toolbar, text="Next Line", command=lambda: self.goto_line(self.line_index + 1)).pack(side="left")
        ttk.Button(toolbar, text="Zoom -", command=lambda: self.set_zoom(self.zoom / 1.2)).pack(side="left", padx=(20, 0))
        ttk.Button(toolbar, text="Zoom +", command=lambda: self.set_zoom(self.zoom * 1.2)).pack(side="left")
        ttk.Button(toolbar, text="Fit", command=self.fit_zoom).pack(side="left")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="left", padx=(20, 0))

        page_frame = ttk.Frame(self, padding=(6, 0, 0, 6))
        page_frame.grid(row=1, column=0, sticky="ns")
        ttk.Label(page_frame, text="Pages").pack(anchor="w")
        self.page_list = tk.Listbox(page_frame, width=22, exportselection=False)
        self.page_list.pack(fill="y", expand=True)
        for case in self.cases:
            self.page_list.insert("end", f"{case.page_number:03d}")
        self.page_list.bind("<<ListboxSelect>>", self.on_page_selected)

        jp_frame = self._canvas_frame("Japanese Source")
        jp_frame.grid(row=1, column=1, sticky="nsew", padx=(6, 3), pady=(0, 6))
        self.jp_canvas = jp_frame.canvas

        en_frame = self._canvas_frame("English Reference")
        en_frame.grid(row=1, column=2, sticky="nsew", padx=(3, 6), pady=(0, 6))
        self.en_canvas = en_frame.canvas

        editor = ttk.Frame(self, padding=(6, 0, 6, 6), width=390)
        editor.grid(row=1, column=3, sticky="ns")
        editor.grid_propagate(False)
        editor.columnconfigure(0, weight=1)

        ttk.Label(editor, textvariable=self.line_info_var, wraplength=360).grid(row=0, column=0, sticky="ew")
        ttk.Label(editor, text="Reference Text").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.reference_text = tk.Text(editor, width=44, height=8, wrap="word")
        self.reference_text.grid(row=2, column=0, sticky="ew")
        self.reference_text.bind("<Control-Return>", lambda _event: self.save_and_next())

        form = ttk.Frame(editor)
        form.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Confidence").grid(row=0, column=0, sticky="w")
        ttk.Combobox(form, textvariable=self.confidence_var, values=ALIGNMENT_CONFIDENCES, state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Label(form, text="Role").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(form, textvariable=self.role_var, values=REFERENCE_ROLES, state="readonly").grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(form, text="Skip reference", variable=self.skip_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(form, text="Skip reason").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.skip_reason_var).grid(row=3, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(editor, text="Notes").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.notes_text = tk.Text(editor, width=44, height=5, wrap="word")
        self.notes_text.grid(row=5, column=0, sticky="ew")

        ttk.Button(editor, text="Save And Next", command=self.save_and_next).grid(row=6, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            editor,
            text="Tip: transcribe the matching English bubble. Use skip when the English page has no usable equivalent. Ctrl+Enter saves and advances.",
            wraplength=360,
        ).grid(row=7, column=0, sticky="ew", pady=(14, 0))

        line_frame = ttk.Frame(editor)
        line_frame.grid(row=8, column=0, sticky="nsew", pady=(14, 0))
        line_frame.columnconfigure(0, weight=1)
        editor.rowconfigure(8, weight=1)
        ttk.Label(line_frame, text="Lines").grid(row=0, column=0, sticky="w")
        self.line_list = tk.Listbox(line_frame, height=12, exportselection=False)
        self.line_list.grid(row=1, column=0, sticky="nsew")
        line_frame.rowconfigure(1, weight=1)
        self.line_list.bind("<<ListboxSelect>>", self.on_line_selected)

        self.bind("<Control-s>", lambda _event: self.save_current())
        self.bind("<Prior>", lambda _event: self.goto_page(self.page_index - 1))
        self.bind("<Next>", lambda _event: self.goto_page(self.page_index + 1))

    def _canvas_frame(self, title: str):
        frame = ttk.Frame(self)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        ttk.Label(frame, text=title).grid(row=0, column=0, sticky="w")
        canvas = tk.Canvas(frame, background="#202020", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        xscroll.grid(row=2, column=0, sticky="ew")
        yscroll.grid(row=1, column=1, sticky="ns")
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.bind("<Button-1>", self.on_canvas_click)
        frame.canvas = canvas  # type: ignore[attr-defined]
        return frame

    def _load_all_labels(self) -> None:
        for case in self.cases:
            path = self.label_path(case)
            if path.exists():
                self.page_payloads[case.page_id] = load_json(path)
            else:
                self.page_payloads[case.page_id] = default_label_payload(case)

    def label_path(self, case: PageCase) -> Path:
        return self.labels_dir / f"{case.page_id}.references.json"

    def current_case(self) -> PageCase:
        return self.cases[self.page_index]

    def current_payload(self) -> dict[str, Any]:
        return self.page_payloads[self.current_case().page_id]

    def current_label(self) -> dict[str, Any]:
        return self.current_payload()["labels"][self.line_index]

    def _load_page(self, index: int) -> None:
        self.page_index = max(0, min(len(self.cases) - 1, index))
        self.line_index = 0
        case = self.current_case()
        self.jp_image = Image.open(case.japanese_image).convert("RGB")
        self.en_image = Image.open(case.english_image).convert("RGB")
        self.page_list.selection_clear(0, "end")
        self.page_list.selection_set(self.page_index)
        self.page_list.see(self.page_index)
        self.fit_zoom()
        self.refresh_line_list()
        self.load_editor()

    def refresh_line_list(self) -> None:
        self.line_list.delete(0, "end")
        for label in self.current_payload()["labels"]:
            status = "skip" if label.get("skip_reference") else ("ok" if str(label.get("reference_text") or "").strip() else "todo")
            preview = " ".join(str(label.get("reference_text") or "").split())
            if len(preview) > 28:
                preview = preview[:25] + "..."
            self.line_list.insert("end", f"{int(label['page_order']):02d} {label['line_id']} [{status}] {preview}")
        self.line_list.selection_clear(0, "end")
        self.line_list.selection_set(self.line_index)
        self.line_list.see(self.line_index)

    def load_editor(self) -> None:
        label = self.current_label()
        self.reference_text.delete("1.0", "end")
        self.reference_text.insert("1.0", str(label.get("reference_text") or ""))
        self.notes_text.delete("1.0", "end")
        self.notes_text.insert("1.0", str(label.get("notes") or ""))
        self.confidence_var.set(str(label.get("alignment_confidence") or "high"))
        self.role_var.set(str(label.get("reference_role") or "speech"))
        self.skip_var.set(bool(label.get("skip_reference")))
        self.skip_reason_var.set(str(label.get("skip_reason") or ""))
        self.line_info_var.set(
            f"Page {self.current_case().page_number}, order {label['page_order']}, {label['line_id']}\n"
            f"Source: {label.get('source_text', '')}"
        )
        self.status_var.set(self.progress_text())
        self.draw_canvases()

    def save_editor_to_memory(self) -> None:
        label = self.current_label()
        label["reference_text"] = self.reference_text.get("1.0", "end").strip()
        label["alignment_confidence"] = self.confidence_var.get() or "high"
        label["reference_role"] = self.role_var.get() or "speech"
        label["skip_reference"] = bool(self.skip_var.get())
        label["skip_reason"] = self.skip_reason_var.get().strip()
        label["notes"] = self.notes_text.get("1.0", "end").strip()

    def save_current(self) -> None:
        self.save_editor_to_memory()
        case = self.current_case()
        write_json(self.label_path(case), self.current_payload())
        self.export_jsonl(show_message=False)
        self.refresh_line_list()
        self.status_var.set(f"Saved {case.page_id}. {self.progress_text()}")

    def save_and_next(self) -> str:
        self.save_current()
        if self.line_index < len(self.current_payload()["labels"]) - 1:
            self.goto_line(self.line_index + 1)
        elif self.page_index < len(self.cases) - 1:
            self.goto_page(self.page_index + 1)
        return "break"

    def export_jsonl(self, *, show_message: bool = True) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for case in self.cases:
            payload = self.page_payloads[case.page_id]
            for label in payload.get("labels", []):
                rows.append(dict(label))
        target = self.output_dir / "references.jsonl"
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "dataset": "frieren_ch26_pages_002_010",
            "pages": [case.page_id for case in self.cases],
            "references": str(target),
            "labels_dir": str(self.labels_dir),
        }
        write_json(self.output_dir / "manifest.json", manifest)
        if show_message:
            messagebox.showinfo("Export complete", f"Wrote {target}")

    def goto_page(self, index: int) -> None:
        if index < 0 or index >= len(self.cases):
            return
        self.save_current()
        self._load_page(index)

    def goto_line(self, index: int) -> None:
        labels = self.current_payload()["labels"]
        if index < 0 or index >= len(labels):
            return
        self.save_editor_to_memory()
        self.line_index = index
        self.refresh_line_list()
        self.load_editor()

    def on_page_selected(self, _event) -> None:
        selection = self.page_list.curselection()
        if selection and selection[0] != self.page_index:
            self.goto_page(selection[0])

    def on_line_selected(self, _event) -> None:
        selection = self.line_list.curselection()
        if selection and selection[0] != self.line_index:
            self.goto_line(selection[0])

    def set_zoom(self, value: float) -> None:
        self.zoom = max(0.15, min(2.5, value))
        self.draw_canvases()

    def fit_zoom(self) -> None:
        case = self.current_case()
        target_width = 560
        target_height = 780
        self.zoom = min(target_width / max(1, case.image_size[0]), target_height / max(1, case.image_size[1]))
        self.draw_canvases()

    def draw_canvases(self) -> None:
        if self.jp_image is None or self.en_image is None:
            return
        self._draw_image_canvas(self.jp_canvas, self.jp_image, is_japanese=True)
        self._draw_image_canvas(self.en_canvas, self.en_image, is_japanese=False)

    def _draw_image_canvas(self, canvas: tk.Canvas, image: Image.Image, *, is_japanese: bool) -> None:
        canvas.delete("all")
        width = max(1, int(image.width * self.zoom))
        height = max(1, int(image.height * self.zoom))
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        if is_japanese:
            self.jp_photo = photo
        else:
            self.en_photo = photo
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, width, height))
        labels = self.current_payload()["labels"]
        selected = self.current_label()
        for index, label in enumerate(labels):
            if not is_japanese and label is not selected:
                continue
            box = self.scaled_box(label["source_box"], image.size)
            color = "#ffcc33" if index == self.line_index else "#33d17a"
            line_width = 4 if index == self.line_index else 2
            canvas.create_rectangle(*box, outline=color, width=line_width, tags=(f"line:{index}",))
            if is_japanese:
                canvas.create_text(box[0] + 4, max(10, box[1] - 10), text=str(label["page_order"]), fill=color, anchor="w", tags=(f"line:{index}",))

    def scaled_box(self, box: list[int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        case_width, case_height = self.current_case().image_size
        image_width, image_height = image_size
        sx = image_width / max(1, case_width) * self.zoom
        sy = image_height / max(1, case_height) * self.zoom
        x1, y1, x2, y2 = box
        return (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))

    def on_canvas_click(self, event) -> None:
        canvas = event.widget
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        for index, label in enumerate(self.current_payload()["labels"]):
            x1, y1, x2, y2 = self.scaled_box(label["source_box"], self.jp_image.size if canvas is self.jp_canvas else self.en_image.size)  # type: ignore[union-attr]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.goto_line(index)
                return

    def progress_text(self) -> str:
        labels = [label for payload in self.page_payloads.values() for label in payload.get("labels", [])]
        done = sum(1 for label in labels if label.get("skip_reference") or str(label.get("reference_text") or "").strip())
        return f"{done}/{len(labels)} references labeled"

    def on_close(self) -> None:
        self.save_current()
        self.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label same-layout English Frieren references for translation quality evaluation.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("quality-runs/frieren-cat-final-20260527/quality/.manga-work"))
    parser.add_argument("--japanese-dir", type=Path, default=Path("mangafolder/sousou-no-frieren-chapter-26"))
    parser.add_argument("--english-dir", type=Path, default=Path("mangafolder/Frieren-Chapter 26 Present for a Warrior-ENGLISH"))
    parser.add_argument("--output", type=Path, default=Path("translation_quality_autoresearch/benchmarks/frieren_ch26_pages_002_010"))
    parser.add_argument("--start-page", type=int, default=2)
    parser.add_argument("--end-page", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [
        load_page_case(
            page_number=page_number,
            prepared_dir=args.prepared_dir,
            japanese_dir=args.japanese_dir,
            english_dir=args.english_dir,
        )
        for page_number in range(args.start_page, args.end_page + 1)
    ]
    app = ReferenceLabeler(cases, args.output)
    app.mainloop()


if __name__ == "__main__":
    main()
