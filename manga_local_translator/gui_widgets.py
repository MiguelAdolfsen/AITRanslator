"""Theme, image assets, and reusable Tkinter widgets for the desktop GUI."""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:  # pragma: no cover - Pillow is a runtime dependency; widgets fall back to text.
    Image = ImageDraw = ImageFont = ImageTk = None

logger = logging.getLogger(__name__)

# Palette
BG = "#F5F6F8"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F9FAFB"
TRACK = "#F2F4F7"
BORDER = "#E4E7EC"
BORDER_STRONG = "#D0D5DD"
TEXT = "#101828"
TEXT_SECONDARY = "#344054"
MUTED = "#667085"
SUBTLE = "#98A2B3"
ACCENT = "#4F46E5"
ACCENT_HOVER = "#4338CA"
ACCENT_PRESSED = "#3730A3"
ACCENT_SOFT = "#EEF2FF"
ACCENT_SOFT_BORDER = "#C7D2FE"
ACCENT_DISABLED = "#A5B4FC"
SUCCESS = "#079455"
SUCCESS_SOFT = "#ECFDF3"
WARNING = "#DC6803"
WARNING_SOFT = "#FFFAEB"
DANGER = "#D92D20"
DANGER_SOFT = "#FEF3F2"
SIDEBAR_BG = "#0F172A"
SIDEBAR_HOVER = "#1A2338"
SIDEBAR_ACTIVE = "#222D45"
SIDEBAR_TEXT = "#94A3B8"
SIDEBAR_TEXT_ACTIVE = "#F8FAFC"
SIDEBAR_ICON_ACTIVE = "#A5B4FC"
LOG_BG = "#0B1220"
LOG_TEXT = "#CBD5E1"
LOG_MUTED = "#64748B"
LOG_ACCENT = "#A5B4FC"
LOG_SUCCESS = "#4ADE80"
LOG_WARNING = "#FBBF24"
LOG_DANGER = "#F87171"

# Segoe MDL2 / Fluent icon code points (shared by both fonts).
ICONS = {
    "project": "\uE8B7",
    "translation": "\uE8C1",
    "detection": "\uE721",
    "vision": "\uE890",
    "typesetting": "\uE8D2",
    "setup": "\uE896",
    "info": "\uE946",
    "warning": "\uE7BA",
}

LEVEL_COLORS = {
    "ok": (SUCCESS, SUCCESS_SOFT),
    "warning": (WARNING, WARNING_SOFT),
    "error": (DANGER, DANGER_SOFT),
    "info": (ACCENT, ACCENT_SOFT),
    "neutral": (MUTED, TRACK),
}


def ui_scale(root: tk.Misc) -> float:
    try:
        return max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
    except tk.TclError:
        return 1.0


@dataclass(frozen=True)
class Fonts:
    body: tkfont.Font
    body_strong: tkfont.Font
    small: tkfont.Font
    small_strong: tkfont.Font
    caption: tkfont.Font
    heading: tkfont.Font
    title: tkfont.Font
    brand: tkfont.Font
    mono: tkfont.Font
    icon: tkfont.Font | None


def create_fonts(root: tk.Misc) -> Fonts:
    families = set(tkfont.families(root))

    def pick(*candidates: str) -> str | None:
        return next((candidate for candidate in candidates if candidate in families), None)

    base = pick("Segoe UI", "Helvetica Neue", "Helvetica", "Arial") or tkfont.nametofont("TkDefaultFont").actual("family")
    semibold = pick("Segoe UI Semibold")
    mono = pick("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono") or "Courier"
    icon = pick("Segoe Fluent Icons", "Segoe MDL2 Assets")

    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
        try:
            tkfont.nametofont(name).configure(family=base, size=10)
        except tk.TclError:
            pass

    def regular(size: int) -> tkfont.Font:
        return tkfont.Font(root, family=base, size=size)

    def strong(size: int) -> tkfont.Font:
        if semibold:
            return tkfont.Font(root, family=semibold, size=size)
        return tkfont.Font(root, family=base, size=size, weight="bold")

    return Fonts(
        body=regular(10),
        body_strong=strong(10),
        small=regular(9),
        small_strong=strong(9),
        caption=strong(8),
        heading=strong(12),
        title=strong(18),
        brand=strong(12),
        mono=tkfont.Font(root, family=mono, size=9),
        icon=tkfont.Font(root, family=icon, size=12) if icon else None,
    )


def apply_styles(root: tk.Misc, fonts: Fonts, scale: float) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        logger.warning("ttk clam theme unavailable; GUI styling will be limited")
        return

    root.option_add("*Font", fonts.body)
    style.configure(
        ".",
        font=fonts.body,
        background=BG,
        foreground=TEXT,
        bordercolor=BORDER_STRONG,
        lightcolor=SURFACE,
        darkcolor=SURFACE,
        troughcolor=TRACK,
        focuscolor=ACCENT,
        selectbackground=ACCENT_SOFT_BORDER,
        selectforeground=TEXT,
        insertcolor=TEXT,
        arrowcolor=MUTED,
    )

    style.configure(
        "TEntry",
        fieldbackground=SURFACE,
        foreground=TEXT,
        bordercolor=BORDER_STRONG,
        lightcolor=SURFACE,
        darkcolor=SURFACE,
        padding=(10, 7),
        insertwidth=1,
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", ACCENT), ("hover", SUBTLE)],
        lightcolor=[("focus", ACCENT_SOFT_BORDER)],
        darkcolor=[("focus", ACCENT_SOFT_BORDER)],
        fieldbackground=[("disabled", SURFACE_ALT)],
        foreground=[("disabled", SUBTLE)],
    )

    button_layout = [
        (
            "Button.border",
            {
                "sticky": "nswe",
                "border": "1",
                "children": [("Button.padding", {"sticky": "nswe", "children": [("Button.label", {"sticky": "nswe"})]})],
            },
        )
    ]
    for name in ("TButton", "Accent.TButton", "Small.TButton", "Ghost.TButton"):
        style.layout(name, button_layout)

    def button_style(
        name: str,
        *,
        bg: str,
        fg: str,
        border: str,
        hover: str,
        pressed: str,
        disabled_bg: str,
        disabled_fg: str,
        focus_border: str,
        padding: tuple[int, int],
        font: tkfont.Font,
    ) -> None:
        style.configure(
            name,
            background=bg,
            foreground=fg,
            bordercolor=border,
            lightcolor=bg,
            darkcolor=bg,
            padding=padding,
            font=font,
            anchor="center",
            width=0,
        )
        state_bg = [("disabled", disabled_bg), ("pressed", pressed), ("active", hover)]
        style.map(
            name,
            background=state_bg,
            lightcolor=state_bg,
            darkcolor=state_bg,
            foreground=[("disabled", disabled_fg)],
            bordercolor=[("disabled", disabled_bg if border == bg else BORDER), ("focus", focus_border)],
        )

    button_style(
        "TButton",
        bg=SURFACE,
        fg=TEXT_SECONDARY,
        border=BORDER_STRONG,
        hover=SURFACE_ALT,
        pressed=TRACK,
        disabled_bg=SURFACE_ALT,
        disabled_fg=SUBTLE,
        focus_border=ACCENT,
        padding=(14, 7),
        font=fonts.body_strong,
    )
    button_style(
        "Small.TButton",
        bg=SURFACE,
        fg=TEXT_SECONDARY,
        border=BORDER_STRONG,
        hover=SURFACE_ALT,
        pressed=TRACK,
        disabled_bg=SURFACE_ALT,
        disabled_fg=SUBTLE,
        focus_border=ACCENT,
        padding=(10, 5),
        font=fonts.small_strong,
    )
    button_style(
        "Accent.TButton",
        bg=ACCENT,
        fg="#FFFFFF",
        border=ACCENT,
        hover=ACCENT_HOVER,
        pressed=ACCENT_PRESSED,
        disabled_bg=ACCENT_DISABLED,
        disabled_fg="#EEF2FF",
        focus_border=ACCENT_PRESSED,
        padding=(18, 11),
        font=fonts.body_strong,
    )
    button_style(
        "Ghost.TButton",
        bg=SURFACE,
        fg=MUTED,
        border=SURFACE,
        hover=TRACK,
        pressed=BORDER,
        disabled_bg=SURFACE,
        disabled_fg=SUBTLE,
        focus_border=BORDER_STRONG,
        padding=(8, 4),
        font=fonts.small_strong,
    )

    for name, color in (("Accent", ACCENT), ("Success", SUCCESS), ("Danger", DANGER)):
        style.configure(
            f"{name}.Horizontal.TProgressbar",
            troughcolor=TRACK,
            background=color,
            bordercolor=TRACK,
            lightcolor=color,
            darkcolor=color,
            thickness=max(6, int(round(6 * scale))),
        )

    scroll_layout = [
        ("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})
    ]
    for name, trough, thumb, thumb_hover in (
        ("Slim.Vertical.TScrollbar", BG, BORDER_STRONG, SUBTLE),
        ("Log.Vertical.TScrollbar", LOG_BG, "#273449", "#3B4A63"),
    ):
        style.layout(name, scroll_layout)
        style.configure(
            name,
            troughcolor=trough,
            background=thumb,
            bordercolor=trough,
            lightcolor=thumb,
            darkcolor=thumb,
            arrowsize=max(8, int(round(9 * scale))),
            gripcount=0,
            relief="flat",
        )
        style.map(
            name,
            background=[("pressed", thumb_hover), ("active", thumb_hover)],
            lightcolor=[("pressed", thumb_hover), ("active", thumb_hover)],
            darkcolor=[("pressed", thumb_hover), ("active", thumb_hover)],
        )



class Assets:
    """Anti-aliased images rendered with Pillow and cached as Tk photo images."""

    SUPERSAMPLE = 4
    JAPANESE_FONTS = (
        "C:/Windows/Fonts/YuGothB.ttc",
        "C:/Windows/Fonts/meiryob.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    )

    def __init__(self, root: tk.Misc, scale: float) -> None:
        self.root = root
        self.scale = scale
        self.available = ImageTk is not None
        self._cache: dict[tuple, object] = {}

    def _px(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    def _photo(self, key: tuple, render: Callable[[], object]):
        if not self.available:
            return None
        if key not in self._cache:
            try:
                self._cache[key] = ImageTk.PhotoImage(render(), master=self.root)
            except Exception:  # pragma: no cover - asset rendering must never break the GUI.
                logger.exception("Failed to render GUI asset %s", key)
                self.available = False
                return None
        return self._cache[key]

    def _canvas(self, width: int, height: int):
        ss = self.SUPERSAMPLE
        image = Image.new("RGBA", (width * ss, height * ss), (0, 0, 0, 0))
        return image, ImageDraw.Draw(image), ss

    @staticmethod
    def _finish(image, width: int, height: int):
        return image.resize((width, height), Image.LANCZOS)

    def switch(self, on: bool, enabled: bool = True):
        width, height = self._px(38), self._px(22)

        def render():
            image, draw, ss = self._canvas(width, height)
            if enabled:
                track = ACCENT if on else BORDER_STRONG
            else:
                track = ACCENT_SOFT_BORDER if on else BORDER
            w, h = width * ss, height * ss
            draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=h / 2, fill=track)
            pad = 2.5 * self.scale * ss
            diameter = h - 2 * pad
            x = w - pad - diameter if on else pad
            draw.ellipse((x, pad, x + diameter, pad + diameter), fill="#FFFFFF" if enabled else SURFACE_ALT)
            return self._finish(image, width, height)

        return self._photo(("switch", on, enabled), render)

    def radio(self, selected: bool, enabled: bool = True):
        size = self._px(18)

        def render():
            image, draw, ss = self._canvas(size, size)
            s = size * ss
            if selected:
                fill = ACCENT if enabled else ACCENT_DISABLED
                draw.ellipse((0, 0, s - 1, s - 1), fill=fill)
                inset = s * 0.32
                draw.ellipse((inset, inset, s - inset, s - inset), fill="#FFFFFF")
            else:
                line = max(1.0, 1.5 * self.scale) * ss
                draw.ellipse((0, 0, s - 1, s - 1), fill=BORDER_STRONG if enabled else BORDER)
                draw.ellipse((line, line, s - 1 - line, s - 1 - line), fill="#FFFFFF")
            return self._finish(image, size, size)

        return self._photo(("radio", selected, enabled), render)

    def status(self, level: str):
        size = self._px(18)

        def render():
            image, draw, ss = self._canvas(size, size)
            s = size * ss
            color, soft = LEVEL_COLORS.get(level, LEVEL_COLORS["neutral"])
            draw.ellipse((0, 0, s - 1, s - 1), fill=soft)
            width = max(2, int(round(1.8 * self.scale * ss)))

            def point(x: float, y: float) -> tuple[float, float]:
                return (x * s, y * s)

            if level == "ok":
                draw.line([point(0.29, 0.52), point(0.44, 0.66), point(0.72, 0.36)], fill=color, width=width, joint="curve")
            elif level in {"warning", "info"}:
                top, bottom = (0.26, 0.56) if level == "warning" else (0.44, 0.74)
                dot_y = 0.73 if level == "warning" else 0.29
                draw.line([point(0.5, top), point(0.5, bottom)], fill=color, width=width)
                radius = width * 0.62
                cx, cy = point(0.5, dot_y)
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
            elif level == "error":
                draw.line([point(0.35, 0.35), point(0.65, 0.65)], fill=color, width=width)
                draw.line([point(0.65, 0.35), point(0.35, 0.65)], fill=color, width=width)
            else:
                radius = s * 0.13
                draw.ellipse((s / 2 - radius, s / 2 - radius, s / 2 + radius, s / 2 + radius), fill=color)
            return self._finish(image, size, size)

        return self._photo(("status", level), render)

    def slider_thumb(self, focused: bool = False):
        size = self._px(22)

        def render():
            image, draw, ss = self._canvas(size, size)
            s = size * ss
            if focused:
                draw.ellipse((0, 0, s - 1, s - 1), fill=ACCENT_SOFT_BORDER)
            inset = s * 0.14
            draw.ellipse((inset, inset, s - 1 - inset, s - 1 - inset), fill=ACCENT)
            ring = inset + max(1.5, 2.2 * self.scale) * ss
            draw.ellipse((ring, ring, s - 1 - ring, s - 1 - ring), fill="#FFFFFF")
            return self._finish(image, size, size)

        return self._photo(("slider_thumb", focused), render)

    def logo(self, logical_size: int = 36, *, unscaled: bool = False):
        size = logical_size if unscaled else self._px(logical_size)
        return self._photo(("logo", size), lambda: self.logo_image(size))

    def logo_image(self, size: int):
        image, draw, ss = self._canvas(size, size)
        s = size * ss
        gradient = Image.new("RGBA", (s, s))
        top, bottom = (99, 102, 241), (67, 56, 202)
        gradient_draw = ImageDraw.Draw(gradient)
        for y in range(s):
            t = y / max(1, s - 1)
            gradient_draw.line(
                [(0, y), (s, y)],
                fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)) + (255,),
            )
        mask = Image.new("L", (s, s), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=s * 0.26, fill=255)
        image.paste(gradient, (0, 0), mask)
        glyph, font = self._logo_glyph(int(s * 0.56))
        box = draw.textbbox((0, 0), glyph, font=font)
        x = (s - (box[2] - box[0])) / 2 - box[0]
        y = (s - (box[3] - box[1])) / 2 - box[1]
        draw.text((x, y), glyph, font=font, fill="#FFFFFF")
        return self._finish(image, size, size)

    def _logo_glyph(self, pixel_size: int):
        for candidate in self.JAPANESE_FONTS:
            if Path(candidate).exists():
                try:
                    return "訳", ImageFont.truetype(candidate, pixel_size)
                except OSError:
                    continue
        try:
            return "JA", ImageFont.truetype("arialbd.ttf", int(pixel_size * 0.7))
        except OSError:
            return "JA", ImageFont.load_default()


def is_descendant(widget: tk.Misc | None, ancestor: tk.Misc) -> bool:
    while widget is not None:
        if widget is ancestor:
            return True
        widget = widget.master
    return False


def bind_tree(widget: tk.Misc, sequence: str, callback: Callable) -> None:
    widget.bind(sequence, callback, add="+")
    for child in widget.winfo_children():
        bind_tree(child, sequence, callback)


class WrapLabel(tk.Label):
    """Label that re-wraps its text to the width it is given."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        kwargs.setdefault("justify", "left")
        kwargs.setdefault("anchor", "w")
        kwargs.setdefault("wraplength", 320)
        super().__init__(master, **kwargs)
        self.bind("<Configure>", self._rewrap, add="+")

    def _rewrap(self, event: tk.Event) -> None:
        width = max(80, event.width - 2)
        if abs(int(float(self.cget("wraplength") or 0)) - width) > 2:
            self.configure(wraplength=width)


class ScrollFrame(tk.Frame):
    """Vertically scrolling page whose content column is centered with a max width."""

    def __init__(self, master: tk.Misc, *, bg: str, max_width: int, padding: tuple[int, int]) -> None:
        super().__init__(master, bg=bg)
        self.max_width = max_width
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0, yscrollincrement=12)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview, style="Slim.Vertical.TScrollbar")
        self.content = tk.Frame(self.canvas, bg=bg, padx=padding[0], pady=padding[1])
        self._window = self.canvas.create_window(0, 0, window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self._on_scroll_set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_content_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), self.content.winfo_reqheight()))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        width = min(event.width, self.max_width)
        self.canvas.itemconfigure(self._window, width=width)
        self.canvas.coords(self._window, max(0, (event.width - width) // 2), 0)
        self.canvas.configure(scrollregion=(0, 0, event.width, self.content.winfo_reqheight()))

    def _on_scroll_set(self, first: str, last: str) -> None:
        if float(first) <= 0.0 and float(last) >= 1.0:
            self.scrollbar.grid_remove()
        else:
            self.scrollbar.grid()
        self.scrollbar.set(first, last)

    def scroll_wheel(self, delta: int) -> None:
        first, last = self.canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return
        steps = -1 if delta > 0 else 1
        self.canvas.yview_scroll(steps * 4, "units")

    def scroll_to_top(self) -> None:
        self.canvas.yview_moveto(0)


class Card(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        fonts: Fonts,
        *,
        title: str | None = None,
        description: str | None = None,
        trailing: Callable[[tk.Frame], tk.Widget] | None = None,
        padding: int = 20,
    ) -> None:
        super().__init__(master, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER, highlightcolor=BORDER)
        if title:
            header = tk.Frame(self, bg=SURFACE)
            header.pack(fill="x", padx=padding, pady=(padding - 2, 0))
            header.columnconfigure(0, weight=1)
            tk.Label(header, text=title, font=fonts.heading, bg=SURFACE, fg=TEXT, anchor="w").grid(row=0, column=0, sticky="w")
            if trailing is not None:
                trailing(header).grid(row=0, column=1, rowspan=2, sticky="ne")
            if description:
                WrapLabel(header, text=description, font=fonts.small, bg=SURFACE, fg=MUTED).grid(
                    row=1, column=0, sticky="ew", pady=(3, 0)
                )
        self.body = tk.Frame(self, bg=SURFACE)
        self.body.pack(fill="both", expand=True, padx=padding, pady=(16 if title else padding, padding))


def badge(master: tk.Misc, fonts: Fonts, text: str, level: str = "neutral") -> tk.Label:
    fg, bg = LEVEL_COLORS.get(level, LEVEL_COLORS["neutral"])
    return tk.Label(master, text=text, font=fonts.caption, fg=fg, bg=bg, padx=7, pady=1)


def set_badge(label: tk.Label, text: str, level: str) -> None:
    fg, bg = LEVEL_COLORS.get(level, LEVEL_COLORS["neutral"])
    label.configure(text=text, fg=fg, bg=bg)


def separator(master: tk.Misc, *, color: str = BORDER, pady: int | tuple[int, int] = 14) -> tk.Frame:
    line = tk.Frame(master, bg=color, height=1)
    line.pack(fill="x", pady=pady)
    return line


class Callout(tk.Frame):
    def __init__(self, master: tk.Misc, fonts: Fonts, *, text: str, level: str = "info", action: tuple[str, Callable[[], None]] | None = None) -> None:
        color, soft = LEVEL_COLORS.get(level, LEVEL_COLORS["info"])
        super().__init__(master, bg=soft)
        tk.Frame(self, bg=color, width=3).pack(side="left", fill="y")
        body = tk.Frame(self, bg=soft, padx=12, pady=10)
        body.pack(side="left", fill="both", expand=True)
        self.label = WrapLabel(body, text=text, font=fonts.small, bg=soft, fg=TEXT_SECONDARY)
        self.label.pack(fill="x")
        if action:
            link = LinkLabel(body, fonts, text=action[0], command=action[1], bg=soft)
            link.pack(anchor="w", pady=(4, 0))


class LinkLabel(tk.Label):
    def __init__(self, master: tk.Misc, fonts: Fonts, *, text: str, command: Callable[[], None], bg: str, fg: str = ACCENT) -> None:
        super().__init__(master, text=text, font=fonts.small_strong, bg=bg, fg=fg, cursor="hand2", bd=0, takefocus=1)
        self._command = command
        self._fg = fg
        self.bind("<Button-1>", lambda _e: self._command())
        self.bind("<Return>", lambda _e: self._command())
        self.bind("<Enter>", lambda _e: self.configure(fg=ACCENT_PRESSED))
        self.bind("<Leave>", lambda _e: self.configure(fg=self._fg))


class ToggleSwitch(tk.Label):
    def __init__(
        self,
        master: tk.Misc,
        variable: tk.BooleanVar,
        assets: Assets,
        *,
        bg: str,
        command: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            master,
            bg=bg,
            bd=0,
            cursor="hand2",
            takefocus=1,
            highlightthickness=1,
            highlightbackground=bg,
            highlightcolor=ACCENT_SOFT_BORDER,
        )
        self._variable = variable
        self._assets = assets
        self._enabled = True
        self._command = command
        self._trace = variable.trace_add("write", lambda *_: self._refresh())
        self.bind("<Destroy>", lambda _e: self._remove_trace(), add="+")
        for sequence in ("<Button-1>", "<space>", "<Return>"):
            self.bind(sequence, self.toggle)
        self._refresh()

    def _remove_trace(self) -> None:
        try:
            self._variable.trace_remove("write", self._trace)
        except tk.TclError:
            pass

    def toggle(self, _event: tk.Event | None = None) -> str:
        if self._enabled:
            self._variable.set(not bool(self._variable.get()))
            if self._command:
                self._command()
        return "break"

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._refresh()

    def _refresh(self) -> None:
        on = bool(self._variable.get())
        image = self._assets.switch(on, self._enabled)
        if image is not None:
            self.configure(image=image)
        else:
            self.configure(text="On" if on else "Off", fg=ACCENT if on else MUTED, font=("Segoe UI", 9, "bold"))


class SettingRow(tk.Frame):
    """Title and description on the left with a control on the right."""

    def __init__(self, master: tk.Misc, fonts: Fonts, *, title: str, description: str | None = None, bg: str = SURFACE) -> None:
        super().__init__(master, bg=bg)
        self.columnconfigure(0, weight=1)
        self.title = tk.Label(self, text=title, font=fonts.body_strong, bg=bg, fg=TEXT, anchor="w")
        self.title.grid(row=0, column=0, sticky="w")
        self.description = None
        if description:
            self.description = WrapLabel(self, text=description, font=fonts.small, bg=bg, fg=MUTED)
            self.description.grid(row=1, column=0, sticky="ew", pady=(2, 0))

    def place_control(self, control: tk.Widget) -> tk.Widget:
        control.grid(row=0, column=1, rowspan=2, sticky="e", padx=(24, 0))
        return control

    def set_enabled(self, enabled: bool) -> None:
        self.title.configure(fg=TEXT if enabled else SUBTLE)
        if self.description is not None:
            self.description.configure(fg=MUTED if enabled else SUBTLE)


def switch_row(
    master: tk.Misc,
    fonts: Fonts,
    assets: Assets,
    *,
    title: str,
    description: str,
    variable: tk.BooleanVar,
    command: Callable[[], None] | None = None,
) -> tuple[SettingRow, ToggleSwitch]:
    row = SettingRow(master, fonts, title=title, description=description)
    switch = ToggleSwitch(row, variable, assets, bg=SURFACE, command=command)
    row.place_control(switch)
    row.title.configure(cursor="hand2")
    row.title.bind("<Button-1>", switch.toggle)
    return row, switch


class SegmentedControl(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        fonts: Fonts,
        *,
        variable: tk.StringVar,
        options: Sequence[tuple[str, str]],
        command: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, bg=BORDER, padx=1, pady=1)
        inner = tk.Frame(self, bg=TRACK, padx=3, pady=3)
        inner.pack(fill="both", expand=True)
        self._variable = variable
        self._command = command
        self._enabled = True
        self._hover: str | None = None
        self._segments: dict[str, tk.Label] = {}
        for index, (value, label) in enumerate(options):
            segment = tk.Label(
                inner,
                text=label,
                font=fonts.small_strong,
                padx=14,
                pady=5,
                bd=0,
                highlightthickness=1,
                cursor="hand2",
                takefocus=1,
            )
            segment.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 2, 0))
            for sequence in ("<Button-1>", "<space>", "<Return>"):
                segment.bind(sequence, lambda _e, v=value: self._select(v))
            segment.bind("<Enter>", lambda _e, v=value: self._set_hover(v))
            segment.bind("<Leave>", lambda _e: self._set_hover(None))
            self._segments[value] = segment
        self._trace = variable.trace_add("write", lambda *_: self._refresh())
        self.bind("<Destroy>", lambda _e: self._remove_trace(), add="+")
        self._refresh()

    def _remove_trace(self) -> None:
        try:
            self._variable.trace_remove("write", self._trace)
        except tk.TclError:
            pass

    def _select(self, value: str) -> str:
        if self._enabled and self._variable.get() != value:
            self._variable.set(value)
            if self._command:
                self._command()
        return "break"

    def _set_hover(self, value: str | None) -> None:
        self._hover = value
        self._refresh()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        for segment in self._segments.values():
            segment.configure(cursor="hand2" if enabled else "arrow", takefocus=1 if enabled else 0)
        self._refresh()

    def _refresh(self) -> None:
        current = self._variable.get()
        for value, segment in self._segments.items():
            selected = value == current
            if selected:
                bg, fg, border = (SURFACE, TEXT, BORDER) if self._enabled else (SURFACE_ALT, SUBTLE, BORDER)
            else:
                bg, border = TRACK, TRACK
                fg = SUBTLE if not self._enabled else TEXT if value == self._hover else MUTED
            segment.configure(bg=bg, fg=fg, highlightbackground=border, highlightcolor=ACCENT_SOFT_BORDER)


class OptionCard(tk.Frame):
    """Large selectable card used for picking the translator."""

    def __init__(
        self,
        master: tk.Misc,
        fonts: Fonts,
        assets: Assets,
        *,
        variable: tk.StringVar,
        value: str,
        title: str,
        description: str,
        tag: str | None = None,
        tag_level: str = "neutral",
        command: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER, cursor="hand2", takefocus=1)
        self._variable = variable
        self._value = value
        self._assets = assets
        self._command = command
        self._hover = False
        self.columnconfigure(1, weight=1)
        self._indicator = tk.Label(self, bd=0, bg=SURFACE, text="", font=fonts.small)
        self._indicator.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(14, 10), pady=(14, 0))
        top = tk.Frame(self, bg=SURFACE)
        top.grid(row=0, column=1, sticky="ew", padx=(0, 14), pady=(12, 0))
        self._title = tk.Label(top, text=title, font=fonts.body_strong, bg=SURFACE, fg=TEXT, anchor="w")
        self._title.pack(side="left")
        self._tag = badge(top, fonts, tag, tag_level) if tag else None
        if self._tag is not None:
            self._tag.pack(side="left", padx=(8, 0))
        self._description = WrapLabel(self, text=description, font=fonts.small, bg=SURFACE, fg=MUTED, wraplength=220)
        self._description.grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=(3, 13))
        self._backgrounds = [self, self._indicator, top, self._title, self._description]
        bind_tree(self, "<Button-1>", self._select)
        bind_tree(self, "<Enter>", lambda _e: self._set_hover(True))
        bind_tree(self, "<Leave>", self._on_leave)
        for sequence in ("<space>", "<Return>"):
            self.bind(sequence, self._select)
        self.bind("<FocusIn>", lambda _e: self._refresh(), add="+")
        self.bind("<FocusOut>", lambda _e: self._refresh(), add="+")
        self._trace = variable.trace_add("write", lambda *_: self._refresh())
        self.bind("<Destroy>", lambda _e: self._remove_trace(), add="+")
        self._refresh()

    def _remove_trace(self) -> None:
        try:
            self._variable.trace_remove("write", self._trace)
        except tk.TclError:
            pass

    def _select(self, _event: tk.Event | None = None) -> str:
        if self._variable.get() != self._value:
            self._variable.set(self._value)
            if self._command:
                self._command()
        return "break"

    def _set_hover(self, hover: bool) -> None:
        if self._hover != hover:
            self._hover = hover
            self._refresh()

    def _on_leave(self, _event: tk.Event) -> None:
        pointer = self.winfo_containing(*self.winfo_pointerxy())
        self._set_hover(is_descendant(pointer, self))

    def _refresh(self) -> None:
        selected = self._variable.get() == self._value
        bg = ACCENT_SOFT if selected else SURFACE_ALT if self._hover else SURFACE
        border = ACCENT if selected else BORDER_STRONG if self._hover or self.focus_get() is self else BORDER
        for widget in self._backgrounds:
            widget.configure(bg=bg)
        self.configure(highlightbackground=border, highlightcolor=border)
        image = self._assets.radio(selected)
        if image is not None:
            self._indicator.configure(image=image)
        else:
            self._indicator.configure(text="●" if selected else "○", fg=ACCENT if selected else SUBTLE)


class NavItem(tk.Frame):
    def __init__(self, master: tk.Misc, fonts: Fonts, *, icon: str, text: str, command: Callable[[], None]) -> None:
        super().__init__(master, bg=SIDEBAR_BG, cursor="hand2")
        self._command = command
        self._active = False
        self._hover = False
        self._bar = tk.Frame(self, bg=SIDEBAR_BG, width=3)
        self._bar.pack(side="left", fill="y")
        self._inner = tk.Frame(self, bg=SIDEBAR_BG, padx=12, pady=9)
        self._inner.pack(side="left", fill="both", expand=True)
        self._icon = tk.Label(self._inner, text=icon if fonts.icon else "", font=fonts.icon or fonts.body, bg=SIDEBAR_BG, fg=SIDEBAR_TEXT, width=2, anchor="w")
        if fonts.icon:
            self._icon.pack(side="left", padx=(0, 8))
        self._label = tk.Label(self._inner, text=text, font=fonts.body_strong, bg=SIDEBAR_BG, fg=SIDEBAR_TEXT, anchor="w")
        self._label.pack(side="left", fill="x", expand=True)
        self._dot = tk.Label(self._inner, text="●", font=fonts.small, bg=SIDEBAR_BG, fg=LOG_WARNING)
        bind_tree(self, "<Button-1>", lambda _e: self._command())
        bind_tree(self, "<Enter>", lambda _e: self._set_hover(True))
        bind_tree(self, "<Leave>", self._on_leave)

    def _on_leave(self, _event: tk.Event) -> None:
        pointer = self.winfo_containing(*self.winfo_pointerxy())
        self._set_hover(is_descendant(pointer, self))

    def _set_hover(self, hover: bool) -> None:
        self._hover = hover
        self._refresh()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._refresh()

    def set_attention(self, level: str | None) -> None:
        if level is None:
            self._dot.pack_forget()
            return
        self._dot.configure(fg=LOG_DANGER if level == "error" else LOG_WARNING)
        self._dot.pack(side="right")

    def _refresh(self) -> None:
        bg = SIDEBAR_ACTIVE if self._active else SIDEBAR_HOVER if self._hover else SIDEBAR_BG
        for widget in (self, self._inner, self._icon, self._label, self._dot):
            widget.configure(bg=bg)
        self._bar.configure(bg=ACCENT if self._active else bg)
        self._label.configure(fg=SIDEBAR_TEXT_ACTIVE if self._active or self._hover else SIDEBAR_TEXT)
        self._icon.configure(fg=SIDEBAR_ICON_ACTIVE if self._active else SIDEBAR_TEXT_ACTIVE if self._hover else SIDEBAR_TEXT)


class PathField(tk.Frame):
    """Labelled entry with browse buttons and an optional hint."""

    def __init__(
        self,
        master: tk.Misc,
        fonts: Fonts,
        *,
        label: str,
        variable: tk.StringVar,
        buttons: Sequence[tuple[str, Callable[[], None]]],
        hint: str | None = None,
        optional: bool = False,
        bg: str = SURFACE,
    ) -> None:
        super().__init__(master, bg=bg)
        header = tk.Frame(self, bg=bg)
        header.pack(fill="x", pady=(0, 6))
        self._label = tk.Label(header, text=label, font=fonts.body_strong, bg=bg, fg=TEXT, anchor="w")
        self._label.pack(side="left")
        if optional:
            tk.Label(header, text="Optional", font=fonts.small, bg=bg, fg=SUBTLE).pack(side="left", padx=(8, 0))
        row = tk.Frame(self, bg=bg)
        row.pack(fill="x")
        row.columnconfigure(0, weight=1)
        self.entry = ttk.Entry(row, textvariable=variable)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.buttons: list[ttk.Button] = []
        for index, (text, command) in enumerate(buttons):
            button = ttk.Button(row, text=text, command=command)
            button.grid(row=0, column=index + 1, sticky="ns", padx=(8, 0))
            self.buttons.append(button)
        self._hint = None
        if hint:
            self._hint = WrapLabel(self, text=hint, font=fonts.small, bg=bg, fg=MUTED)
            self._hint.pack(fill="x", pady=(6, 0))

    def set_enabled(self, enabled: bool) -> None:
        state = "!disabled" if enabled else "disabled"
        self.entry.state([state])
        for button in self.buttons:
            button.state([state])
        self._label.configure(fg=TEXT if enabled else SUBTLE)


class SliderField(tk.Frame):
    """Labelled slider with a live value readout."""

    def __init__(
        self,
        master: tk.Misc,
        fonts: Fonts,
        assets: Assets,
        *,
        label: str,
        description: str,
        variable: tk.Variable,
        minimum: float,
        maximum: float,
        step: float,
        formatter: Callable[[float], str],
        bg: str = SURFACE,
    ) -> None:
        super().__init__(master, bg=bg)
        self._variable = variable
        self._step = step
        self._minimum = minimum
        self._formatter = formatter
        self.columnconfigure(0, weight=1)
        tk.Label(self, text=label, font=fonts.body_strong, bg=bg, fg=TEXT, anchor="w").grid(row=0, column=0, sticky="w")
        self._value = tk.Label(self, font=fonts.body_strong, bg=ACCENT_SOFT, fg=ACCENT, padx=8, pady=1)
        self._value.grid(row=0, column=1, sticky="e")
        WrapLabel(self, text=description, font=fonts.small, bg=bg, fg=MUTED).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 8))
        self.slider = Slider(
            self,
            assets,
            minimum=minimum,
            maximum=maximum,
            step=step,
            value=self._read(),
            on_change=self._on_slide,
            bg=bg,
        )
        self.slider.grid(row=2, column=0, columnspan=2, sticky="ew")
        ticks = tk.Frame(self, bg=bg)
        ticks.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        tk.Label(ticks, text=formatter(minimum), font=fonts.small, bg=bg, fg=SUBTLE).pack(side="left")
        tk.Label(ticks, text=formatter(maximum), font=fonts.small, bg=bg, fg=SUBTLE).pack(side="right")
        self._trace = variable.trace_add("write", lambda *_: self._sync_from_variable())
        self.bind("<Destroy>", lambda _e: self._remove_trace(), add="+")
        self._sync_from_variable()

    def _remove_trace(self) -> None:
        try:
            self._variable.trace_remove("write", self._trace)
        except tk.TclError:
            pass

    def _read(self) -> float:
        try:
            return float(self._variable.get())
        except (tk.TclError, ValueError):
            return self._minimum

    def _snap(self, value: float) -> float:
        snapped = self._minimum + round((value - self._minimum) / self._step) * self._step
        return round(snapped, 4)

    def _on_slide(self, raw: float) -> None:
        value = self._snap(raw)
        if isinstance(self._variable, tk.IntVar):
            value = int(round(value))
        if self._read() != value:
            self._variable.set(value)

    def _sync_from_variable(self) -> None:
        value = self._read()
        self.slider.set(value)
        self._value.configure(text=self._formatter(value))


class Slider(tk.Canvas):
    """Anti-aliased horizontal slider drawn on a canvas."""

    def __init__(
        self,
        master: tk.Misc,
        assets: Assets,
        *,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        on_change: Callable[[float], None],
        bg: str = SURFACE,
    ) -> None:
        self._assets = assets
        self._height = assets._px(26)
        super().__init__(master, height=self._height, bg=bg, highlightthickness=0, bd=0, cursor="hand2", takefocus=1)
        self._minimum = minimum
        self._maximum = maximum
        self._step = step
        self._value = value
        self._on_change = on_change
        self._focused = False
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", lambda event: self._on_change(self._value_at(event.x)))
        for sequence, direction in (("<Left>", -1), ("<Down>", -1), ("<Right>", 1), ("<Up>", 1)):
            self.bind(sequence, lambda _e, d=direction: self._nudge(d))
        self.bind("<FocusIn>", lambda _e: self._set_focused(True))
        self.bind("<FocusOut>", lambda _e: self._set_focused(False))

    def set(self, value: float) -> None:
        self._value = value
        self._draw()

    def _set_focused(self, focused: bool) -> None:
        self._focused = focused
        self._draw()

    def _span(self) -> tuple[float, float]:
        radius = self._assets._px(11)
        width = max(self.winfo_width(), 2 * radius + 1)
        return radius, width - radius

    def _clamp(self, value: float) -> float:
        return max(self._minimum, min(self._maximum, value))

    def _value_at(self, x: float) -> float:
        start, end = self._span()
        fraction = (x - start) / max(1.0, end - start)
        return self._clamp(self._minimum + fraction * (self._maximum - self._minimum))

    def _on_press(self, event: tk.Event) -> str:
        self.focus_set()
        self._on_change(self._value_at(event.x))
        return "break"

    def _nudge(self, direction: int) -> str:
        self._on_change(self._clamp(self._value + direction * self._step))
        return "break"

    def _draw(self) -> None:
        self.delete("all")
        start, end = self._span()
        middle = self._height / 2
        span = max(1e-9, self._maximum - self._minimum)
        x = start + (self._clamp(self._value) - self._minimum) / span * (end - start)
        thickness = self._assets._px(4)
        self.create_line(start, middle, end, middle, width=thickness, fill=BORDER, capstyle="round")
        if x > start:
            self.create_line(start, middle, x, middle, width=thickness, fill=ACCENT, capstyle="round")
        thumb = self._assets.slider_thumb(self._focused)
        if thumb is not None:
            self.create_image(x, middle, image=thumb)
        else:
            radius = self._assets._px(8)
            self.create_oval(x - radius, middle - radius, x + radius, middle + radius, fill=SURFACE, outline=ACCENT, width=2)
