# spectroview/view/theme/icons.py
"""Central icon facade and legacy mapping for SPECTROview.

Provides static `Icons` factory methods returning vector QIcons and QPixmaps
backed by `_SvgIconEngine` with dynamic theme resolution, plus `LEGACY_ICON_MAP`
to translate legacy raster icon names into vector SVG keys.
"""

from __future__ import annotations

from PySide6.QtGui import QIcon, QPixmap
from .icons_svg import svg_icon, svg_pixmap

# ---------------------------------------------------------------------------
# Legacy icon name mapping
# ---------------------------------------------------------------------------
LEGACY_ICON_MAP: dict[str, str] = {
    # File & Workspace
    "open": "open",
    "open.png": "open",
    "load": "open",
    "load.png": "open",
    "load_color": "open",
    "load_color.png": "open",
    "save": "save",
    "save.png": "save",
    "save-94": "save",
    "save-94.png": "save",
    "save12": "save",
    "save12.png": "save",
    "save_color": "save",
    "save_color.png": "save",
    "save-as": "save_as",
    "save-as.png": "save_as",
    "save-all": "save_all",
    "save-all.png": "save_all",
    "save-recipe": "recipe",
    "save-recipe.png": "recipe",
    "recipe": "recipe",
    "recipe.png": "recipe",
    "clear": "clear",
    "clear.png": "clear",
    "fileconvert": "convert",
    "fileconvert.png": "convert",
    "calc": "calc",
    "calc.png": "calc",
    "settings": "settings",
    "settings.png": "settings",
    "options": "settings",
    "options.png": "settings",
    "check_update": "bell",
    "check_update.png": "bell",
    "update": "refresh",
    "update.png": "refresh",
    "dark-light": "sun",
    "dark-light.png": "sun",
    "dark": "moon",
    "dark.png": "moon",
    "manual": "book",
    "manual.png": "book",
    "github": "github",
    "github.png": "github",
    "github1": "github",
    "github1.png": "github",
    "about": "info",
    "about.png": "info",
    "support": "help_circle",
    "support.png": "help_circle",

    # Edit & Clipboard
    "trash": "trash",
    "trash.png": "trash",
    "trash2": "trash",
    "trash2.png": "trash",
    "trash3": "trash",
    "trash3.png": "trash",
    "eraser": "eraser",
    "eraser.png": "eraser",
    "edit": "edit",
    "edit.png": "edit",
    "copy": "copy",
    "copy.png": "copy",
    "copy2": "copy",
    "copy2.png": "copy",
    "copy3": "copy",
    "copy3.png": "copy",
    "copy_label": "copy",
    "copy_label.png": "copy",
    "copy_label.png.png": "copy",
    "cut": "cut",
    "cut.png": "cut",
    "paste": "paste",
    "paste.png": "paste",
    "paste3": "paste",
    "paste3.png": "paste",
    "duplicate": "duplicate",
    "duplicate.png": "duplicate",
    "duplicate2": "duplicate",
    "duplicate2.png": "duplicate",
    "replicate": "duplicate",
    "replicate.png": "duplicate",
    "undo": "undo",
    "undo.png": "undo",
    "undo2": "undo",
    "undo2.png": "undo",
    "redo": "redo",
    "redo.png": "redo",
    "redo2": "redo",
    "redo2.png": "redo",
    "refresh": "refresh",
    "refresh.png": "refresh",
    "filter": "filter",
    "filter.png": "filter",
    "export": "export",
    "export.png": "export",
    "export_all": "export_all",
    "export_all.png": "export_all",
    "close": "close",
    "close.png": "close",
    "remove": "close",
    "remove.png": "close",
    "add": "plus",
    "add.png": "plus",
    "add_color": "plus",
    "add_color.png": "plus",
    "done": "check",
    "done.png": "check",
    "select-all": "check_circle",
    "select-all.png": "check_circle",
    "compose": "edit",
    "compose.png": "edit",
    "multipanel": "multipanel",
    "multipanel.png": "multipanel",
    "collect": "database",
    "collect.png": "database",

    # Spectroscopy & Hyperspectral
    "spectrum_1d": "spectrum_1d",
    "spectrum": "spectrum_1d",
    "2dmap": "map_2d",
    "2dmap.png": "map_2d",
    "wafer": "wafer_map",
    "wafer.png": "wafer_map",
    "wafer_map": "wafer_map",
    "wafer_map.png": "wafer_map",
    "baseline": "baseline",
    "baseline.png": "baseline",
    "peak": "peak",
    "peak.png": "peak",
    "bestfit": "bestfit",
    "bestfit.png": "bestfit",
    "norm": "norm",
    "norm.png": "norm",
    "stats": "stats",
    "stats.png": "stats",

    # Graphing
    "line": "graph_line",
    "line.png": "graph_line",
    "scatter": "graph_scatter",
    "scatter.png": "graph_scatter",
    "point": "graph_point",
    "point.png": "graph_point",

    # Graph Annotations & Scientific
    "vline": "ann_vline",
    "vline.png": "ann_vline",
    "hline": "ann_hline",
    "hline.png": "ann_hline",
    "text": "ann_text",
    "text.png": "ann_text",
    "arrow": "ann_arrow",
    "arrow.png": "ann_arrow",
    "arrow-up": "chevron_up",
    "arrow-up.svg": "chevron_up",
    "arrow-down": "chevron_down",
    "arrow-down.svg": "chevron_down",
    "vspan": "ann_vspan",
    "vspan.png": "ann_vspan",
    "hspan": "ann_hspan",
    "hspan.png": "ann_hspan",
    "box_ann": "ann_box",
    "box_ann.png": "ann_box",
    "callout": "ann_callout",
    "callout.png": "ann_callout",
    "bar": "graph_bar",
    "bar.png": "graph_bar",
    "box": "graph_box",
    "box.png": "graph_box",
    "histogram": "graph_histogram",
    "histogram.png": "graph_histogram",
    "trendline": "graph_trendline",
    "trendline.png": "graph_trendline",
    "styling": "styling",
    "styling.png": "styling",
    "customize": "styling",
    "customize.png": "styling",
    "customize2": "styling",
    "customize2.png": "styling",
    "legend": "legend",
    "legend.png": "legend",
    "zoom": "zoom",
    "zoom.png": "zoom",
    "magnify": "zoom",
    "magnify.png": "zoom",
    "magnify2": "zoom",
    "magnify2.png": "zoom",
    "pan": "pan",
    "pan.png": "pan",
    "pan2": "pan",
    "pan2.png": "pan",
    "rescale": "rescale",
    "rescale.png": "rescale",
    "rescale2": "rescale",
    "rescale2.png": "rescale",
    "minimize_windows": "minimize_windows",
    "minimize_windows.png": "minimize_windows",
    "minimize_windows2": "minimize_windows",
    "minimize_windows2.png": "minimize_windows",

    # AI Chat & Extras
    "llm_ai": "sparkles",
    "llm_ai.png": "sparkles",
    "llm_ai.svg": "sparkles",
    "ai_chat": "sparkles",
    "ai_chat.png": "sparkles",
    "send": "send",
    "send.png": "send",
    "stop": "stop",
    "stop.png": "stop",
    "view-details": "history",
    "view-details.png": "history",
    "view": "history",
    "view.png": "history",
    "view_color": "history",
    "view_color.png": "history",
}


# ---------------------------------------------------------------------------
# Icons facade class
# ---------------------------------------------------------------------------
class Icons:
    """Convenience facade for retrieving themed vector QIcons and QPixmaps."""

    @staticmethod
    def get(name: str, size: int = 16, color: str | None = None) -> QIcon:
        """Return a vector QIcon by vector key or legacy filename."""
        canonical = LEGACY_ICON_MAP.get(name.lower(), name)
        return svg_icon(canonical, size=size, color=color)

    @staticmethod
    def pixmap(name: str, size: int = 16, color: str | None = None, dpr: float = 2.0) -> QPixmap:
        """Return a crisp vector QPixmap by vector key or legacy filename."""
        canonical = LEGACY_ICON_MAP.get(name.lower(), name)
        return svg_pixmap(canonical, size=size, color=color, dpr=dpr)

    # ── Common UI & Navigation ──
    @classmethod
    def open(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("open", size=size, color=color)

    @classmethod
    def save(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("save", size=size, color=color)

    @classmethod
    def save_all(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("save_all", size=size, color=color)

    @classmethod
    def save_as(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("save_as", size=size, color=color)

    @classmethod
    def clear(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("clear", size=size, color=color)

    @classmethod
    def convert(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("convert", size=size, color=color)

    @classmethod
    def calc(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("calc", size=size, color=color)

    @classmethod
    def settings(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("settings", size=size, color=color)

    @classmethod
    def check_update(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("bell", size=size, color=color)

    @classmethod
    def theme(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("sun", size=size, color=color)

    @classmethod
    def manual(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("book", size=size, color=color)

    @classmethod
    def github(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("github", size=size, color=color)

    @classmethod
    def about(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("info", size=size, color=color)

    @classmethod
    def folder(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("folder", size=size, color=color)

    @classmethod
    def trash(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("trash", size=size, color=color)

    @classmethod
    def eraser(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("eraser", size=size, color=color)

    @classmethod
    def edit(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("edit", size=size, color=color)

    @classmethod
    def copy(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("copy", size=size, color=color)

    @classmethod
    def cut(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("cut", size=size, color=color)

    @classmethod
    def paste(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("paste", size=size, color=color)

    @classmethod
    def duplicate(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("duplicate", size=size, color=color)

    @classmethod
    def undo(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("undo", size=size, color=color)

    @classmethod
    def redo(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("redo", size=size, color=color)

    @classmethod
    def reply(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("reply", size=size, color=color)

    @classmethod
    def refresh(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("refresh", size=size, color=color)

    @classmethod
    def close(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("close", size=size, color=color)

    @classmethod
    def plus(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("plus", size=size, color=color)

    @classmethod
    def plus_circle(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("plus_circle", size=size, color=color)

    @classmethod
    def check(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("check", size=size, color=color)

    @classmethod
    def check_circle(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("check_circle", size=size, color=color)

    @classmethod
    def search(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("search", size=size, color=color)

    @classmethod
    def filter(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("filter", size=size, color=color)

    @classmethod
    def export(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("export", size=size, color=color)

    @classmethod
    def export_all(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("export_all", size=size, color=color)

    @classmethod
    def download(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("download", size=size, color=color)

    @classmethod
    def upload(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("upload", size=size, color=color)

    @classmethod
    def sliders(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("sliders", size=size, color=color)

    @classmethod
    def multipanel(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("multipanel", size=size, color=color)

    # ── Scientific Spectroscopy ──
    @classmethod
    def spectrum_1d(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("spectrum_1d", size=size, color=color)

    @classmethod
    def map_2d(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("map_2d", size=size, color=color)

    @classmethod
    def wafer_map(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("wafer_map", size=size, color=color)

    @classmethod
    def baseline(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("baseline", size=size, color=color)

    @classmethod
    def peak(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("peak", size=size, color=color)

    @classmethod
    def bestfit(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("bestfit", size=size, color=color)

    @classmethod
    def norm(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("norm", size=size, color=color)

    @classmethod
    def stats(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("stats", size=size, color=color)

    # ── Graphing ──
    @classmethod
    def graph_line(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("graph_line", size=size, color=color)

    @classmethod
    def graph_scatter(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("graph_scatter", size=size, color=color)

    @classmethod
    def graph_point(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("graph_point", size=size, color=color)

    # ── Graph Annotations ──
    @classmethod
    def ann_vline(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("ann_vline", size=size, color=color)

    @classmethod
    def ann_hline(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("ann_hline", size=size, color=color)

    @classmethod
    def ann_text(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("ann_text", size=size, color=color)

    @classmethod
    def ann_arrow(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("ann_arrow", size=size, color=color)

    @classmethod
    def ann_vspan(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("ann_vspan", size=size, color=color)

    @classmethod
    def ann_hspan(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("ann_hspan", size=size, color=color)

    @classmethod
    def ann_box(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("ann_box", size=size, color=color)

    @classmethod
    def ann_callout(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("ann_callout", size=size, color=color)

    @classmethod
    def graph_bar(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("graph_bar", size=size, color=color)

    @classmethod
    def graph_box(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("graph_box", size=size, color=color)

    @classmethod
    def graph_histogram(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("graph_histogram", size=size, color=color)

    @classmethod
    def graph_trendline(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("graph_trendline", size=size, color=color)

    @classmethod
    def styling(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("styling", size=size, color=color)

    @classmethod
    def legend(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("legend", size=size, color=color)

    @classmethod
    def zoom(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("zoom", size=size, color=color)

    @classmethod
    def pan(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("pan", size=size, color=color)

    @classmethod
    def rescale(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("rescale", size=size, color=color)

    @classmethod
    def minimize_windows(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("minimize_windows", size=size, color=color)

    # ── AI Chat & Extras ──
    @classmethod
    def ai_chat(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("sparkles", size=size, color=color)

    @classmethod
    def sparkles(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("sparkles", size=size, color=color)

    @classmethod
    def brain(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("brain", size=size, color=color)

    @classmethod
    def send(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("send", size=size, color=color)

    @classmethod
    def stop(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("stop", size=size, color=color)

    @classmethod
    def play(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("play", size=size, color=color)

    @classmethod
    def mic(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("mic", size=size, color=color)

    @classmethod
    def paperclip(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("paperclip", size=size, color=color)

    @classmethod
    def history(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("history", size=size, color=color)

    @classmethod
    def notification(cls, color: str | None = None, size: int = 16) -> QIcon:
        return svg_icon("bell", size=size, color=color)
