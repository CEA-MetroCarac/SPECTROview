"""Export dialogs for the Graph Workspace: save one graph (VExportDialog) or
every open graph (VBatchExportDialog) to PNG/TIFF/SVG/PDF/EPS with DPI,
transparent-background, and an export-time theme override.

Format/DPI/transparent/theme are per-export-action choices (a user may
reasonably export the same graph multiple times at different settings), so
they're persisted as app-wide preferences via MSettings (remembered across
dialog opens), not as MGraph fields. Physical export size, by contrast,
behaves like plot_width/plot_height -- a per-graph property a user sets once
(especially via a journal preset) and expects to stick -- so it lives on
MGraph/VGraph as export_width_mm/export_height_mm.
"""
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox, QWidget,
)

from spectroview.model.m_settings import MSettings
from spectroview.viewmodel.utils import export_figure_to_file

# Display text -> (file extension, QFileDialog filter string)
_FORMAT_EXTENSIONS = {
    "PNG": ("png", "PNG Image (*.png)"),
    "TIFF": ("tiff", "TIFF Image (*.tiff *.tif)"),
    "SVG": ("svg", "SVG Vector (*.svg)"),
    "PDF": ("pdf", "PDF Document (*.pdf)"),
    "EPS": ("eps", "EPS Vector (*.eps)"),
}

# Display text <-> MGraph.figure_theme value -- matches the display-name
# convention already used by view_options/theme and view_options/copy_fig_theme.
_THEME_DISPLAY_TO_KEY = {"Light Mode": "light", "Dark Mode": "dark", "Soft Dark Mode": "soft_dark"}
_THEME_KEY_TO_DISPLAY = {v: k for k, v in _THEME_DISPLAY_TO_KEY.items()}

_MM_PER_INCH = 25.4

# Approximate, adjustable starting points -- not authoritative journal specs.
# (width_mm, height_mm, dpi)
_JOURNAL_PRESETS = {
    "Custom": None,
    "Nature - single column (89 mm)": (89.0, 65.0, 300),
    "Nature - double column (183 mm)": (183.0, 90.0, 300),
    "Science - single column (55 mm)": (55.0, 45.0, 300),
    "Generic - half page (90x68mm, 300dpi)": (90.0, 68.0, 300),
}


def export_with_theme_override(graph_widget, filepath, fmt, dpi, transparent, theme_key, size_inches=None):
    """Temporarily swap a graph widget to the requested theme, replot,
    export, restore -- mirrors the exact swap-render-restore pattern already
    used by v_spectra_viewer.py's `_emit_copy()` / v_map_viewer.py's
    copy-to-clipboard theme override. A no-op swap (theme already matches
    the graph's own figure_theme, the common case) skips the extra replot
    entirely. Module-level (not a dialog method) so both the single-graph
    and batch export dialogs share one implementation.
    """
    original_theme = graph_widget.figure_theme
    needs_swap = theme_key != original_theme

    if needs_swap:
        graph_widget.figure_theme = theme_key
        if graph_widget.df is not None:
            graph_widget.plot(graph_widget.df)

    try:
        return export_figure_to_file(
            graph_widget.canvas, filepath, fmt, dpi=dpi, transparent=transparent,
            size_inches=size_inches,
        )
    finally:
        if needs_swap:
            graph_widget.figure_theme = original_theme
            if graph_widget.df is not None:
                graph_widget.plot(graph_widget.df)


class _ExportSettingsPanel(QWidget):
    """Format/DPI/transparent-background/theme controls, shared by
    VExportDialog (single graph, plus a physical-size section) and
    VBatchExportDialog (every open graph, format/DPI/transparent/theme only
    -- physical size stays per-graph, read from each graph's own
    export_width_mm/export_height_mm during the batch loop)."""

    def __init__(self, default_theme_key: str, parent=None):
        super().__init__(parent)
        self.settings = MSettings()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(list(_FORMAT_EXTENSIONS.keys()))
        fmt_row.addWidget(self.combo_format)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("DPI:"))
        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(72, 1200)
        self.spin_dpi.setSingleStep(50)
        dpi_row.addWidget(self.spin_dpi)
        note = QLabel("(export resolution -- separate from the on-screen canvas DPI)")
        note.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        dpi_row.addWidget(note)
        dpi_row.addStretch()
        layout.addLayout(dpi_row)

        self.cb_transparent = QCheckBox("Transparent background")
        layout.addWidget(self.cb_transparent)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme:"))
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(list(_THEME_DISPLAY_TO_KEY.keys()))
        theme_row.addWidget(self.combo_theme)
        theme_row.addStretch()
        layout.addLayout(theme_row)

        opts = self.settings.load_export_options()
        idx = self.combo_format.findText(opts["format"].upper())
        if idx >= 0:
            self.combo_format.setCurrentIndex(idx)
        self.spin_dpi.setValue(opts["dpi"])
        self.cb_transparent.setChecked(opts["transparent"])
        self.combo_theme.setCurrentText(
            _THEME_KEY_TO_DISPLAY.get(default_theme_key, "Light Mode")
        )

    def format_ext(self) -> str:
        return _FORMAT_EXTENSIONS[self.combo_format.currentText()][0]

    def file_filter(self) -> str:
        return _FORMAT_EXTENSIONS[self.combo_format.currentText()][1]

    def theme_key(self) -> str:
        return _THEME_DISPLAY_TO_KEY[self.combo_theme.currentText()]

    def save_as_last_used(self):
        self.settings.save_export_options({
            "format": self.format_ext(), "dpi": self.spin_dpi.value(),
            "transparent": self.cb_transparent.isChecked(),
            "theme": self.combo_theme.currentText(),
        })


class VExportDialog(QDialog):
    """Dialog for exporting a single graph to an image/vector file."""

    # Sentinel for "no override" on the physical-size spinboxes -- blank
    # means "use the figure's current on-screen size at export time" (mirrors
    # the sentinel-spinbox pattern used throughout the Customize dialog).
    _UNSET = -999999.0

    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self._unit = "mm"  # canonical storage is always mm; this only affects display

        self.setWindowTitle(f"Export Graph {graph_widget.graph_id}")
        self.setModal(True)
        self.resize(420, 340)

        self._setup_ui()
        self._load_physical_size()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.panel = _ExportSettingsPanel(getattr(self.graph_widget, 'figure_theme', 'light'))
        layout.addWidget(self.panel)
        self.settings = self.panel.settings

        # Physical size -- blank (default) uses the figure's current
        # on-screen size; journal preset selection fills width/height/DPI.
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Journal preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(_JOURNAL_PRESETS.keys()))
        self.combo_preset.currentTextChanged.connect(self._on_preset_selected)
        preset_row.addWidget(self.combo_preset)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Width:"))
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(self._UNSET, 2000)
        self.spin_width.setSpecialValueText(" ")
        self.spin_width.setDecimals(1)
        size_row.addWidget(self.spin_width)

        size_row.addWidget(QLabel("Height:"))
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(self._UNSET, 2000)
        self.spin_height.setSpecialValueText(" ")
        self.spin_height.setDecimals(1)
        size_row.addWidget(self.spin_height)

        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["mm", "in"])
        self.combo_unit.currentTextChanged.connect(self._on_unit_changed)
        size_row.addWidget(self.combo_unit)

        note = QLabel("(blank = use current on-screen size)")
        note.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        size_row.addWidget(note)
        size_row.addStretch()
        layout.addLayout(size_row)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        self.btn_export = QPushButton("Export...")
        self.btn_export.clicked.connect(self._on_export_clicked)
        btn_row.addWidget(self.btn_export)
        layout.addLayout(btn_row)

    def _load_physical_size(self):
        """From the graph's own persisted fields (mm, the canonical storage
        unit), None = blank/unset."""
        width_mm = getattr(self.graph_widget, 'export_width_mm', None)
        height_mm = getattr(self.graph_widget, 'export_height_mm', None)
        self.spin_width.setValue(width_mm if width_mm is not None else self._UNSET)
        self.spin_height.setValue(height_mm if height_mm is not None else self._UNSET)

    def _on_preset_selected(self, preset_name: str):
        preset = _JOURNAL_PRESETS.get(preset_name)
        if preset is None:  # "Custom" -- leave current values untouched
            return
        width_mm, height_mm, dpi = preset
        if self._unit == "in":
            self.spin_width.setValue(width_mm / _MM_PER_INCH)
            self.spin_height.setValue(height_mm / _MM_PER_INCH)
        else:
            self.spin_width.setValue(width_mm)
            self.spin_height.setValue(height_mm)
        self.panel.spin_dpi.setValue(dpi)

    def _on_unit_changed(self, new_unit: str):
        """Convert currently-displayed width/height so they still represent
        the same physical size after switching mm<->in (not just resetting
        the spinboxes to the new unit's raw numbers)."""
        old_unit = self._unit
        self._unit = new_unit
        if old_unit == new_unit:
            return
        factor = _MM_PER_INCH if new_unit == "mm" else (1 / _MM_PER_INCH)
        for spin in (self.spin_width, self.spin_height):
            if spin.value() != self._UNSET:
                spin.setValue(spin.value() * factor)

    def _width_height_mm(self):
        """Current width/height spinbox values converted to mm, or (None, None)
        if either is unset."""
        if self.spin_width.value() == self._UNSET or self.spin_height.value() == self._UNSET:
            return None, None
        factor = 1.0 if self._unit == "mm" else _MM_PER_INCH
        return self.spin_width.value() * factor, self.spin_height.value() * factor

    def _on_export_clicked(self):
        ext, file_filter = self.panel.format_ext(), self.panel.file_filter()

        last_dir = self.settings.get_last_directory()
        default_name = f"graph_{self.graph_widget.graph_id}.{ext}"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Graph", str(Path(last_dir) / default_name), file_filter
        )
        if not filepath:
            return
        if not filepath.lower().endswith(f".{ext}"):
            filepath += f".{ext}"

        dpi = self.panel.spin_dpi.value()
        transparent = self.panel.cb_transparent.isChecked()
        theme_key = self.panel.theme_key()

        width_mm, height_mm = self._width_height_mm()
        size_inches = (width_mm / _MM_PER_INCH, height_mm / _MM_PER_INCH) if width_mm else None

        success = export_with_theme_override(
            self.graph_widget, filepath, ext, dpi, transparent, theme_key, size_inches=size_inches
        )

        if success:
            self.panel.save_as_last_used()
            self.settings.set_last_directory(str(Path(filepath).parent))

            gw = self.graph_widget
            if (width_mm, height_mm) != (gw.export_width_mm, gw.export_height_mm):
                gw.export_width_mm = width_mm
                gw.export_height_mm = height_mm
                gw.properties_changed.emit(gw.graph_id, {
                    'export_width_mm': width_mm, 'export_height_mm': height_mm,
                })

            self.accept()


class VBatchExportDialog(QDialog):
    """Dialog for exporting every currently open graph to a target folder in
    one pass. Physical size isn't offered here -- each graph exports at its
    own persisted export_width_mm/export_height_mm if set, else its own
    on-screen size, exactly as a single export of that graph would."""

    def __init__(self, graph_widgets: dict, parent=None):
        """`graph_widgets`: {graph_id: VGraph}, e.g. derived from
        {gid: (widget, dialog, subwindow)} by the caller."""
        super().__init__(parent)
        self.graph_widgets = graph_widgets

        self.setWindowTitle("Export All Graphs")
        self.setModal(True)
        self.resize(420, 260)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(QLabel(f"Export {len(self.graph_widgets)} open graph(s) to a folder:"))

        self.panel = _ExportSettingsPanel("light")
        layout.addWidget(self.panel)
        self.settings = self.panel.settings

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        self.btn_export_all = QPushButton("Choose Folder && Export All...")
        self.btn_export_all.clicked.connect(self._on_export_all_clicked)
        btn_row.addWidget(self.btn_export_all)
        layout.addLayout(btn_row)

    def _on_export_all_clicked(self):
        last_dir = self.settings.get_last_directory()
        folder = QFileDialog.getExistingDirectory(self, "Export All Graphs To", last_dir)
        if not folder:
            return

        ext = self.panel.format_ext()
        dpi = self.panel.spin_dpi.value()
        transparent = self.panel.cb_transparent.isChecked()
        theme_key = self.panel.theme_key()

        succeeded, failed = 0, []
        for graph_id, gw in self.graph_widgets.items():
            width_mm = getattr(gw, 'export_width_mm', None)
            height_mm = getattr(gw, 'export_height_mm', None)
            size_inches = (width_mm / _MM_PER_INCH, height_mm / _MM_PER_INCH) if width_mm else None

            filename = _sanitize_filename(gw_display_name(gw)) or f"graph_{graph_id}"
            filepath = str(Path(folder) / f"{filename}.{ext}")

            try:
                ok = export_with_theme_override(
                    gw, filepath, ext, dpi, transparent, theme_key, size_inches=size_inches
                )
            except Exception as e:
                ok = False
                failed.append((graph_id, str(e)))
                continue
            if ok:
                succeeded += 1
            else:
                failed.append((graph_id, "export failed"))

        self.panel.save_as_last_used()
        self.settings.set_last_directory(folder)

        if failed:
            lines = [f"Graph {gid}: {reason}" for gid, reason in failed]
            QMessageBox.warning(
                self, "Some Exports Failed",
                f"{succeeded} of {len(self.graph_widgets)} graph(s) exported.\n\n" + "\n".join(lines)
            )
        self.accept()


def gw_display_name(graph_widget) -> str:
    """Best-effort filename stem for a graph widget in a batch export."""
    style = getattr(graph_widget, 'plot_style', 'graph')
    x = getattr(graph_widget, 'x', None) or 'x'
    y = getattr(graph_widget, 'y', None) or ['y']
    return f"{graph_widget.graph_id}-{style}_{x}_vs_{y[0]}"


def _sanitize_filename(name: str) -> str:
    """Strip characters that are invalid/awkward in filenames on any OS."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
