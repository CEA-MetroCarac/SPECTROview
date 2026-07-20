"""Multi-panel figure composer for the Graph Workspace: combine several
currently-open graphs into one exported figure (a grid of subplots), reusing
the export machinery already built for single/batch export.

Each panel is rendered by temporarily repointing its source VGraph onto the
composed figure's subplot Axes and calling the graph's own normal plot()
method, so a panel looks exactly like the live graph (legend, secondary
axes, annotations, inset -- everything). The source widget's real .ax/
.figure are restored and it is replotted immediately afterward, so the live
workspace is never left mutated -- the same swap-render-restore shape
export_with_theme_override() already uses for a theme override.
"""
import math
import string
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QFileDialog, QMessageBox,
)

from spectroview.viewmodel.utils import export_figure_to_file
from spectroview.view.components.v_export_dialog import (
    _ExportSettingsPanel, _MM_PER_INCH, gw_display_name,
)

_LABEL_STYLES = ["a, b, c, ...", "A, B, C, ...", "i, ii, iii, ...", "1, 2, 3, ..."]

_ROMAN_TABLE = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
]


def _to_roman(n: int) -> str:
    """Minimal roman-numeral converter -- panel counts realistically never
    exceed a few dozen, so no need for a full-range implementation."""
    result = []
    for value, symbol in _ROMAN_TABLE:
        count, n = divmod(n, value)
        result.append(symbol * count)
    return ''.join(result)


def panel_label(style: str, index: int) -> str:
    """0-indexed panel label text for `style` (one of _LABEL_STYLES).
    Falls back past single letters (index >= 26) as a1/a2/... rather than
    raising -- unlikely in practice but shouldn't crash a large grid."""
    if style == "A, B, C, ...":
        return string.ascii_uppercase[index] if index < 26 else f"A{index - 25}"
    if style == "i, ii, iii, ...":
        return _to_roman(index + 1).lower()
    if style == "1, 2, 3, ...":
        return str(index + 1)
    return string.ascii_lowercase[index] if index < 26 else f"a{index - 25}"


def suggest_grid(n: int) -> tuple:
    """Suggest a roughly-square (rows, cols) grid that fits `n` panels,
    preferring slightly wider-than-tall (cols >= rows) -- the common shape
    for a multi-panel figure."""
    if n <= 0:
        return 1, 1
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


class VMultiPanelDialog(QDialog):
    """Dialog for composing several currently-open graphs into one
    exported multi-panel figure."""

    def __init__(self, graph_widgets: dict, parent=None):
        """`graph_widgets`: {graph_id: VGraph}, e.g. derived from
        {gid: (widget, dialog, subwindow)} by the caller (same shape
        VBatchExportDialog takes)."""
        super().__init__(parent)
        self.graph_widgets = graph_widgets

        self.setWindowTitle("Compose Multi-Panel Figure")
        self.setModal(True)
        self.resize(480, 620)

        self._setup_ui()
        self._populate_graph_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Select graphs to include and order them (Up/Down):"))

        list_row = QHBoxLayout()
        self.graph_list = QListWidget()
        self.graph_list.setSelectionMode(QListWidget.SingleSelection)
        list_row.addWidget(self.graph_list, stretch=1)

        reorder_col = QVBoxLayout()
        self.btn_move_up = QPushButton("↑")
        self.btn_move_up.setFixedWidth(30)
        self.btn_move_up.setToolTip("Move selected graph earlier in the panel order")
        self.btn_move_up.clicked.connect(self._move_selected_up)
        self.btn_move_down = QPushButton("↓")
        self.btn_move_down.setFixedWidth(30)
        self.btn_move_down.setToolTip("Move selected graph later in the panel order")
        self.btn_move_down.clicked.connect(self._move_selected_down)
        reorder_col.addWidget(self.btn_move_up)
        reorder_col.addWidget(self.btn_move_down)
        reorder_col.addStretch()
        list_row.addLayout(reorder_col)
        layout.addLayout(list_row)

        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid:"))
        grid_row.addWidget(QLabel("Rows"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 12)
        grid_row.addWidget(self.spin_rows)
        grid_row.addWidget(QLabel("Cols"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 12)
        grid_row.addWidget(self.spin_cols)
        self.btn_auto_grid = QPushButton("Auto")
        self.btn_auto_grid.setToolTip("Suggest a roughly-square grid for the checked graphs")
        self.btn_auto_grid.clicked.connect(self._auto_grid)
        grid_row.addWidget(self.btn_auto_grid)
        grid_row.addStretch()
        layout.addLayout(grid_row)

        self.cb_shared_labels = QCheckBox("Shared axis labels (hide tick/axis labels on interior panels)")
        self.cb_shared_labels.setChecked(True)
        layout.addWidget(self.cb_shared_labels)

        label_row = QHBoxLayout()
        self.cb_panel_labels = QCheckBox("Panel labels:")
        self.cb_panel_labels.setChecked(True)
        label_row.addWidget(self.cb_panel_labels)
        self.combo_label_style = QComboBox()
        self.combo_label_style.addItems(_LABEL_STYLES)
        label_row.addWidget(self.combo_label_style)
        label_row.addStretch()
        layout.addLayout(label_row)

        self.panel = _ExportSettingsPanel("light")
        layout.addWidget(self.panel)
        self.settings = self.panel.settings

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Combined figure size (mm):"))
        size_row.addWidget(QLabel("Width"))
        self.spin_fig_width = QDoubleSpinBox()
        self.spin_fig_width.setRange(10.0, 2000.0)
        self.spin_fig_width.setValue(180.0)
        size_row.addWidget(self.spin_fig_width)
        size_row.addWidget(QLabel("Height"))
        self.spin_fig_height = QDoubleSpinBox()
        self.spin_fig_height.setRange(10.0, 2000.0)
        self.spin_fig_height.setValue(150.0)
        size_row.addWidget(self.spin_fig_height)
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

    def _populate_graph_list(self):
        self.graph_list.clear()
        for graph_id, gw in self.graph_widgets.items():
            item = QListWidgetItem(f"{graph_id} - {gw_display_name(gw)}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, graph_id)
            self.graph_list.addItem(item)
        self._auto_grid()

    def _move_selected_up(self):
        row = self.graph_list.currentRow()
        if row > 0:
            item = self.graph_list.takeItem(row)
            self.graph_list.insertItem(row - 1, item)
            self.graph_list.setCurrentRow(row - 1)

    def _move_selected_down(self):
        row = self.graph_list.currentRow()
        if 0 <= row < self.graph_list.count() - 1:
            item = self.graph_list.takeItem(row)
            self.graph_list.insertItem(row + 1, item)
            self.graph_list.setCurrentRow(row + 1)

    def _checked_graph_ids_in_order(self) -> list:
        ids = []
        for i in range(self.graph_list.count()):
            item = self.graph_list.item(i)
            if item.checkState() == Qt.Checked:
                ids.append(item.data(Qt.UserRole))
        return ids

    def _auto_grid(self):
        n = len(self._checked_graph_ids_in_order()) or len(self.graph_widgets)
        rows, cols = suggest_grid(n)
        self.spin_rows.setValue(rows)
        self.spin_cols.setValue(cols)

    # ------------------------------------------------------------------ #
    #  Composition
    # ------------------------------------------------------------------ #

    def _render_graph_onto(self, gw, target_ax):
        """Temporarily repoint `gw` at `target_ax` (and its parent figure)
        and render it, then restore + replot gw against its real figure so
        the live workspace is never left mutated.

        A graph with an active broken axis is composed as a single
        simplified (unbroken) panel via _render_series_on(): a broken axis
        needs two Axes but a grid cell is only one, and reusing the live
        graph's own two-panel layout would corrupt the on-screen graph.
        """
        original_ax = gw.ax
        original_figure = gw.figure
        has_break = bool(gw.axis_breaks.get('x') or gw.axis_breaks.get('y'))

        gw.ax = target_ax
        gw.figure = target_ax.figure
        try:
            if gw.df is None:
                return
            if has_break:
                gw._render_series_on(target_ax)
            else:
                gw.plot(gw.df)
        finally:
            gw.ax = original_ax
            gw.figure = original_figure
            if gw.df is not None:
                gw.plot(gw.df)  # repair: rebuild ax2/ax3/inset/etc against the real figure

    def _compose_figure(self) -> Figure:
        """Build one throwaway Figure with the checked graphs arranged in
        the requested grid. Raises ValueError if nothing is checked."""
        graph_ids = self._checked_graph_ids_in_order()
        if not graph_ids:
            raise ValueError("No graphs selected.")

        rows, cols = self.spin_rows.value(), self.spin_cols.value()
        width_mm, height_mm = self.spin_fig_width.value(), self.spin_fig_height.value()
        size_inches = (width_mm / _MM_PER_INCH, height_mm / _MM_PER_INCH)

        # "constrained" layout auto-adjusts panel spacing to avoid
        # title/label collisions -- VGraph itself uses "compressed",
        # constrained's colorbar-aware sibling.
        composed_fig = Figure(figsize=size_inches, dpi=self.panel.spin_dpi.value(), layout="constrained")
        gs = composed_fig.add_gridspec(rows, cols)

        shared_labels = self.cb_shared_labels.isChecked()
        show_panel_labels = self.cb_panel_labels.isChecked()
        label_style = self.combo_label_style.currentText()
        n = min(len(graph_ids), rows * cols)

        for idx in range(n):
            graph_id = graph_ids[idx]
            r, c = divmod(idx, cols)
            target_ax = composed_fig.add_subplot(gs[r, c])

            self._render_graph_onto(self.graph_widgets[graph_id], target_ax)

            if shared_labels:
                is_last_in_column = (idx + cols >= n)  # nothing below it in this column
                is_left_col = (c == 0)
                if not is_last_in_column:
                    target_ax.set_xlabel('')
                    target_ax.tick_params(axis='x', labelbottom=False)
                if not is_left_col:
                    target_ax.set_ylabel('')
                    target_ax.tick_params(axis='y', labelleft=False)

            if show_panel_labels:
                target_ax.text(
                    -0.12, 1.05, panel_label(label_style, idx),
                    transform=target_ax.transAxes, fontsize=13, fontweight='bold',
                    va='bottom', ha='left',
                )

        return composed_fig

    def _on_export_clicked(self):
        try:
            composed_fig = self._compose_figure()
        except ValueError as e:
            QMessageBox.warning(self, "Nothing to Export", str(e))
            return

        ext, file_filter = self.panel.format_ext(), self.panel.file_filter()
        last_dir = self.settings.get_last_directory()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Composed Figure", str(Path(last_dir) / f"multipanel.{ext}"), file_filter
        )
        if not filepath:
            return
        if not filepath.lower().endswith(f".{ext}"):
            filepath += f".{ext}"

        throwaway_canvas = FigureCanvas(composed_fig)
        success = export_figure_to_file(
            throwaway_canvas, filepath, ext,
            dpi=self.panel.spin_dpi.value(),
            transparent=self.panel.cb_transparent.isChecked(),
        )

        if success:
            self.panel.save_as_last_used()
            self.settings.set_last_directory(str(Path(filepath).parent))
            self.accept()
