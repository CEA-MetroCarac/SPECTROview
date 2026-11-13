"""Dialog for customizing curve labels, colors, and legend box layout in Live Maps."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from spectroview import DEFAULT_COLORS
from spectroview.view.components.v_spectra_legend_editor import SpectrumColorComboBox

LEGEND_LOCATIONS = [
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "outside right",
    "outside top",
]


class LiveMapsLegendEditorDialog(QDialog):
    """Lightweight editor for Live Maps legend box placement and per-series labels/colors."""

    def __init__(
        self,
        entries: List[dict],
        custom_labels: Optional[Dict[str, str]] = None,
        custom_colors: Optional[Dict[str, str]] = None,
        legend_loc: str = "best",
        legend_ncol: int = 1,
        legend_fontsize: int = 8,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit Live Maps Legend")
        self.resize(520, 380)

        self._entries = list(entries)
        self._custom_labels = dict(custom_labels or {})
        self._custom_colors = dict(custom_colors or {})
        self._initial_loc = legend_loc if legend_loc in LEGEND_LOCATIONS else "best"
        self._initial_ncol = max(1, min(4, int(legend_ncol)))
        self._initial_fontsize = max(6, min(16, int(legend_fontsize)))
        self._row_widgets: List[tuple[str, str, str, SpectrumColorComboBox, QLineEdit]] = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ── Legend Box Options ──
        box_group = QGroupBox("Legend box", self)
        box_form = QFormLayout(box_group)

        self.cbb_loc = QComboBox(box_group)
        self.cbb_loc.addItems(LEGEND_LOCATIONS)
        self.cbb_loc.setCurrentText(self._initial_loc)
        box_form.addRow("Location:", self.cbb_loc)

        self.spb_ncol = QSpinBox(box_group)
        self.spb_ncol.setRange(1, 4)
        self.spb_ncol.setValue(self._initial_ncol)
        box_form.addRow("Columns:", self.spb_ncol)

        self.spb_fontsize = QSpinBox(box_group)
        self.spb_fontsize.setRange(6, 16)
        self.spb_fontsize.setValue(self._initial_fontsize)
        box_form.addRow("Font size:", self.spb_fontsize)

        layout.addWidget(box_group)

        # ── Series Labels & Colors Table ──
        table_group = QGroupBox("Series labels and colors", self)
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget(len(self._entries), 3, table_group)
        self.table.setHorizontalHeaderLabels(["Series / Map", "Color", "Custom label"])
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        self._row_widgets.clear()
        for row, entry in enumerate(self._entries):
            key = str(entry.get("key", ""))
            default_label = str(entry.get("default_label", key))
            auto_color = str(entry.get("auto_color", DEFAULT_COLORS[row % len(DEFAULT_COLORS)]))

            name_item = QTableWidgetItem(default_label)
            name_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 0, name_item)

            color_combo = SpectrumColorComboBox(self.table)
            color_combo.set_palette_colors(
                DEFAULT_COLORS,
                auto_color=auto_color,
                selected_color=self._custom_colors.get(key),
            )
            self.table.setCellWidget(row, 1, color_combo)

            label_edit = QLineEdit(self.table)
            label_edit.setPlaceholderText(default_label)
            label_edit.setText(self._custom_labels.get(key, ""))
            self.table.setCellWidget(row, 2, label_edit)

            self._row_widgets.append((key, default_label, auto_color, color_combo, label_edit))

        table_layout.addWidget(self.table)
        layout.addWidget(table_group, stretch=1)

        # ── Buttons ──
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        restore_btn = buttons.button(QDialogButtonBox.RestoreDefaults)
        if restore_btn is not None:
            restore_btn.clicked.connect(self._reset_defaults)
        layout.addWidget(buttons)

    def _reset_defaults(self):
        """Reset custom labels/colors and legend box settings back to defaults."""
        self.cbb_loc.setCurrentText("best")
        self.spb_ncol.setValue(1)
        self.spb_fontsize.setValue(8)
        for _key, _default_label, auto_color, color_combo, label_edit in self._row_widgets:
            color_combo.set_palette_colors(DEFAULT_COLORS, auto_color=auto_color, selected_color=None)
            label_edit.clear()

    def result_state(self) -> dict:
        """Return updated legend settings, custom labels, and custom colors."""
        labels: Dict[str, str] = {}
        colors: Dict[str, str] = {}
        for key, _default_label, _auto_color, color_combo, label_edit in self._row_widgets:
            text = label_edit.text().strip()
            if text:
                labels[key] = text
            col = color_combo.selected_color()
            if col:
                colors[key] = col
        return {
            "custom_labels": labels,
            "custom_colors": colors,
            "legend_loc": self.cbb_loc.currentText(),
            "legend_ncol": int(self.spb_ncol.value()),
            "legend_fontsize": int(self.spb_fontsize.value()),
        }
