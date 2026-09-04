"""Batch editor for labels and colors in the SpectraViewer legend."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QComboBox,
)

from spectroview.view.components.customized_widgets import CustomizedPalette


class SpectrumColorComboBox(QComboBox):
    """Color selector with an automatic palette color and custom-color entry."""

    _MORE_COLORS = "__more_colors__"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(QSize(42, 14))
        self.setMinimumWidth(150)
        self._auto_color = "#000000"
        self._last_selected_color = None
        self.activated.connect(self._on_activated)

    @staticmethod
    def _normalized_color(color):
        qcolor = QColor(color or "")
        return qcolor.name() if qcolor.isValid() else None

    @staticmethod
    def _color_icon(color):
        pixmap = QPixmap(42, 14)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)

    def set_palette_colors(self, palette_colors, auto_color,
                           selected_color=None):
        """Populate the selector while preserving automatic/custom state."""
        selected = self._normalized_color(selected_color)
        self._auto_color = (
            self._normalized_color(auto_color) or "#000000")

        available = []
        for color in palette_colors:
            normalized = self._normalized_color(color)
            if normalized and normalized not in available:
                available.append(normalized)

        self.blockSignals(True)
        self.clear()
        self.addItem(
            self._color_icon(self._auto_color),
            "Automatic",
            None,
        )
        self.setItemData(
            0,
            f"Assigned from the selected palette ({self._auto_color})",
            Qt.ToolTipRole,
        )

        if selected and selected not in available:
            self.addItem(
                self._color_icon(selected), f"{selected} (current)", selected)

        for color in available:
            self.addItem(self._color_icon(color), color, color)

        self.insertSeparator(self.count())
        self.addItem("More colors…", self._MORE_COLORS)

        self._select_color(selected)
        self._last_selected_color = selected
        self.blockSignals(False)

    def _select_color(self, color):
        normalized = self._normalized_color(color)
        target = 0
        if normalized:
            for index in range(self.count()):
                if self.itemData(index) == normalized:
                    target = index
                    break
        self.setCurrentIndex(target)

    def selected_color(self):
        """Return an explicit color or ``None`` for palette-driven color."""
        data = self.currentData()
        if data == self._MORE_COLORS:
            return self._last_selected_color
        return self._normalized_color(data)

    def _on_activated(self, index):
        data = self.itemData(index)
        if data != self._MORE_COLORS:
            self._last_selected_color = self._normalized_color(data)
            return

        initial = self._last_selected_color or self._auto_color
        color = QColorDialog.getColor(
            QColor(initial), self, "Choose spectrum color")
        if color.isValid():
            selected = color.name()
            self.insertItem(
                max(1, self.count() - 2),
                self._color_icon(selected),
                selected,
                selected,
            )
            self._select_color(selected)
            self._last_selected_color = selected
        else:
            self._select_color(self._last_selected_color)


class SpectraLegendEditorDialog(QDialog):
    """Edit all plotted spectrum legend entries in one transaction."""

    def __init__(self, entries, palette_names, custom_palettes,
                 current_palette, max_legend_items,
                 series_colors_provider, palette_colors_provider,
                 total_entry_count=None, truncation_message="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit spectrum legend")
        self.setModal(True)

        self._entries = [dict(entry) for entry in entries]
        self._total_entry_count = (
            len(self._entries)
            if total_entry_count is None else int(total_entry_count)
        )
        self._series_colors_provider = series_colors_provider
        self._palette_colors_provider = palette_colors_provider
        self.color_combos = []
        self.label_edits = []

        layout = QVBoxLayout(self)

        settings_group = QGroupBox("Legend settings")
        settings_layout = QFormLayout(settings_group)
        self.palette_combo = CustomizedPalette(
            palette_list=palette_names,
            custom_palettes=custom_palettes,
        )
        self.palette_combo.setCurrentText(current_palette)
        settings_layout.addRow("Color palette:", self.palette_combo)

        self.max_items_spin = QSpinBox()
        self.max_items_spin.setRange(1, 1000)
        self.max_items_spin.setValue(max_legend_items)
        self.max_items_spin.setToolTip(
            "Maximum number of entries displayed in the plot legend")
        settings_layout.addRow("Max legend items:", self.max_items_spin)
        layout.addWidget(settings_group)

        if self._total_entry_count > len(self._entries):
            entries_title = (
                f"Plotted spectra (showing {len(self._entries)} of "
                f"{self._total_entry_count})"
            )
        else:
            entries_title = f"Plotted spectra ({len(self._entries)})"
        entries_group = QGroupBox(entries_title)
        entries_layout = QVBoxLayout(entries_group)
        if truncation_message:
            truncation_label = QLabel(truncation_message)
            truncation_label.setWordWrap(True)
            truncation_label.setObjectName("legendEditorLimitNotice")
            entries_layout.addWidget(truncation_label)
        if self._entries:
            self.table = QTableWidget(len(self._entries), 3)
            self.table.setHorizontalHeaderLabels(
                ["Spectrum", "Color", "Label"])
            self.table.verticalHeader().setVisible(False)
            self.table.setAlternatingRowColors(True)
            self.table.setSelectionMode(QAbstractItemView.NoSelection)
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.Stretch)

            for row, entry in enumerate(self._entries):
                name_item = QTableWidgetItem(entry["name"])
                name_item.setToolTip(entry["name"])
                self.table.setItem(row, 0, name_item)

                color_combo = SpectrumColorComboBox()
                self.color_combos.append(color_combo)
                self.table.setCellWidget(row, 1, color_combo)

                label_edit = QLineEdit(entry["display_label"])
                label_edit.setClearButtonEnabled(True)
                label_edit.setToolTip(
                    "Leave empty to restore the spectrum filename")
                self.label_edits.append(label_edit)
                self.table.setCellWidget(row, 2, label_edit)
                self.table.setRowHeight(row, 28)

            entries_layout.addWidget(self.table)
        else:
            self.table = QTableWidget(0, 3)
            entries_layout.addWidget(QLabel("No spectra are currently plotted."))
        layout.addWidget(entries_group, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.palette_combo.currentTextChanged.connect(
            self._refresh_color_choices)
        self._refresh_color_choices(current_palette)

        visible_rows = min(max(len(self._entries), 3), 12)
        self.resize(720, 245 + visible_rows * 31)
        self.setMinimumSize(600, 360)

    def _refresh_color_choices(self, palette_name):
        spectrum_count = self._total_entry_count
        automatic_colors = self._series_colors_provider(
            palette_name, spectrum_count)
        palette_colors = self._palette_colors_provider(
            palette_name, spectrum_count)

        for row, combo in enumerate(self.color_combos):
            selected = combo.selected_color()
            if combo.count() == 0:
                selected = self._entries[row].get("stored_color")
            auto_color = (
                automatic_colors[row]
                if row < len(automatic_colors) else "#000000"
            )
            combo.set_palette_colors(
                palette_colors, auto_color, selected)

    def values(self):
        """Return the edited values without changing the underlying spectra."""
        spectra = []
        for row, entry in enumerate(self._entries):
            label = self.label_edits[row].text().strip() or entry["name"]
            spectra.append({
                "index": entry["index"],
                "label": label,
                "color": self.color_combos[row].selected_color(),
                "original_display_label": entry["display_label"],
                "original_color": entry.get("stored_color"),
            })
        return {
            "palette": self.palette_combo.currentText(),
            "max_legend_items": self.max_items_spin.value(),
            "spectra": spectra,
        }
