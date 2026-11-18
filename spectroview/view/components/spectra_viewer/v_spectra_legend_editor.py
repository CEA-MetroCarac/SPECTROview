"""Batch legend editor dialog, color selector, and legend/colorbar mixin for SpectraViewer."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
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
)

from spectroview.view.components.common.customized_widgets import CustomizedPalette
from spectroview.view.components.spectra_viewer.common import (
    LEGEND_EDITOR_MAX_ROWS,
    SPECTRA_COLOR_PALETTES,
    SPECTRA_CUSTOM_PALETTES,
    SPECTRA_DISCRETE_PALETTES,
    normalized_optional_color,
    record_token_selection,
)


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

    def set_palette_colors(self, palette_colors, auto_color, selected_color=None):
        """Populate the selector while preserving automatic/custom state."""
        selected = self._normalized_color(selected_color)
        self._auto_color = self._normalized_color(auto_color) or "#000000"

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
                self._color_icon(selected), f"{selected} (current)", selected
            )

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
            QColor(initial), self, "Choose spectrum color"
        )
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

    def __init__(
        self,
        entries,
        palette_names,
        custom_palettes,
        current_palette,
        max_legend_items,
        series_colors_provider,
        palette_colors_provider,
        total_entry_count=None,
        truncation_message="",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit spectrum legend")
        self.setModal(True)

        self._entries = [dict(entry) for entry in entries]
        self._total_entry_count = (
            len(self._entries)
            if total_entry_count is None
            else int(total_entry_count)
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
            "Maximum number of entries displayed in the plot legend"
        )
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
            self.table.setHorizontalHeaderLabels(["Spectrum", "Color", "Label"])
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
                label_edit.setToolTip("Leave empty to restore the spectrum filename")
                self.label_edits.append(label_edit)
                self.table.setCellWidget(row, 2, label_edit)
                self.table.setRowHeight(row, 28)

            entries_layout.addWidget(self.table)
        else:
            self.table = QTableWidget(0, 3)
            entries_layout.addWidget(QLabel("No spectra are currently plotted."))
        layout.addWidget(entries_group, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.palette_combo.currentTextChanged.connect(self._refresh_color_choices)
        self._refresh_color_choices(current_palette)

        visible_rows = min(max(len(self._entries), 3), 12)
        self.resize(720, 245 + visible_rows * 31)
        self.setMinimumSize(600, 360)

    def _refresh_color_choices(self, palette_name):
        spectrum_count = self._total_entry_count
        automatic_colors = self._series_colors_provider(
            palette_name, spectrum_count
        )
        palette_colors = self._palette_colors_provider(
            palette_name, spectrum_count
        )

        for row, combo in enumerate(self.color_combos):
            selected = combo.selected_color()
            if combo.count() == 0:
                selected = self._entries[row].get("stored_color")
            auto_color = (
                automatic_colors[row]
                if row < len(automatic_colors)
                else "#000000"
            )
            combo.set_palette_colors(palette_colors, auto_color, selected)

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


class SpectraLegendMixin:
    """Legend double-click editor and colorbar-legend mode helpers for VSpectraViewer."""

    def _is_legend_colorbar_active(self) -> bool:
        """Return whether the 1D/3D spectra plot should show a colorbar instead of a legend."""
        return bool(
            not getattr(self, "_heatmap_mode", False)
            and hasattr(self, "btn_legend")
            and self.btn_legend.isChecked()
            and hasattr(self, "act_legend_colorbar")
            and self.act_legend_colorbar.isChecked()
        )

    def _update_colorbar_options_visibility(self, visible: bool):
        """Show colorbar parameter and label controls only when colorbar mode is checked."""
        vis = bool(visible)
        for w in (
            getattr(self, "row_colorbar_param", None),
            getattr(self, "row_colorbar_label", None),
            getattr(self, "cbb_colorbar_param", None),
            getattr(self, "le_colorbar_label", None),
        ):
            if w is not None:
                w.setVisible(vis)
        if hasattr(self, "_options_panel") and self._options_panel is not None:
            self._options_panel.adjustSize()
        if hasattr(self, "options_menu") and self.options_menu is not None:
            self.options_menu.adjustSize()

    def _on_legend_colorbar_toggled(self, checked: bool):
        """Auto-enable legend toggle when colorbar mode is turned on, update sub-option visibility, then redraw."""
        self._update_colorbar_options_visibility(checked)
        if checked and not getattr(self, "_is_setting_options", False):
            if hasattr(self, "btn_legend") and not self.btn_legend.isChecked():
                self.btn_legend.blockSignals(True)
                self.btn_legend.setChecked(True)
                self.btn_legend.blockSignals(False)
        self._emit_view_options()

    def _resolve_colorbar_cmap_and_norm(self, count: int, param_values: np.ndarray):
        """Build colormap, normalization, and per-spectrum colors for colorbar mode."""
        palette_name = (
            self.cbb_color_palette.currentText()
            if hasattr(self, "cbb_color_palette")
            else "DEFAULT_COLORS"
        )
        if palette_name in SPECTRA_DISCRETE_PALETTES:
            disc_colors = self._get_colors_for_palette(palette_name, max(count, 1))
            cmap = mpl.colors.ListedColormap(disc_colors)
        else:
            try:
                cmap = mpl.colormaps[palette_name]
            except (KeyError, AttributeError):
                cmap = mpl.colormaps["viridis"]

        valid = (
            ~np.isnan(param_values)
            if len(param_values) > 0
            else np.array([], dtype=bool)
        )
        if count > 0 and np.any(valid):
            vmin = float(np.nanmin(param_values))
            vmax = float(np.nanmax(param_values))
        else:
            vmin, vmax = 0.0, float(max(count - 1, 1))

        if vmin == vmax:
            norm = mpl.colors.Normalize(vmin=vmin - 0.5, vmax=vmax + 0.5)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        if (
            palette_name not in SPECTRA_DISCRETE_PALETTES
            and count > 0
            and np.any(valid)
            and vmin < vmax
        ):
            fallback = self._get_colors_for_palette(palette_name, count)
            mapped_colors = [
                mpl.colors.to_hex(cmap(norm(param_values[i])))
                if i < len(param_values) and not np.isnan(param_values[i])
                else fallback[i]
                for i in range(count)
            ]
        else:
            mapped_colors = self._get_colors_for_palette(palette_name, count)

        return cmap, norm, mapped_colors

    def _on_colorbar_param_changed(self):
        """Track user-chosen colorbar parameter for the session and redraw."""
        record_token_selection(
            self.cbb_colorbar_param, self._colorbar_param_selection
        )
        self._emit_view_options()

    def _refresh_colorbar_param_choices(self):
        """Rebuild the colorbar parameter combobox items from the selected filenames."""
        if hasattr(self, "cbb_colorbar_param"):
            self._refresh_token_combobox(
                self.cbb_colorbar_param, self._colorbar_param_selection
            )

    def _get_colorbar_values(self):
        """Return ``(values, label)`` for the 1D spectrum colorbar."""
        return self._extract_series_parameter_values(
            self.cbb_colorbar_param.currentText(),
            self.le_colorbar_label.text(),
            token_pos=self.cbb_colorbar_param.currentData(),
        )

    def _make_legend_pickable(self, legend):
        """Cache the legend bounds for whole-box double-click interaction."""
        self._legend_obj = legend
        # ``renderer`` is only installed as an attribute after the first draw;
        # get_renderer() also works when data arrives before the widget has
        # been painted (common during workspace restoration and in tests).
        self._legend_bbox = legend.get_window_extent(
            self.canvas.get_renderer()
        )

        # Connect double-click handler once (replaces pick_event)
        if not hasattr(self, "_legend_dblclick_connected"):
            self.canvas.mpl_connect("button_press_event", self._on_legend_double_click)
            self._legend_dblclick_connected = True

    def _on_legend_double_click(self, event):
        """Open the batch editor when any part of the legend is double-clicked."""
        if not getattr(event, "dblclick", False):
            return

        legend_obj = getattr(self, "_legend_obj", None)
        if legend_obj is not None:
            try:
                self._legend_bbox = legend_obj.get_window_extent(
                    self.canvas.get_renderer()
                )
            except Exception:
                pass

        legend_bbox = getattr(self, "_legend_bbox", None)
        if legend_bbox is None or event.x is None or event.y is None:
            return

        if legend_bbox.contains(event.x, event.y):
            self._open_legend_editor()

    def _legend_editor_entries(self):
        """Describe editable plotted spectra, capped for responsive editing."""
        if not self._tensor_data:
            return []

        spectra = self._tensor_data.get("y")
        spectrum_count = len(spectra) if spectra is not None else 0
        editor_count = min(spectrum_count, LEGEND_EDITOR_MAX_ROWS)
        raw_labels = self._tensor_data.get("labels")
        raw_fnames = self._tensor_data.get("fnames")
        raw_colors = self._tensor_data.get("colors")
        labels = [] if raw_labels is None else list(raw_labels)
        fnames = [] if raw_fnames is None else list(raw_fnames)
        stored_colors = [] if raw_colors is None else list(raw_colors)

        entries = []
        for index in range(editor_count):
            name = (
                str(fnames[index])
                if index < len(fnames) and fnames[index]
                else f"Spectrum {index + 1}"
            )
            display_label = (
                str(labels[index])
                if index < len(labels) and labels[index]
                else name
            )
            stored_color = (
                stored_colors[index]
                if index < len(stored_colors) and stored_colors[index]
                else None
            )
            entries.append({
                "index": index,
                "name": name,
                "display_label": display_label,
                "stored_color": stored_color,
            })
        return entries

    def _create_legend_editor_dialog(self):
        spectrum_count = len(self._tensor_data.get("y", []))
        entries = self._legend_editor_entries()
        truncation_message = ""
        if len(entries) < spectrum_count:
            truncation_message = (
                f"Performance safeguard: showing the first "
                f"{len(entries)} of {spectrum_count} plotted spectra. "
                "Narrow the selection to edit spectra outside this batch."
            )
        return SpectraLegendEditorDialog(
            entries=entries,
            total_entry_count=spectrum_count,
            truncation_message=truncation_message,
            palette_names=SPECTRA_COLOR_PALETTES,
            custom_palettes=SPECTRA_CUSTOM_PALETTES,
            current_palette=self.cbb_color_palette.currentText(),
            max_legend_items=self.spin_max_legend_items.value(),
            series_colors_provider=self._get_colors_for_palette,
            palette_colors_provider=self._get_palette_color_choices,
            parent=self,
        )

    def _open_legend_editor(self):
        """Open the transactional editor and apply its values only on OK."""
        if not self._legend_editor_entries():
            return

        dialog = self._create_legend_editor_dialog()
        self._legend_editor_dialog = dialog
        if dialog.exec() == QDialog.Accepted:
            self._apply_legend_editor_values(dialog.values())

    @staticmethod
    def _normalized_optional_color(color):
        return normalized_optional_color(color)

    def _apply_legend_editor_values(self, values):
        """Commit accepted dialog values to view state and spectrum proxies."""
        if not self._tensor_data:
            return

        palette = values.get("palette", self.cbb_color_palette.currentText())
        max_items = int(
            values.get("max_legend_items", self.spin_max_legend_items.value())
        )
        options_changed = (
            palette != self.cbb_color_palette.currentText()
            or max_items != self.spin_max_legend_items.value()
        )

        self.cbb_color_palette.blockSignals(True)
        self.spin_max_legend_items.blockSignals(True)
        self.cbb_color_palette.setCurrentText(palette)
        self.spin_max_legend_items.setValue(max_items)
        self.cbb_color_palette.blockSignals(False)
        self.spin_max_legend_items.blockSignals(False)

        spectrum_count = len(self._tensor_data.get("y", []))
        raw_labels = self._tensor_data.get("labels")
        raw_colors = self._tensor_data.get("colors")
        raw_proxies = self._tensor_data.get("proxies")
        labels = [] if raw_labels is None else list(raw_labels)
        colors = [] if raw_colors is None else list(raw_colors)
        proxies = [] if raw_proxies is None else list(raw_proxies)
        labels.extend([None] * (spectrum_count - len(labels)))
        colors.extend([None] * (spectrum_count - len(colors)))

        spectra_changed = False
        for edited in values.get("spectra", []):
            index = int(edited.get("index", -1))
            if index < 0 or index >= spectrum_count:
                continue

            new_label = str(edited.get("label", "")).strip()
            if not new_label:
                new_label = (
                    self._tensor_data.get("fnames", [])[index]
                    if index < len(self._tensor_data.get("fnames", []))
                    else f"Spectrum {index + 1}"
                )
            if new_label != edited.get("original_display_label"):
                labels[index] = new_label
                if index < len(proxies):
                    proxies[index].label = new_label
                spectra_changed = True

            old_color = self._normalized_optional_color(
                edited.get("original_color")
            )
            new_color = self._normalized_optional_color(edited.get("color"))
            if new_color != old_color:
                colors[index] = new_color
                if index < len(proxies):
                    proxies[index].color = new_color
                spectra_changed = True

        self._tensor_data["labels"] = labels
        self._tensor_data["colors"] = colors

        if spectra_changed:
            self.spectrumCustomized.emit()
        if options_changed:
            self._emit_view_options()
        elif spectra_changed:
            self._plot()
