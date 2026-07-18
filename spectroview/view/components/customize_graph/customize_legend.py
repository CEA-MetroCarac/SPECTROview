"""Legend / Color tab of the Customize Graph dialog: per-series label,
marker, color, and style editing, plus scatter marker size/edge-color
settings and error-bar options.

Split out of customize_graph_dialog.py; no behavior changes.
"""
import copy

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QColorDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
)

from spectroview import DEFAULT_COLORS, MARKERS
from spectroview.view.components.customize_graph.customize_annotation_dialogs import ColorDelegate
from spectroview.view.components.customize_graph.spin_widgets import (
    PlaceholderDoubleSpinBox, PlaceholderSpinBox,
)

# Plot styles that draw per-series markers whose size/edge-color make sense
# to override (mirrors the existing scatter_group visibility condition).
_MARKER_STYLES = ('scatter', 'trendline', 'point')
# Plot styles with a configurable error-bar TYPE (bar's on/off gate is the
# pre-existing show_bar_plot_error_bar checkbox in the More Options tab;
# this tab only ever picks *which statistic* once bars are shown).
_ERROR_BAR_STYLES = ('point', 'line', 'bar')

_ERROR_BAR_LABELS_POINT_LINE = [
    ("None", "none"), ("Std. Dev.", "sd"), ("Std. Error", "sem"), ("95% CI", "ci95"),
]
_ERROR_BAR_LABELS_BAR = [
    ("Std. Dev.", "sd"), ("Std. Error", "sem"), ("95% CI", "ci95"),
]


class CustomizeLegend(QWidget):
    """Widget Tab for customizing legend properties (labels, markers, colors,
    per-series style overrides) and error-bar options."""

    # Sentinel for "no override" on the optional per-series numeric spinboxes
    # -- mirrors the pattern used for axis limits in customize_axis.py.
    _UNSET = -999999.0

    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.original_legend_properties = None

        self._setup_ui()

        # Load initial properties
        self.load_legend_properties()

    def switch_graph(self, graph_widget):
        """Switch to a different graph widget and reload legend properties."""
        self.graph_widget = graph_widget
        self.original_legend_properties = None
        self.load_legend_properties()

    def _make_optional_spinbox(self, is_int=False, minimum=None, maximum=None,
                                decimals=2, step=None, start_value=0.0):
        """Create an optional per-series override spinbox: shows a grayed
        "default" placeholder when unset, and the first arrow-click from
        that state jumps to `start_value` instead of stepping from the
        sentinel itself."""
        cls = PlaceholderSpinBox if is_int else PlaceholderDoubleSpinBox
        spin = cls(self._UNSET, start_value, "default")
        lo = self._UNSET if minimum is None else min(minimum, self._UNSET)
        hi = 999999 if maximum is None else maximum
        spin.setRange(lo, hi)
        if not is_int:
            spin.setDecimals(decimals)
        if step is not None:
            spin.setSingleStep(step)
        return spin

    def _setup_ui(self):
        """Setup the UI components for the legend customization widget."""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(8)

        # Container for legend widgets (labels, markers, colors, styles)
        self.legend_container = QGroupBox("Legends box:")
        box_layout = QVBoxLayout(self.legend_container)
        box_layout.setContentsMargins(4, 4, 4, 4)
        box_layout.setSpacing(8)

        self.cb_legend_outside = QCheckBox("Put legend box outside")
        self.cb_legend_outside.stateChanged.connect(self._on_legend_outside_changed)
        box_layout.addWidget(self.cb_legend_outside)

        self.legend_layout = QHBoxLayout()
        box_layout.addLayout(self.legend_layout)

        self.main_layout.addWidget(self.legend_container)

        # ───── Legend box style (ncol/frame/title/fontsize/alpha/location) ─────
        self.legend_style_group = QGroupBox("Legend style:")
        style_layout = QVBoxLayout(self.legend_style_group)
        style_layout.setContentsMargins(4, 4, 4, 4)
        style_layout.setSpacing(8)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Columns:"))
        self.spin_legend_ncol = QSpinBox()
        self.spin_legend_ncol.setRange(1, 10)
        row1.addWidget(self.spin_legend_ncol)

        row1.addSpacing(10)
        self.cb_legend_frame = QCheckBox("Frame")
        row1.addWidget(self.cb_legend_frame)

        row1.addSpacing(10)
        row1.addWidget(QLabel("Title:"))
        self.edit_legend_title = QLineEdit()
        self.edit_legend_title.setPlaceholderText("Optional legend title")
        row1.addWidget(self.edit_legend_title)
        style_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Font size:"))
        self.spin_legend_fontsize = QSpinBox()
        self.spin_legend_fontsize.setRange(4, 72)
        self.spin_legend_fontsize.setSingleStep(1)
        self.spin_legend_fontsize.setValue(10)  # matches mplstyle's legend.fontsize
        row2.addWidget(self.spin_legend_fontsize)

        row2.addSpacing(10)
        row2.addWidget(QLabel("Alpha:"))
        self.spin_legend_alpha = QDoubleSpinBox()
        self.spin_legend_alpha.setRange(0.0, 1.0)
        self.spin_legend_alpha.setDecimals(2)
        self.spin_legend_alpha.setSingleStep(0.1)
        row2.addWidget(self.spin_legend_alpha)

        row2.addSpacing(10)
        row2.addWidget(QLabel("Position:"))
        self.combo_legend_loc = QComboBox()
        self.combo_legend_loc.addItems([
            "best", "upper right", "upper left", "lower left", "lower right",
            "center left", "center right", "lower center", "upper center", "center",
        ])
        row2.addWidget(self.combo_legend_loc)
        row2.addStretch()
        style_layout.addLayout(row2)

        self.main_layout.addWidget(self.legend_style_group)

        # ───── Scatter/Marker-specific settings ─────
        self.scatter_group = QGroupBox("Scatter / Marker settings:")
        scatter_layout = QHBoxLayout(self.scatter_group)
        scatter_layout.setContentsMargins(4, 4, 4, 4)
        scatter_layout.setSpacing(8)

        # Marker size
        scatter_layout.addWidget(QLabel("Marker size:"))
        self.spin_scatter_size = QSpinBox()
        self.spin_scatter_size.setRange(5, 500)
        self.spin_scatter_size.setSingleStep(10)
        self.spin_scatter_size.setValue(70)
        scatter_layout.addWidget(self.spin_scatter_size)

        scatter_layout.addSpacing(10)

        # Edge color
        scatter_layout.addWidget(QLabel("Edge color:"))
        self.btn_scatter_edgecolor = QPushButton()
        self.btn_scatter_edgecolor.setFixedWidth(80)
        self._set_color_button(self.btn_scatter_edgecolor, 'black')
        self.btn_scatter_edgecolor.clicked.connect(self._pick_scatter_edgecolor)
        scatter_layout.addWidget(self.btn_scatter_edgecolor)

        scatter_layout.addStretch()
        self.main_layout.addWidget(self.scatter_group)

        # Only show scatter group when plot is scatter, trendline, or point style
        self.scatter_group.setVisible(
            self.graph_widget.plot_style in _MARKER_STYLES
        )

        # ───── Error-bar settings ─────
        self.error_bar_group = QGroupBox("Error bars:")
        error_bar_layout = QHBoxLayout(self.error_bar_group)
        error_bar_layout.setContentsMargins(4, 4, 4, 4)
        error_bar_layout.setSpacing(8)

        error_bar_layout.addWidget(QLabel("Type:"))
        self.cbb_error_bar_type = QComboBox()
        error_bar_layout.addWidget(self.cbb_error_bar_type)

        error_bar_layout.addSpacing(10)
        error_bar_layout.addWidget(QLabel("Cap size:"))
        self.spin_error_bar_capsize = QDoubleSpinBox()
        self.spin_error_bar_capsize.setRange(0.0, 20.0)
        self.spin_error_bar_capsize.setDecimals(1)
        self.spin_error_bar_capsize.setSingleStep(0.5)
        self.spin_error_bar_capsize.setValue(3.0)
        error_bar_layout.addWidget(self.spin_error_bar_capsize)

        error_bar_layout.addStretch()
        self.main_layout.addWidget(self.error_bar_group)
        self.error_bar_group.setVisible(
            self.graph_widget.plot_style in _ERROR_BAR_STYLES
        )

        self.main_layout.addStretch()

    def load_legend_properties(self):
        """Load current legend properties from graph and populate the GUI."""
        legend_properties = self.graph_widget.get_legend_properties()
        self.original_legend_properties = copy.deepcopy(legend_properties)
        self.original_legend_outside = getattr(self.graph_widget, 'legend_outside', False)
        self.original_legend_bbox = getattr(self.graph_widget, 'legend_bbox', None)

        self.cb_legend_outside.blockSignals(True)
        self.cb_legend_outside.setChecked(self.original_legend_outside)
        self.cb_legend_outside.blockSignals(False)

        # Load legend box style
        self.spin_legend_ncol.setValue(getattr(self.graph_widget, 'legend_ncol', 1))
        self.cb_legend_frame.setChecked(getattr(self.graph_widget, 'legend_frame', True))
        self.edit_legend_title.setText(getattr(self.graph_widget, 'legend_title', None) or "")
        self.spin_legend_fontsize.setValue(getattr(self.graph_widget, 'legend_fontsize', None) or 10)
        self.spin_legend_alpha.setValue(getattr(self.graph_widget, 'legend_alpha', 0.7))
        self.combo_legend_loc.setCurrentText(getattr(self.graph_widget, 'legend_loc', 'best'))
        self.combo_legend_loc.setEnabled(not self.original_legend_outside)

        # Clear existing widgets from legend layout
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.layout():
                # Recursively clear sub-layouts
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
            if item.widget():
                item.widget().deleteLater()

        if not legend_properties:
            no_legend_label = QLabel("No legend available for this plot.")
            self.legend_layout.addWidget(no_legend_label)
        else:
            # Build legend customization widgets
            self._build_legend_widgets(legend_properties)

        # Load scatter settings
        self.spin_scatter_size.setValue(
            getattr(self.graph_widget, 'scatter_size', 70)
        )
        edge_c = getattr(self.graph_widget, 'scatter_edgecolor', 'black')
        if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
            edge_c = 'black'
        self._set_color_button(self.btn_scatter_edgecolor, edge_c)
        # Show/hide scatter group based on plot style
        self.scatter_group.setVisible(
            self.graph_widget.plot_style in _MARKER_STYLES
        )

        # Load error-bar settings
        self._load_error_bar_settings()

    def _load_error_bar_settings(self):
        """Populate the error-bar group, adapting the type dropdown's
        options to the current plot style (point/line show a "None" option
        since their error bar used to be unconditional; bar doesn't, since
        its on/off gate is a separate checkbox in the More Options tab)."""
        style = self.graph_widget.plot_style
        self.error_bar_group.setVisible(style in _ERROR_BAR_STYLES)
        if style not in _ERROR_BAR_STYLES:
            return

        is_bar = (style == 'bar')
        labels = _ERROR_BAR_LABELS_BAR if is_bar else _ERROR_BAR_LABELS_POINT_LINE
        current_value = (
            getattr(self.graph_widget, 'bar_error_bar_type', 'sd') if is_bar
            else getattr(self.graph_widget, 'error_bar_type', 'ci95')
        )

        self.cbb_error_bar_type.blockSignals(True)
        self.cbb_error_bar_type.clear()
        for text, value in labels:
            self.cbb_error_bar_type.addItem(text, value)
        idx = self.cbb_error_bar_type.findData(current_value)
        self.cbb_error_bar_type.setCurrentIndex(idx if idx >= 0 else 0)
        self.cbb_error_bar_type.blockSignals(False)

        self.spin_error_bar_capsize.setValue(
            getattr(self.graph_widget, 'error_bar_capsize', 3.0)
        )

    def _build_legend_widgets(self, legend_properties):
        """Build the legend customization widgets (label, marker, color,
        and optional per-series style overrides)."""
        label_layout = QVBoxLayout()
        marker_layout = QVBoxLayout()
        color_layout = QVBoxLayout()
        linewidth_layout = QVBoxLayout()
        alpha_layout = QVBoxLayout()
        zorder_layout = QVBoxLayout()
        show_marker_col = self.graph_widget.plot_style in _MARKER_STYLES
        marker_size_layout = QVBoxLayout() if show_marker_col else None
        edge_color_layout = QVBoxLayout() if show_marker_col else None

        # Headers
        for header in ['Label', 'Marker', 'Color', 'Line width', 'Alpha', 'Z-order']:
            lbl = QLabel(header)
            lbl.setAlignment(Qt.AlignCenter)
            if header == 'Label':
                label_layout.addWidget(lbl)
            elif header == 'Marker':
                if self.graph_widget.plot_style == 'point':
                    marker_layout.addWidget(lbl)
            elif header == 'Color':
                color_layout.addWidget(lbl)
            elif header == 'Line width':
                linewidth_layout.addWidget(lbl)
            elif header == 'Alpha':
                alpha_layout.addWidget(lbl)
            elif header == 'Z-order':
                zorder_layout.addWidget(lbl)
        if show_marker_col:
            for lbl_text, layout in (('Marker size', marker_size_layout), ('Edge color', edge_color_layout)):
                lbl = QLabel(lbl_text)
                lbl.setAlignment(Qt.AlignCenter)
                layout.addWidget(lbl)

        for idx, prop in enumerate(legend_properties):
            # Label
            label = QLineEdit(prop['label'])
            label.setFixedWidth(140)
            label.textChanged.connect(
                lambda text, i=idx: self._update_legend_property(i, 'label', text)
            )
            label_layout.addWidget(label)

            # Marker (only for point plots)
            if self.graph_widget.plot_style == 'point':
                marker = QComboBox()
                marker.addItems(MARKERS)
                marker.setCurrentText(prop['marker'])
                marker.currentTextChanged.connect(
                    lambda text, i=idx: self._update_legend_property(i, 'marker', text)
                )
                marker_layout.addWidget(marker)

            # Color combobox with colored items
            color = QComboBox()
            delegate = ColorDelegate(color)
            color.setItemDelegate(delegate)

            unique_colors = list(dict.fromkeys(DEFAULT_COLORS))[:12]
            for color_code in unique_colors:
                color.addItem(color_code)
                item = color.model().item(color.count() - 1)
                item.setBackground(QColor(color_code))

            color.setCurrentText(prop['color'])
            color.currentIndexChanged.connect(
                lambda *_a, cb=color: self._update_combobox_color(cb)
            )
            color.currentTextChanged.connect(
                lambda text, i=idx: self._update_legend_property(i, 'color', text)
            )
            color_layout.addWidget(color)
            self._update_combobox_color(color)

            # Optional per-series style overrides -- blank (unset) by
            # default, since old saved graphs have none of these keys and
            # must keep falling back to the graph-wide default.
            lw_spin = self._make_optional_spinbox(
                minimum=0.0, maximum=20.0, decimals=2, step=0.5, start_value=1.5
            )
            lw_spin.setValue(prop.get('linewidth', self._UNSET))
            lw_spin.valueChanged.connect(
                lambda v, i=idx: self._update_legend_property_numeric(i, 'linewidth', v)
            )
            linewidth_layout.addWidget(lw_spin)

            alpha_spin = self._make_optional_spinbox(
                minimum=0.0, maximum=1.0, decimals=2, step=0.1, start_value=1.0
            )
            alpha_spin.setValue(prop.get('alpha', self._UNSET))
            alpha_spin.valueChanged.connect(
                lambda v, i=idx: self._update_legend_property_numeric(i, 'alpha', v)
            )
            alpha_layout.addWidget(alpha_spin)

            zorder_spin = self._make_optional_spinbox(
                minimum=-100.0, maximum=100.0, decimals=1, step=1.0, start_value=2.0
            )
            zorder_spin.setValue(prop.get('zorder', self._UNSET))
            zorder_spin.valueChanged.connect(
                lambda v, i=idx: self._update_legend_property_numeric(i, 'zorder', v)
            )
            zorder_layout.addWidget(zorder_spin)

            if show_marker_col:
                default_marker_size = getattr(self.graph_widget, 'scatter_size', 70)
                ms_spin = self._make_optional_spinbox(
                    is_int=True, minimum=1, maximum=500, start_value=default_marker_size
                )
                ms_spin.setValue(int(prop.get('marker_size', self._UNSET)))
                ms_spin.valueChanged.connect(
                    lambda v, i=idx: self._update_legend_property_numeric(i, 'marker_size', v)
                )
                marker_size_layout.addWidget(ms_spin)

                edge_btn = QPushButton()
                edge_btn.setFixedWidth(70)
                edge_val = prop.get('edge_color')
                self._set_color_button(edge_btn, edge_val or 'black')
                if not edge_val:
                    edge_btn.setText("(default)")
                edge_btn.clicked.connect(
                    lambda *_a, i=idx, btn=edge_btn: self._pick_series_edge_color(i, btn)
                )
                edge_color_layout.addWidget(edge_btn)

        # Stretch at bottom
        label_layout.addStretch()
        if self.graph_widget.plot_style == 'point':
            marker_layout.addStretch()
        color_layout.addStretch()
        linewidth_layout.addStretch()
        alpha_layout.addStretch()
        zorder_layout.addStretch()
        if show_marker_col:
            marker_size_layout.addStretch()
            edge_color_layout.addStretch()

        self.legend_layout.addLayout(label_layout)
        self.legend_layout.addLayout(marker_layout)
        self.legend_layout.addLayout(color_layout)
        self.legend_layout.addLayout(linewidth_layout)
        self.legend_layout.addLayout(alpha_layout)
        self.legend_layout.addLayout(zorder_layout)
        if show_marker_col:
            self.legend_layout.addLayout(marker_size_layout)
            self.legend_layout.addLayout(edge_color_layout)

    def _update_legend_property(self, idx, property_type, text):
        """Update a legend property and refresh the plot live."""
        self.graph_widget.legend_properties[idx][property_type] = text
        self.graph_widget._set_legend()
        self.graph_widget.canvas.draw_idle()

    def _update_legend_property_numeric(self, idx, property_type, value):
        """Store or clear an optional per-series numeric override. The
        sentinel marks "no override" -- store as an absent key so renderers
        fall back to the graph's global default, not as a literal sentinel
        value. Takes effect on the next full replot (Apply), matching the
        existing per-series marker/color controls' behavior."""
        if value == self._UNSET:
            self.graph_widget.legend_properties[idx].pop(property_type, None)
        else:
            self.graph_widget.legend_properties[idx][property_type] = value

    def _pick_series_edge_color(self, idx, button):
        """Open color picker for a per-series marker edge-color override."""
        current = QColor(button.text()) if button.text() != "(default)" else QColor('black')
        color = QColorDialog.getColor(current, self, "Select Marker Edge Color")
        if color.isValid():
            self.graph_widget.legend_properties[idx]['edge_color'] = color.name()
            self._set_color_button(button, color.name())

    def _on_legend_outside_changed(self, state):
        """Update whether the legend is placed outside the plot."""
        self.graph_widget.legend_outside = self.cb_legend_outside.isChecked()
        self.graph_widget.legend_bbox = None  # Reset drag position
        # legend_loc only governs the inside-legend position
        self.combo_legend_loc.setEnabled(not self.graph_widget.legend_outside)
        self.graph_widget._set_legend()
        if self.graph_widget.df is not None:
            # Need a full replot to adjust layout for the outside legend
            self.graph_widget.plot(self.graph_widget.df)
        else:
            self.graph_widget.canvas.draw_idle()

    def _update_combobox_color(self, combobox):
        """Update combobox background color to match selected color.

        Uses inline setStyleSheet() instead of QPalette because the global QSS
        stylesheet always overrides palette-based colours.
        """
        selected_color = combobox.currentText()
        qc = QColor(selected_color)
        if not qc.isValid():
            return
        # Choose contrasting text: white on dark colours, black on light ones
        text_color = "white" if qc.lightnessF() < 0.6 else "black"
        combobox.setStyleSheet(
            f"QComboBox {{ background: {qc.name()}; color: {text_color}; }}"
        )

    def _set_color_button(self, button, color_name):
        """Set a QPushButton background and text to a given color."""
        qc = QColor(color_name)
        button.setStyleSheet(f"background-color: {qc.name()};")
        button.setText(qc.name())

    def _pick_scatter_edgecolor(self):
        """Open color picker for scatter marker edge color."""
        current = QColor(self.btn_scatter_edgecolor.text())
        color = QColorDialog.getColor(current, self, "Select Marker Edge Color")
        if color.isValid():
            self._set_color_button(self.btn_scatter_edgecolor, color.name())

    def apply_changes(self):
        """Apply legend changes by doing a full replot."""
        # Error-bar settings
        style = self.graph_widget.plot_style
        if style in _ERROR_BAR_STYLES:
            error_value = self.cbb_error_bar_type.currentData()
            if style == 'bar':
                self.graph_widget.bar_error_bar_type = error_value
            else:
                self.graph_widget.error_bar_type = error_value
            self.graph_widget.error_bar_capsize = self.spin_error_bar_capsize.value()

        # Legend box style
        self.graph_widget.legend_ncol = self.spin_legend_ncol.value()
        self.graph_widget.legend_frame = self.cb_legend_frame.isChecked()
        self.graph_widget.legend_title = self.edit_legend_title.text().strip() or None
        self.graph_widget.legend_fontsize = self.spin_legend_fontsize.value()
        self.graph_widget.legend_alpha = self.spin_legend_alpha.value()
        self.graph_widget.legend_loc = self.combo_legend_loc.currentText()

        # Full replot to ensure all changes are committed
        if self.graph_widget.df is not None:
            self.graph_widget.plot(self.graph_widget.df)
        else:
            self.graph_widget.canvas.draw_idle()

        # Update the backup so Cancel won't revert applied changes
        self.original_legend_properties = copy.deepcopy(self.graph_widget.get_legend_properties())
        self.original_legend_outside = getattr(self.graph_widget, 'legend_outside', False)
        self.original_legend_bbox = getattr(self.graph_widget, 'legend_bbox', None)

        # Apply scatter-specific properties
        if self.graph_widget.plot_style in ['scatter', 'trendline', 'point']:
            self.graph_widget.scatter_size = self.spin_scatter_size.value()
            edge_c = self.btn_scatter_edgecolor.text()
            if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
                edge_c = 'black'
            self.graph_widget.scatter_edgecolor = edge_c
            # Replot to reflect changes
            if self.graph_widget.df is not None:
                self.graph_widget.plot(self.graph_widget.df)

        # Notify ViewModel of the updated legend/scatter/error-bar properties
        props = {
            'legend_properties': self.graph_widget.legend_properties,
            'legend_outside': getattr(self.graph_widget, 'legend_outside', False),
            'legend_bbox': getattr(self.graph_widget, 'legend_bbox', None),
            'legend_ncol': self.graph_widget.legend_ncol,
            'legend_frame': self.graph_widget.legend_frame,
            'legend_title': self.graph_widget.legend_title,
            'legend_fontsize': self.graph_widget.legend_fontsize,
            'legend_alpha': self.graph_widget.legend_alpha,
            'legend_loc': self.graph_widget.legend_loc,
        }
        if self.graph_widget.plot_style in ['scatter', 'trendline', 'point']:
            props['scatter_size'] = self.graph_widget.scatter_size
            props['scatter_edgecolor'] = self.graph_widget.scatter_edgecolor
        if style in _ERROR_BAR_STYLES:
            props['error_bar_capsize'] = self.graph_widget.error_bar_capsize
            if style == 'bar':
                props['bar_error_bar_type'] = self.graph_widget.bar_error_bar_type
            else:
                props['error_bar_type'] = self.graph_widget.error_bar_type
        self.graph_widget.properties_changed.emit(
            self.graph_widget.graph_id, props
        )

    def cancel_changes(self):
        """Cancel legend changes and restore original properties."""
        if self.original_legend_properties is not None:
            self.graph_widget.legend_properties = copy.deepcopy(self.original_legend_properties)
            self.graph_widget.legend_outside = getattr(self, 'original_legend_outside', False)
            self.graph_widget.legend_bbox = getattr(self, 'original_legend_bbox', None)
            self.graph_widget._set_legend()
            if self.graph_widget.df is not None:
                self.graph_widget.plot(self.graph_widget.df)
            else:
                self.graph_widget.canvas.draw()
            # Reload widgets to show restored properties
            self.load_legend_properties()
