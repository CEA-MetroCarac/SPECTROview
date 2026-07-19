"""Legend / Color tab of the Customize Graph dialog: per-series label,
marker, color, and style editing, plus scatter marker size/edge-color
settings and error-bar options.

Split out of customize_graph_dialog.py; no behavior changes.
"""
import copy

from PySide6.QtCore import Qt, QTimer
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
# to set (mirrors the unified_marker_widget/per-series-columns visibility
# condition in load_legend_properties()/_build_legend_widgets()).
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
        self.legend_container = QGroupBox("Legend box:")
        box_layout = QVBoxLayout(self.legend_container)
        box_layout.setContentsMargins(4, 4, 4, 4)
        box_layout.setSpacing(8)

        top_row_layout = QHBoxLayout()
        
        self.cb_legend_outside = QCheckBox("Put legend box outside")
        self.cb_legend_outside.stateChanged.connect(self._on_legend_outside_changed)
        top_row_layout.addWidget(self.cb_legend_outside)

        top_row_layout.addSpacing(10)
        self.cb_legend_frame = QCheckBox("Frame")
        top_row_layout.addWidget(self.cb_legend_frame)

        top_row_layout.addSpacing(10)
        top_row_layout.addWidget(QLabel("Columns:"))
        self.spin_legend_ncol = QSpinBox()
        self.spin_legend_ncol.setRange(1, 10)
        top_row_layout.addWidget(self.spin_legend_ncol)

        top_row_layout.addSpacing(10)
        top_row_layout.addWidget(QLabel("Title:"))
        self.edit_legend_title = QLineEdit()
        self.edit_legend_title.setPlaceholderText("Optional legend title")
        top_row_layout.addWidget(self.edit_legend_title)

        top_row_layout.addStretch()
        box_layout.addLayout(top_row_layout)

        self.unify_row_widget = QWidget()
        unify_row_layout = QHBoxLayout(self.unify_row_widget)
        unify_row_layout.setContentsMargins(0, 0, 0, 0)
        unify_row_layout.setSpacing(8)

        self.cb_unify_marker_style = QCheckBox("Unify marker size / edge color")
        self.cb_unify_marker_style.setToolTip(
            "Checked: every series uses the same marker size and edge color, "
            "set below. Unchecked: each series can override them "
            "individually in the table below."
        )
        self.cb_unify_marker_style.toggled.connect(self._on_unify_marker_style_toggled)
        unify_row_layout.addWidget(self.cb_unify_marker_style)

        # Shown only while Unify is checked (see load_legend_properties()) --
        # the single shared marker size / edge color, replacing what used to
        # be its own always-visible "Scatter / Marker settings" groupbox.
        self.unified_marker_widget = QWidget()
        unified_marker_layout = QHBoxLayout(self.unified_marker_widget)
        unified_marker_layout.setContentsMargins(0, 0, 0, 0)
        unified_marker_layout.setSpacing(8)
        unified_marker_layout.addWidget(QLabel("Marker size:"))
        self.spin_scatter_size = QSpinBox()
        self.spin_scatter_size.setRange(5, 500)
        self.spin_scatter_size.setSingleStep(10)
        self.spin_scatter_size.setValue(70)
        unified_marker_layout.addWidget(self.spin_scatter_size)
        unified_marker_layout.addSpacing(10)
        unified_marker_layout.addWidget(QLabel("Edge color:"))
        self.btn_scatter_edgecolor = QPushButton()
        self.btn_scatter_edgecolor.setFixedWidth(80)
        self._set_color_button(self.btn_scatter_edgecolor, 'black')
        self.btn_scatter_edgecolor.clicked.connect(self._pick_scatter_edgecolor)
        unified_marker_layout.addWidget(self.btn_scatter_edgecolor)
        
        unify_row_layout.addWidget(self.unified_marker_widget)
        unify_row_layout.addStretch()
        box_layout.addWidget(self.unify_row_widget)

        self.legend_layout = QHBoxLayout()
        box_layout.addLayout(self.legend_layout)

        # ───── Legend box style (fontsize/alpha/location) ─────
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
        
        box_layout.addLayout(row2)

        self.main_layout.addWidget(self.legend_container)

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

        self.cb_unify_marker_style.blockSignals(True)
        self.cb_unify_marker_style.setChecked(getattr(self.graph_widget, 'unify_marker_style', True))
        self.cb_unify_marker_style.blockSignals(False)

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
        # Only shown for marker-drawing styles, and only while Unify is
        # checked (unchecked: each series's own column in the table above
        # is where marker size / edge color get set instead).
        is_marker_style = self.graph_widget.plot_style in _MARKER_STYLES
        self.cb_unify_marker_style.setVisible(is_marker_style)
        self.unified_marker_widget.setVisible(
            self.cb_unify_marker_style.isChecked() and is_marker_style
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
        alpha_layout = QVBoxLayout()
        
        show_linewidth_col = (
            self.graph_widget.plot_style in ('line', 'trendline') or
            (self.graph_widget.plot_style == 'point' and getattr(self.graph_widget, 'join_for_point_plot', False))
        )
        linewidth_layout = QVBoxLayout() if show_linewidth_col else None
        
        show_marker_col = (
            self.graph_widget.plot_style in _MARKER_STYLES
            and not getattr(self.graph_widget, 'unify_marker_style', True)
        )
        marker_size_layout = QVBoxLayout() if show_marker_col else None
        edge_color_layout = QVBoxLayout() if show_marker_col else None

        # Headers
        for header in ['Label', 'Marker', 'Color', 'Line width', 'Alpha']:
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
                if show_linewidth_col:
                    linewidth_layout.addWidget(lbl)
            elif header == 'Alpha':
                alpha_layout.addWidget(lbl)
        if show_marker_col:
            for lbl_text, layout in (('Marker size', marker_size_layout), ('Edge color', edge_color_layout)):
                lbl = QLabel(lbl_text)
                lbl.setAlignment(Qt.AlignCenter)
                layout.addWidget(lbl)

        for idx, prop in enumerate(legend_properties):
            # Label
            label = QLineEdit(prop['label'])
            label.setFixedWidth(120)
            label.textChanged.connect(
                lambda text, i=idx: self._update_legend_property(i, 'label', text)
            )
            label_layout.addWidget(label)

            # Marker (only for point plots)
            if self.graph_widget.plot_style == 'point':
                marker = QComboBox()
                marker.setFixedWidth(50)
                marker.addItems(MARKERS)
                marker.setCurrentText(prop['marker'])
                marker.currentTextChanged.connect(
                    lambda text, i=idx: self._update_legend_property(i, 'marker', text)
                )
                marker_layout.addWidget(marker)

            # Color combobox with colored items
            color = QComboBox()
            color.setFixedWidth(90)
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
            if show_linewidth_col:
                lw_spin = self._make_optional_spinbox(
                    minimum=0.0, maximum=20.0, decimals=2, step=0.5, start_value=1.5
                )
                lw_spin.setFixedWidth(60)
                lw_spin.setValue(prop.get('linewidth', self._UNSET))
                lw_spin.set_placeholder_value(1.5)
                lw_spin.valueChanged.connect(
                    lambda v, i=idx: self._update_legend_property_numeric(i, 'linewidth', v)
                )
                linewidth_layout.addWidget(lw_spin)

            alpha_spin = self._make_optional_spinbox(
                minimum=0.0, maximum=1.0, decimals=2, step=0.1, start_value=1.0
            )
            alpha_spin.setFixedWidth(60)
            alpha_spin.setValue(prop.get('alpha', self._UNSET))
            alpha_spin.set_placeholder_value(1.0)
            alpha_spin.valueChanged.connect(
                lambda v, i=idx: self._update_legend_property_numeric(i, 'alpha', v)
            )
            alpha_layout.addWidget(alpha_spin)

            if show_marker_col:
                default_marker_size = getattr(self.graph_widget, 'scatter_size', 70)
                ms_spin = self._make_optional_spinbox(
                    is_int=True, minimum=1, maximum=500, step=10, start_value=default_marker_size
                )
                ms_spin.setFixedWidth(60)
                ms_spin.setValue(int(prop.get('marker_size', self._UNSET)))
                ms_spin.set_placeholder_value(default_marker_size)
                ms_spin.valueChanged.connect(
                    lambda v, i=idx: self._update_legend_property_numeric(i, 'marker_size', v)
                )
                marker_size_layout.addWidget(ms_spin)

                edge_btn = QPushButton()
                edge_btn.setFixedWidth(70)
                edge_val = prop.get('edge_color')
                # Show the resolved color -- an unset override falls back to
                # the graph-wide scatter_edgecolor, not a generic "(default)".
                self._set_color_button(edge_btn, edge_val or getattr(self.graph_widget, 'scatter_edgecolor', 'black'))
                if not edge_val:
                    edge_btn.setToolTip("Using the global marker edge color")
                edge_btn.clicked.connect(
                    lambda *_a, i=idx, btn=edge_btn: self._pick_series_edge_color(i, btn)
                )
                edge_color_layout.addWidget(edge_btn)

        # Stretch at bottom
        label_layout.addStretch()
        if self.graph_widget.plot_style == 'point':
            marker_layout.addStretch()
        color_layout.addStretch()
        if show_linewidth_col:
            linewidth_layout.addStretch()
        alpha_layout.addStretch()
        if show_marker_col:
            marker_size_layout.addStretch()
            edge_color_layout.addStretch()

        self.legend_layout.addLayout(label_layout)
        if self.graph_widget.plot_style == 'point':
            self.legend_layout.addLayout(marker_layout)
        self.legend_layout.addLayout(color_layout)
        if show_linewidth_col:
            self.legend_layout.addLayout(linewidth_layout)
        self.legend_layout.addLayout(alpha_layout)
        if show_marker_col:
            self.legend_layout.addLayout(marker_size_layout)
            self.legend_layout.addLayout(edge_color_layout)
            
        self.legend_layout.addStretch()

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

    def _on_unify_marker_style_toggled(self, checked):
        """Unify marker size/edge color across every series (checked) vs.
        letting each series override them individually (unchecked --
        today's per-series Marker size/Edge color table columns)."""
        self.graph_widget.unify_marker_style = checked
        if self.graph_widget.df is not None:
            self.graph_widget.plot(self.graph_widget.df)
        else:
            self.graph_widget.canvas.draw_idle()
        # Rebuild the table (show/hide per-series columns) and refresh the
        # unified row's own visibility -- load_legend_properties() is the
        # single place both are already kept in sync.
        self.load_legend_properties()
        # Deferred to the next event-loop iteration: the dialog doesn't
        # auto-grow until Qt settles the old columns' deleteLater() cleanup.
        QTimer.singleShot(0, self._resize_window_to_fit_content)

    def _resize_window_to_fit_content(self):
        window = self.window()
        if window is not None:
            window.adjustSize()

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

    def apply_changes(self, replot: bool = True):
        """Apply legend changes by doing a full replot.

        `replot=False` (used by CustomizeGraphDialog._preview_apply()'s
        debounced live preview) skips both replot points below, leaving the
        combined restyle()-or-replot decision to the dialog after all four
        tabs have applied their fields, instead of each tab replotting on
        its own."""
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
        if replot:
            if self.graph_widget.df is not None:
                self.graph_widget.plot(self.graph_widget.df)
            else:
                self.graph_widget.canvas.draw_idle()

        # Update the backup so Cancel won't revert applied changes
        self.original_legend_properties = copy.deepcopy(self.graph_widget.get_legend_properties())
        self.original_legend_outside = getattr(self.graph_widget, 'legend_outside', False)
        self.original_legend_bbox = getattr(self.graph_widget, 'legend_bbox', None)

        self.graph_widget.unify_marker_style = self.cb_unify_marker_style.isChecked()

        # Apply scatter-specific properties
        if self.graph_widget.plot_style in ['scatter', 'trendline', 'point']:
            self.graph_widget.scatter_size = self.spin_scatter_size.value()
            edge_c = self.btn_scatter_edgecolor.text()
            if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
                edge_c = 'black'
            # Compare as colors, not strings (button text is hex-normalized)
            # so re-applying the same color doesn't register as a change --
            # this runs on every live-preview tick, so a string mismatch
            # alone would force a full replot every tick.
            if QColor(edge_c) != QColor(self.graph_widget.scatter_edgecolor):
                self.graph_widget.scatter_edgecolor = edge_c
            # Replot to reflect changes
            if replot and self.graph_widget.df is not None:
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
            'unify_marker_style': self.graph_widget.unify_marker_style,
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
