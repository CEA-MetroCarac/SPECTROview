"""Legend / Color tab of the Customize Graph dialog: per-series label,
marker, and color editing, plus scatter marker size/edge-color settings.

Split out of customize_graph_dialog.py; no behavior changes.
"""
import copy

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QColorDialog, QComboBox, QSpinBox, QCheckBox, QLineEdit,
)

from spectroview import DEFAULT_COLORS, MARKERS
from spectroview.view.components.customize_graph.customize_annotation_dialogs import ColorDelegate


class CustomizeLegend(QWidget):
    """Widget Tab for customizing legend properties (labels, markers, colors)"""

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

    def _setup_ui(self):
        """Setup the UI components for the legend customization widget."""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(8)

        # Container for legend widgets (labels, markers, colors)
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
            self.graph_widget.plot_style in ['scatter', 'trendline', 'point']
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
            self.graph_widget.plot_style in ['scatter', 'trendline', 'point']
        )

    def _build_legend_widgets(self, legend_properties):
        """Build the legend customization widgets (label, marker, color)."""
        label_layout = QVBoxLayout()
        marker_layout = QVBoxLayout()
        color_layout = QVBoxLayout()

        # Headers
        for header in ['Label', 'Marker', 'Color']:
            lbl = QLabel(header)
            lbl.setAlignment(Qt.AlignCenter)
            if header == 'Label':
                label_layout.addWidget(lbl)
            elif header == 'Marker':
                if self.graph_widget.plot_style == 'point':
                    marker_layout.addWidget(lbl)
            elif header == 'Color':
                color_layout.addWidget(lbl)

        for idx, prop in enumerate(legend_properties):
            # Label
            label = QLineEdit(prop['label'])
            label.setFixedWidth(200)
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
                lambda _, cb=color: self._update_combobox_color(cb)
            )
            color.currentTextChanged.connect(
                lambda text, i=idx: self._update_legend_property(i, 'color', text)
            )
            color_layout.addWidget(color)
            self._update_combobox_color(color)

        # Stretch at bottom
        label_layout.addStretch()
        if self.graph_widget.plot_style == 'point':
            marker_layout.addStretch()
        color_layout.addStretch()

        self.legend_layout.addLayout(label_layout)
        self.legend_layout.addLayout(marker_layout)
        self.legend_layout.addLayout(color_layout)

    def _update_legend_property(self, idx, property_type, text):
        """Update a legend property and refresh the plot live."""
        self.graph_widget.legend_properties[idx][property_type] = text
        self.graph_widget._set_legend()
        self.graph_widget.canvas.draw_idle()

    def _on_legend_outside_changed(self, state):
        """Update whether the legend is placed outside the plot."""
        self.graph_widget.legend_outside = self.cb_legend_outside.isChecked()
        self.graph_widget.legend_bbox = None  # Reset drag position
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

        # Notify ViewModel of the updated legend/scatter properties
        props = {
            'legend_properties': self.graph_widget.legend_properties,
            'legend_outside': getattr(self.graph_widget, 'legend_outside', False),
            'legend_bbox': getattr(self.graph_widget, 'legend_bbox', None)
        }
        if self.graph_widget.plot_style in ['scatter', 'trendline', 'point']:
            props['scatter_size'] = self.graph_widget.scatter_size
            props['scatter_edgecolor'] = self.graph_widget.scatter_edgecolor
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
