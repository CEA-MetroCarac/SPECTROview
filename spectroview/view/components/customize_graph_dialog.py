import time
import os
import copy

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon, QColor, QPalette
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QPushButton, QListWidget, QListWidgetItem, QLabel,
    QDialogButtonBox, QMessageBox, QTabWidget, QWidget,
    QColorDialog, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLineEdit, QFormLayout, QStyledItemDelegate)

from spectroview import DEFAULT_COLORS, MARKERS
from spectroview import ICON_DIR

class CustomizeGraphDialog(QDialog):
    """Dialog for customizing graph"""
    
    def __init__(self, graph_widget, graph_id, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.graph_id = graph_id
        
        self.setWindowTitle(f"Customize Graph {graph_id}")
        self.setModal(False)
        self.resize(450, 550)
        
        self._setup_ui()
    
    
    def _setup_ui(self):
        """Setup dialog UI with tabs."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs
        tab_annotations = self._create_annotations_tab()
        tab_legend = self._create_legend_tab()
        tab_general = self._create_general_tab()
        tab_axis = self._create_axis_tab()
        
        # Add tabs to widget
        self.tabs.addTab(tab_legend, "Legend")
        self.tabs.addTab(tab_annotations, "Annotations")
        self.tabs.addTab(tab_axis, "Axis")
        self.tabs.addTab(tab_general, "General")
        
        layout.addWidget(self.tabs)

    def _create_legend_tab(self):
        """Create legend customization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.legend_widget = CustomizeLegend(self.graph_widget, parent=tab)
        layout.addWidget(self.legend_widget)
        layout.addStretch()
        return tab
    
    def _create_annotations_tab(self):
        """Create annotations tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.annotations_widget = CustomizeAnnotations(self.graph_widget, parent=tab)
        layout.addWidget(self.annotations_widget)
        return tab
    
    def _create_general_tab(self):
        """Create general settings tab (placeholder for future)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("General graph settings will be added here."))
        layout.addStretch()
        return tab
    
    def _create_axis_tab(self):
        """Create axis settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.axis_widget = CustomizeAxis(self.graph_widget, parent=tab)
        layout.addWidget(self.axis_widget)
        return tab

    def open_legend_tab(self):
        """Open the dialog and switch to the Legend tab."""
        self.legend_widget.load_legend_properties()
        self.tabs.setCurrentIndex(0) # Switch to Legend tab (index 0)
        self.show()
        self.raise_()
        self.activateWindow()


class CustomizeLegend(QWidget):
    """Widget Tab for customizing legend properties (labels, markers, colors)"""
    legend_applied = Signal(int)
    
    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.original_legend_properties = None
        
        self._setup_ui()
        
        # Load initial properties
        self.load_legend_properties()
    
    def _setup_ui(self):
        """Setup the UI components for the legend customization widget."""
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Info label
        info_label = QLabel("Customize legend labels, colors, and markers:")
        self.main_layout.addWidget(info_label)
        
        # Container for legend widgets (labels, markers, colors)
        self.legend_container = QWidget()
        self.legend_layout = QHBoxLayout(self.legend_container)
        self.legend_layout.setContentsMargins(0, 0, 0, 0)
        
        self.main_layout.addWidget(self.legend_container)
        self.main_layout.addStretch()
        
        # Apply / Cancel buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.cancel_changes)
        btn_layout.addWidget(btn_cancel)
        
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self.apply_changes)
        btn_layout.addWidget(btn_apply)
        
        self.main_layout.addLayout(btn_layout)
    
    def load_legend_properties(self):
        """Load current legend properties from graph and populate the GUI."""
        legend_properties = self.graph_widget.get_legend_properties()
        self.original_legend_properties = copy.deepcopy(legend_properties)
        
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
            return
        
        # Build legend customization widgets
        self._build_legend_widgets(legend_properties)
    
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
    
    def _update_combobox_color(self, combobox):
        """Update combobox background color to match selected color."""
        selected_color = combobox.currentText()
        color = QColor(selected_color)
        palette = combobox.palette()
        palette.setColor(QPalette.Button, color)
        palette.setColor(QPalette.ButtonText, Qt.white)
        combobox.setAutoFillBackground(True)
        combobox.setPalette(palette)
        combobox.update()
    
    def apply_changes(self):
        """Apply legend changes by doing a full replot."""
        # Full replot to ensure all changes are committed
        if self.graph_widget.df is not None:
            self.graph_widget.plot(self.graph_widget.df)
        else:
            self.graph_widget.canvas.draw_idle()
        
        # Update the backup so Cancel won't revert applied changes
        self.original_legend_properties = copy.deepcopy(self.graph_widget.get_legend_properties())
        
        # Notify whoever is listening (dialop -> workspace)
        self.legend_applied.emit(self.graph_widget.graph_id)
        
    def cancel_changes(self):
        """Cancel legend changes and restore original properties."""
        if self.original_legend_properties is not None:
            self.graph_widget.legend_properties = copy.deepcopy(self.original_legend_properties)
            self.graph_widget._set_legend()
            self.graph_widget.canvas.draw() 
            # Reload widgets to show restored properties
            self.load_legend_properties()


class CustomizeAnnotations(QWidget):
    """Widget for customizing graph annotations (vline, hline, text)."""
    
    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        
        self._setup_ui()
        
        # Connect to annotation position changed signal to update list when dragging
        self.graph_widget.annotation_position_changed.connect(self._on_annotation_dragged)
        
        # Load initial annotations
        self.load_annotations()
    
    def _setup_ui(self):
        """Setup the UI components for the annotations widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add buttons
        btn_layout = QHBoxLayout()
        
        self.btn_add_vline = QPushButton("V-Line")
        self.btn_add_vline.setIcon(QIcon(os.path.join(ICON_DIR, "add.png")))
        self.btn_add_vline.setIconSize(QSize(16, 16))
        
        self.btn_add_hline = QPushButton("H-Line")
        self.btn_add_hline.setIcon(QIcon(os.path.join(ICON_DIR, "add.png")))
        self.btn_add_hline.setIconSize(QSize(16, 16))
        
        self.btn_add_text = QPushButton("Text")
        self.btn_add_text.setIcon(QIcon(os.path.join(ICON_DIR, "add.png")))
        self.btn_add_text.setIconSize(QSize(16, 16))
        
        btn_layout.addWidget(self.btn_add_vline)
        btn_layout.addWidget(self.btn_add_hline)
        btn_layout.addWidget(self.btn_add_text)
        
        # Annotation list
        self.annotation_list = QListWidget()
        
        # Edit and Delete buttons
        mgmt_layout = QHBoxLayout()
        self.btn_edit = QPushButton("Edit")
        self.btn_edit.setIcon(QIcon(os.path.join(ICON_DIR, "edit.png")))
        self.btn_edit.setIconSize(QSize(16, 16))
        
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setIcon(QIcon(os.path.join(ICON_DIR, "trash3.png")))
        self.btn_delete.setIconSize(QSize(16, 16))
        
        mgmt_layout.addWidget(self.btn_edit)
        mgmt_layout.addWidget(self.btn_delete)
        
        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("Current Annotations:"))
        layout.addWidget(self.annotation_list)
        layout.addLayout(mgmt_layout)
        
        # Connect signals
        self.btn_add_vline.clicked.connect(self._add_vline)
        self.btn_add_hline.clicked.connect(self._add_hline)
        self.btn_add_text.clicked.connect(self._add_text)
        self.btn_edit.clicked.connect(self._edit_annotation)
        self.btn_delete.clicked.connect(self._delete_annotation)

    def _on_annotation_dragged(self, graph_id, ann_id, new_x, new_y):
        """Handle annotation position change from dragging - refresh the list widget."""
        if graph_id == self.graph_widget.graph_id:
            self.load_annotations()
    
    def _get_plot_center(self):
        """Get center coordinates of the plot."""
        ax = self.graph_widget.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        return center_x, center_y
    
    def _add_vline(self):
        """Add vertical line at plot center."""
        center_x, _ = self._get_plot_center()
        
        ann_id = f"vline_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'vline',
            'x': center_x,
            'color': 'red',
            'linestyle': '--',
            'linewidth': 1.5,
            'label': f'V-Line at x={center_x:.2f}'
        }
        
        self.graph_widget.annotations.append(annotation)
        self._refresh_plot()
        self.load_annotations()
    
    def _add_hline(self):
        """Add horizontal line at plot center."""
        _, center_y = self._get_plot_center()
        
        ann_id = f"hline_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'hline',
            'y': center_y,
            'color': 'blue',
            'linestyle': '--',
            'linewidth': 1.5,
            'label': f'H-Line at y={center_y:.2f}'
        }
        
        self.graph_widget.annotations.append(annotation)
        self._refresh_plot()
        self.load_annotations()
    
    def _add_text(self):
        """Add text annotation at plot center."""
        center_x, center_y = self._get_plot_center()
        
        ann_id = f"text_{int(time.time() * 1000000)}"
        annotation = {
            'id': ann_id,
            'type': 'text',
            'x': center_x,
            'y': center_y,
            'text': 'Text',
            'fontsize': 11,
            'color': 'black',
            'ha': 'center',
            'va': 'center',
            'bbox': {
                'facecolor': 'yellow',
                'edgecolor': 'black',
                'boxstyle': 'round,pad=0.3',
                'alpha': 0.7
            }
        }
        
        self.graph_widget.annotations.append(annotation)
        self._refresh_plot()
        self.load_annotations()
    
    def _edit_annotation(self):
        """Edit selected annotation."""
        selected = self.annotation_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select an annotation to edit.")
            return
        
        ann_id = selected.data(Qt.UserRole)
        
        # Find the annotation
        annotation = None
        for ann in self.graph_widget.annotations:
            if ann.get('id') == ann_id:
                annotation = ann
                break
        
        if not annotation:
            return
        
        # Open appropriate edit dialog based on type
        if annotation['type'] in ['vline', 'hline']:
            dialog = EditLineDialog(annotation, self)
            if dialog.exec() == QDialog.Accepted:
                # Update annotation properties
                props = dialog.get_properties()
                annotation.update(props)
                
                # Update label
                if annotation['type'] == 'vline':
                    annotation['label'] = f"V-Line at x={annotation['x']:.2f}"
                else:
                    annotation['label'] = f"H-Line at y={annotation['y']:.2f}"
                
                self._refresh_plot()
                self.load_annotations()
        
        elif annotation['type'] == 'text':
            dialog = EditTextDialog(annotation, self)
            if dialog.exec() == QDialog.Accepted:
                # Update annotation properties
                props = dialog.get_properties()
                annotation.update(props)
                
                self._refresh_plot()
                self.load_annotations()
    
    def _delete_annotation(self):
        """Delete selected annotation."""
        selected = self.annotation_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select an annotation to delete.")
            return
        
        ann_id = selected.data(Qt.UserRole)
        self.graph_widget.annotations = [
            ann for ann in self.graph_widget.annotations
            if ann.get('id') != ann_id
        ]
        
        self._refresh_plot()
        self.load_annotations()
    
    def _refresh_plot(self):
        """Refresh the plot with updated annotations."""
        self.graph_widget.ax.clear()
        if self.graph_widget.df is not None:
            self.graph_widget.plot(self.graph_widget.df)
    
    def load_annotations(self):
        """Load annotations into the list widget."""
        self.annotation_list.clear()
        
        for ann in self.graph_widget.annotations:
            if ann['type'] == 'vline':
                text = f"├ VLine @ x={ann['x']:.2f} ({ann.get('color', 'red')})"
            elif ann['type'] == 'hline':
                text = f"├ HLine @ y={ann['y']:.2f} ({ann.get('color', 'blue')})"
            elif ann['type'] == 'text':
                text = f"└ Text \"{ann['text'][:20]}...\" @ ({ann['x']:.1f},{ann['y']:.1f})"
            else:
                text = f"Unknown type: {ann['type']}"
            
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, ann['id'])
            self.annotation_list.addItem(item)


class CustomizeAxis(QWidget):
    """Widget for customizing axis settings (breaks)."""
    
    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        
        self._setup_ui()
        
        # Load current breaks
        self.load_axis_breaks()
    
    def _setup_ui(self):
        """Setup the UI components for the axis customization widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # ===== X-Axis Break Section =====
        x_break_group = QGroupBox("X-Axis Break")
        x_break_layout = QVBoxLayout()
        
        # Enable checkbox
        self.x_break_enabled = QCheckBox("Enable X-axis break")
        
        # Input fields
        x_input_layout = QFormLayout()
        self.x_break_start = QDoubleSpinBox()
        self.x_break_start.setRange(-999999, 999999)
        self.x_break_start.setDecimals(2)
        
        self.x_break_end = QDoubleSpinBox()
        self.x_break_end.setRange(-999999, 999999)
        self.x_break_end.setDecimals(2)
        
        x_input_layout.addRow("Break from:", self.x_break_start)
        x_input_layout.addRow("Break to:", self.x_break_end)
        
        x_break_layout.addWidget(self.x_break_enabled)
        x_break_layout.addLayout(x_input_layout)
        x_break_group.setLayout(x_break_layout)
        
        # ===== Y-Axis Break Section =====
        y_break_group = QGroupBox("Y-Axis Break")
        y_break_layout = QVBoxLayout()
        
        # Enable checkbox
        self.y_break_enabled = QCheckBox("Enable Y-axis break")
        
        # Input fields
        y_input_layout = QFormLayout()
        self.y_break_start = QDoubleSpinBox()
        self.y_break_start.setRange(-999999, 999999)
        self.y_break_start.setDecimals(2)
        
        self.y_break_end = QDoubleSpinBox()
        self.y_break_end.setRange(-999999, 999999)
        self.y_break_end.setDecimals(2)
        
        y_input_layout.addRow("Break from:", self.y_break_start)
        y_input_layout.addRow("Break to:", self.y_break_end)
        
        y_break_layout.addWidget(self.y_break_enabled)
        y_break_layout.addLayout(y_input_layout)
        y_break_group.setLayout(y_break_layout)
        
        # ===== Apply Button =====
        self.btn_apply_breaks = QPushButton("Apply Breaks")
        self.btn_apply_breaks.clicked.connect(self._apply_axis_breaks)
        
        # Add to main layout
        layout.addWidget(x_break_group)
        layout.addWidget(y_break_group)
        layout.addWidget(self.btn_apply_breaks)
        layout.addStretch()
    
    def load_axis_breaks(self):
        """Load current axis breaks from graph widget."""
        if not hasattr(self.graph_widget, 'axis_breaks'):
            self.graph_widget.axis_breaks = {'x': None, 'y': None}
        
        breaks = self.graph_widget.axis_breaks
        
        # Get current axis limits from the graph
        x_min, x_max, y_min, y_max = 0, 100, 0, 100  # defaults
        if hasattr(self.graph_widget, 'ax') and self.graph_widget.ax is not None:
            try:
                x_min, x_max = self.graph_widget.ax.get_xlim()
                y_min, y_max = self.graph_widget.ax.get_ylim()
            except:
                pass  # Use defaults if axis not available
        
        # Load X-axis break
        if breaks.get('x'):
            self.x_break_enabled.setChecked(True)
            self.x_break_start.setValue(breaks['x']['start'])
            self.x_break_end.setValue(breaks['x']['end'])
        else:
            self.x_break_enabled.setChecked(False)
            # Set to middle range as suggestion
            mid_x = (x_min + x_max) / 2
            range_x = (x_max - x_min) / 4
            self.x_break_start.setValue(mid_x - range_x/2)
            self.x_break_end.setValue(mid_x + range_x/2)
        
        # Load Y-axis break
        if breaks.get('y'):
            self.y_break_enabled.setChecked(True)
            self.y_break_start.setValue(breaks['y']['start'])
            self.y_break_end.setValue(breaks['y']['end'])
        else:
            self.y_break_enabled.setChecked(False)
            # Set to middle range as suggestion
            mid_y = (y_min + y_max) / 2
            range_y = (y_max - y_min) / 4
            self.y_break_start.setValue(mid_y - range_y/2)
            self.y_break_end.setValue(mid_y + range_y/2)
    
    def _apply_axis_breaks(self):
        """Apply axis breaks to the graph."""
        # Validate and save X-axis break
        if self.x_break_enabled.isChecked():
            start = self.x_break_start.value()
            end = self.x_break_end.value()
            
            if start >= end:
                QMessageBox.warning(self, "Invalid Range", 
                                  "X-axis break start must be less than end.")
                return
            
            self.graph_widget.axis_breaks['x'] = {'start': start, 'end': end}
        else:
            self.graph_widget.axis_breaks['x'] = None
        
        # Validate and save Y-axis break
        if self.y_break_enabled.isChecked():
            start = self.y_break_start.value()
            end = self.y_break_end.value()
            
            if start >= end:
                QMessageBox.warning(self, "Invalid Range", 
                                  "Y-axis break start must be less than end.")
                return
            
            self.graph_widget.axis_breaks['y'] = {'start': start, 'end': end}
        else:
            self.graph_widget.axis_breaks['y'] = None
        
        # Refresh the plot with breaks applied
        self._refresh_plot()
        
        QMessageBox.information(self, "Success", "Axis breaks applied successfully!")
    
    def _refresh_plot(self):
        """Refresh the plot with updated axis breaks."""
        self.graph_widget.ax.clear()
        if self.graph_widget.df is not None:
            self.graph_widget.plot(self.graph_widget.df)

class EditLineDialog(QDialog):
    """Dialog for editing line annotations (vline/hline)."""
    
    def __init__(self, annotation, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.setWindowTitle("Edit Line Annotation")
        self.resize(350, 200)
        
        layout = QFormLayout(self)
        
        # Color picker
        self.color_button = QPushButton()
        current_color = QColor(annotation.get('color', 'red'))
        self.color_button.setStyleSheet(f"background-color: {current_color.name()};")
        self.color_button.setText(current_color.name())
        self.color_button.clicked.connect(self._pick_color)
        
        # Line style
        self.linestyle_combo = QComboBox()
        self.linestyle_combo.addItem("Solid", "-")
        self.linestyle_combo.addItem("Dashed", "--")
        self.linestyle_combo.addItem("Dotted", ":")
        self.linestyle_combo.addItem("Dash-Dot", "-.")
        
        current_style = annotation.get('linestyle', '--')
        index = self.linestyle_combo.findData(current_style)
        if index >= 0:
            self.linestyle_combo.setCurrentIndex(index)
        
        # Line width
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.5, 5.0)
        self.linewidth_spin.setSingleStep(0.5)
        self.linewidth_spin.setValue(annotation.get('linewidth', 1.5))
        
        layout.addRow("Color:", self.color_button)
        layout.addRow("Line Style:", self.linestyle_combo)
        layout.addRow("Line Width:", self.linewidth_spin)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
    
    def _pick_color(self):
        """Open color picker dialog."""
        current_color = QColor(self.color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Line Color")
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()};")
            self.color_button.setText(color.name())
    
    def get_properties(self):
        """Return updated properties."""
        return {
            'color': self.color_button.text(),
            'linestyle': self.linestyle_combo.currentData(),
            'linewidth': self.linewidth_spin.value()
        }


class EditTextDialog(QDialog):
    """Dialog for editing text annotations."""
    
    def __init__(self, annotation, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.setWindowTitle("Edit Text Annotation")
        self.resize(400, 350)
        
        layout = QFormLayout(self)
        
        # Text content
        self.text_edit = QLineEdit()
        self.text_edit.setText(annotation.get('text', 'Text'))
        
        # Font size
        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(6, 72)
        self.fontsize_spin.setValue(annotation.get('fontsize', 12))
        
        # Text color
        self.text_color_button = QPushButton()
        current_color = QColor(annotation.get('color', 'black'))
        self.text_color_button.setStyleSheet(f"background-color: {current_color.name()};")
        self.text_color_button.setText(current_color.name())
        self.text_color_button.clicked.connect(self._pick_text_color)
        
        # Background/frame options
        self.frame_checkbox = QCheckBox("Show frame/box")
        bbox = annotation.get('bbox')
        self.frame_checkbox.setChecked(bbox is not None)
        
        # Background color picker button (similar to text color)
        self.bg_color_button = QPushButton()
        if isinstance(bbox, dict) and bbox.get('facecolor'):
            bg_color = QColor(bbox.get('facecolor'))
        else:
            bg_color = QColor('yellow')  # Default background color
        self.bg_color_button.setStyleSheet(f"background-color: {bg_color.name()};")
        self.bg_color_button.setText(bg_color.name())
        self.bg_color_button.clicked.connect(self._pick_bg_color)
        
        # Transparency slider (0-100%)
        self.transparency_slider = QSpinBox()
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setSingleStep(10)
        self.transparency_slider.setSuffix("%")

        # Get current alpha from bbox, default to 70%
        current_alpha = 0.7
        if isinstance(bbox, dict) and 'alpha' in bbox:
            current_alpha = bbox['alpha']
        self.transparency_slider.setValue(int(current_alpha * 100))
        
        layout.addRow("Text:", self.text_edit)
        layout.addRow("Font Size:", self.fontsize_spin)
        layout.addRow("Text Color:", self.text_color_button)
        layout.addRow("", self.frame_checkbox)
        layout.addRow("BG Color:", self.bg_color_button)
        layout.addRow("Transparency:", self.transparency_slider)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
    
    def _pick_text_color(self):
        """Open color picker for text color."""
        current_color = QColor(self.text_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Text Color")
        if color.isValid():
            self.text_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.text_color_button.setText(color.name())
    
    def _pick_bg_color(self):
        """Open color picker for background color."""
        current_color = QColor(self.bg_color_button.text())
        color = QColorDialog.getColor(current_color, self, "Select Background Color")
        if color.isValid():
            self.bg_color_button.setStyleSheet(f"background-color: {color.name()};")
            self.bg_color_button.setText(color.name())
    
    def get_properties(self):
        """Return updated properties."""
        props = {
            'text': self.text_edit.text(),
            'fontsize': self.fontsize_spin.value(),
            'color': self.text_color_button.text(),
            'ha': 'center',  # Always use center alignment
            'va': 'center'   # Always use center alignment
        }
        
        # Handle bbox
        if self.frame_checkbox.isChecked():
            facecolor = self.bg_color_button.text()
            
            # Get transparency from slider (convert percentage to 0-1 range)
            alpha = self.transparency_slider.value() / 100.0
            
            props['bbox'] = {
                'facecolor': facecolor,
                'edgecolor': 'black',
                'boxstyle': 'round,pad=0.3',
                'alpha': alpha
            }
        else:
            props['bbox'] = None
        
        return props

class ColorDelegate(QStyledItemDelegate):
    """Show color in background of color selector comboboxes."""
    
    def paint(self, painter, option, index):
        painter.save()
        color = index.data(Qt.BackgroundRole)
        if color:
            painter.fillRect(option.rect, color)
        painter.drawText(option.rect, Qt.AlignCenter, index.data(Qt.DisplayRole))
        painter.restore()
    
    def sizeHint(self, option, index):
        return QSize(70, 20)