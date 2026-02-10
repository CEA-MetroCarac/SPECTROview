import time
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QPushButton, QListWidget, QListWidgetItem, QLabel,
                               QDialogButtonBox, QMessageBox, QTabWidget, QWidget)
from PySide6.QtCore import Qt


class CustomizeGraphDialog(QDialog):
    """Dialog for customizing graph including annotations."""
    
    def __init__(self, graph_widget, graph_id, parent=None):
        super().__init__(parent)
        self.graph_widget = graph_widget
        self.graph_id = graph_id
        
        self.setWindowTitle(f"Customize Graph {graph_id}")
        self.resize(450, 550)
        
        # Make dialog non-modal and always on top
        #self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setModal(False)
        
        self._setup_ui()
        self._load_annotations()
        
        # Connect to annotation position changed signal to update list when dragging
        self.graph_widget.annotation_position_changed.connect(self._on_annotation_dragged)
    
    
    def _setup_ui(self):
        """Setup dialog UI with tabs."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs
        tab_annotations = self._create_annotations_tab()
        tab_general = self._create_general_tab()
        tab_axis = self._create_axis_tab()
        
        # Add tabs to widget
        self.tabs.addTab(tab_annotations, "Annotations")
        self.tabs.addTab(tab_general, "General")
        self.tabs.addTab(tab_axis, "Axis")
        
        layout.addWidget(self.tabs)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)
    
    def _create_annotations_tab(self):
        """Create annotations tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add buttons
        btn_layout = QHBoxLayout()
        self.btn_add_vline = QPushButton("Add Vertical Line")
        self.btn_add_hline = QPushButton("Add Horizontal Line")
        self.btn_add_text = QPushButton("Add Text")
        btn_layout.addWidget(self.btn_add_vline)
        btn_layout.addWidget(self.btn_add_hline)
        btn_layout.addWidget(self.btn_add_text)
        
        # Annotation list
        self.annotation_list = QListWidget()
        
        # Delete button
        self.btn_delete = QPushButton("Delete Selected")
        
        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("Current Annotations:"))
        layout.addWidget(self.annotation_list)
        layout.addWidget(self.btn_delete)
        
        # Connect signals
        self.btn_add_vline.clicked.connect(self._add_vline)
        self.btn_add_hline.clicked.connect(self._add_hline)
        self.btn_add_text.clicked.connect(self._add_text)
        self.btn_delete.clicked.connect(self._delete_annotation)
        
        return tab
    
    def _create_general_tab(self):
        """Create general settings tab (placeholder for future)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("General graph settings will be added here."))
        layout.addStretch()
        return tab
    
    def _create_axis_tab(self):
        """Create axis settings tab (placeholder for future)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Axis configuration will be added here."))
        layout.addStretch()
        return tab
    
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
        self._load_annotations()
    
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
        self._load_annotations()
    
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
            'fontsize': 12,
            'color': 'black',
            'ha': 'center',
            'va': 'center'
        }
        
        self.graph_widget.annotations.append(annotation)
        self._refresh_plot()
        self._load_annotations()
    
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
        self._load_annotations()
    
    def _refresh_plot(self):
        """Refresh the plot with updated annotations."""
        self.graph_widget.ax.clear()
        if self.graph_widget.df is not None:
            self.graph_widget.plot(self.graph_widget.df)
    
    def _load_annotations(self):
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
    
    def _on_annotation_dragged(self, graph_id, ann_id, new_x, new_y):
        """Handle annotation position change from dragging - refresh the list."""
        # Only update if this is our graph
        if graph_id == self.graph_id:
            self._load_annotations()
