# spectroview/view/components/v_map_list.py
"""View component for Maps list - two-level navigation (Maps → Spectra)."""
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QAbstractItemView, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon

from spectroview import ICON_DIR


class VMapsList(QWidget):
    """Two-level list widget: Maps (top) → Spectra from selected map (bottom)."""
    
    # ───── View → ViewModel signals ─────
    map_selection_changed = Signal(int)       # selected map index
    spectra_selection_changed = Signal(list)  # list of selected spectrum indices
    files_dropped = Signal(list)               # list of file paths dropped on maps list
    view_map_requested = Signal()              # view button clicked
    delete_map_requested = Signal()            # delete button clicked
    save_requested = Signal()              # info button clicked
    select_all_requested = Signal()            # select all button clicked
    reinitialize_requested = Signal()          # reinitialize button clicked
    stats_requested = Signal()                 # stats button clicked
    send_to_spectra_requested = Signal()       # send to spectra tab button clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        # ══════════════════════════════════════════════════════════════
        # MAPS LIST WITH ACTION BUTTONS
        # ══════════════════════════════════════════════════════════════
        maps_container = QHBoxLayout()
        maps_container.setSpacing(4)
        
        # Left: Maps list
        maps_list_layout = QVBoxLayout()
        maps_list_layout.setContentsMargins(0, 0, 0, 0)
        maps_list_layout.setSpacing(2)
        
        maps_label = QLabel("Maps:")
        maps_list_layout.addWidget(maps_label)
        
        self.maps_list = QListWidget()
        self.maps_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.maps_list.setAcceptDrops(True)
        self.maps_list.setMaximumHeight(150)
        self.maps_list.setMinimumHeight(80)
        
        # Enable drag & drop for external files
        self.maps_list.dragEnterEvent = self._on_maps_drag_enter
        self.maps_list.dragMoveEvent = self._on_maps_drag_move
        self.maps_list.dropEvent = self._on_maps_drop
        
        # Connect selection change
        self.maps_list.itemSelectionChanged.connect(self._on_map_selected)
        
        maps_list_layout.addWidget(self.maps_list)
        maps_container.addLayout(maps_list_layout, stretch=1)
        
        # Right: Map action buttons
        map_buttons_layout = QVBoxLayout()
        map_buttons_layout.setSpacing(4)
        
        self.btn_view = QPushButton()
        self.btn_view.setIcon(QIcon(os.path.join(ICON_DIR, "view.png")))
        self.btn_view.setToolTip("View selected map")
        self.btn_view.setFixedSize(32, 32)
        self.btn_view.clicked.connect(self.view_map_requested.emit)
        
        self.btn_delete = QPushButton()
        self.btn_delete.setIcon(QIcon(os.path.join(ICON_DIR, "trash3.png")))
        self.btn_delete.setToolTip("Delete selected map")
        self.btn_delete.setFixedSize(32, 32)
        self.btn_delete.clicked.connect(self.delete_map_requested.emit)
        
        self.btn_save = QPushButton()
        self.btn_save.setIcon(QIcon(os.path.join(ICON_DIR, "save.png")))
        self.btn_save.setToolTip("Save the selected map")
        self.btn_save.setFixedSize(32, 32)
        self.btn_save.clicked.connect(self.save_requested.emit)
        
        map_buttons_layout.addWidget(self.btn_view)
        map_buttons_layout.addWidget(self.btn_delete)
        map_buttons_layout.addWidget(self.btn_save)
        map_buttons_layout.addStretch()
        
        maps_container.addLayout(map_buttons_layout)
        main_layout.addLayout(maps_container)
        
        # ══════════════════════════════════════════════════════════════
        # SPECTRA LIST WITH ACTION BUTTONS
        # ══════════════════════════════════════════════════════════════
        spectra_container = QHBoxLayout()
        spectra_container.setSpacing(4)
        
        # Left: Spectra list
        spectra_list_layout = QVBoxLayout()
        spectra_list_layout.setContentsMargins(0, 0, 0, 0)
        spectra_list_layout.setSpacing(2)
        
        spectra_label = QLabel("Spectrum(s):")
        spectra_list_layout.addWidget(spectra_label)
        
        self.spectra_list = QListWidget()
        self.spectra_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # Connect selection change
        self.spectra_list.itemSelectionChanged.connect(self._on_spectra_selection_changed)
        
        spectra_list_layout.addWidget(self.spectra_list, stretch=1)
        spectra_container.addLayout(spectra_list_layout, stretch=1)
        
        # Right: Spectra action buttons
        spectra_buttons_layout = QVBoxLayout()
        spectra_buttons_layout.setSpacing(4)
        
        self.btn_select_all = QPushButton()
        self.btn_select_all.setIcon(QIcon(os.path.join(ICON_DIR, "select-all.png")))
        self.btn_select_all.setToolTip("Select all spectra")
        self.btn_select_all.setFixedSize(32,32)
        self.btn_select_all.clicked.connect(self.select_all_requested.emit)
        
        self.btn_reinit = QPushButton()
        self.btn_reinit.setIcon(QIcon(os.path.join(ICON_DIR, "undo2.png")))
        self.btn_reinit.setToolTip("Reinitialize selected spectra")
        self.btn_reinit.setFixedSize(32,32)
        self.btn_reinit.clicked.connect(self.reinitialize_requested.emit)
        
        self.btn_stats = QPushButton()
        self.btn_stats.setIcon(QIcon(os.path.join(ICON_DIR, "stats.png")))
        self.btn_stats.setToolTip("Show fitting statistics")
        self.btn_stats.setFixedSize(32,32)
        self.btn_stats.clicked.connect(self.stats_requested.emit)
        
        self.btn_send_to_spectra = QPushButton()
        self.btn_send_to_spectra.setIcon(QIcon(os.path.join(ICON_DIR, "send.png")))
        self.btn_send_to_spectra.setToolTip("Send selected spectra to 'Spectra' Workspace")
        self.btn_send_to_spectra.setFixedSize(32,32)
        self.btn_send_to_spectra.clicked.connect(self.send_to_spectra_requested.emit)
        
        spectra_buttons_layout.addWidget(self.btn_select_all)
        spectra_buttons_layout.addWidget(self.btn_reinit)
        spectra_buttons_layout.addWidget(self.btn_stats)
        spectra_buttons_layout.addWidget(self.btn_send_to_spectra)
        spectra_buttons_layout.addStretch()
        
        spectra_container.addLayout(spectra_buttons_layout)
        main_layout.addLayout(spectra_container, stretch=1)
    
    # ───── Public API (ViewModel → View) ─────────────────────────
    def set_maps_names(self, names: list[str]):
        """Replace entire maps list (ViewModel-driven)."""
        self.maps_list.clear()
        for i, name in enumerate(names):
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, i)  # Store index
            self.maps_list.addItem(item)
    
    def set_spectra_names(self, names: list[str]):
        """Replace spectra list for currently selected map."""
        self.spectra_list.clear()
        for i, name in enumerate(names):
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, i)  # Store index
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.spectra_list.addItem(item)
    
    def get_selected_map_index(self) -> int:
        """Return currently selected map index, or -1 if none."""
        selected = self.maps_list.selectedItems()
        if selected:
            return self.maps_list.row(selected[0])
        return -1
    
    def get_selected_spectra_indices(self) -> list[int]:
        """Return list of selected spectrum indices."""
        return [
            self.spectra_list.row(item)
            for item in self.spectra_list.selectedItems()
        ]
    
    def get_checked_spectra_indices(self) -> list[int]:
        """Return list of checked spectrum indices."""
        checked = []
        for i in range(self.spectra_list.count()):
            item = self.spectra_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(i)
        return checked
    
    def select_all_spectra(self):
        """Select all spectra in the list."""
        self.spectra_list.selectAll()
    
    def check_all_spectra(self, checked: bool):
        """Check or uncheck all spectra."""
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.spectra_list.count()):
            self.spectra_list.item(i).setCheckState(state)
    
    # ───── Internal signal handlers ──────────────────────────────
    def _on_map_selected(self):
        """Emit signal when map selection changes."""
        idx = self.get_selected_map_index()
        if idx >= 0:
            self.map_selection_changed.emit(idx)
    
    def _on_spectra_selection_changed(self):
        """Emit signal when spectra selection changes."""
        indices = self.get_selected_spectra_indices()
        self.spectra_selection_changed.emit(indices)
    
    # ───── Drag & Drop handling ──────────────────────────────────
    def _on_maps_drag_enter(self, event):
        """Accept external file drops on maps list."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def _on_maps_drag_move(self, event):
        """Allow drag movement."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def _on_maps_drop(self, event):
        """Handle file drop on maps list."""
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
