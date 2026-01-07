# spectroview/view/components/v_map_list.py
"""View component for Maps list - two-level navigation (Maps → Spectra)."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal


class VMapsList(QWidget):
    """Two-level list widget: Maps (top) → Spectra from selected map (bottom)."""
    
    # ───── View → ViewModel signals ─────
    map_selection_changed = Signal(int)       # selected map index
    spectra_selection_changed = Signal(list)  # list of selected spectrum indices
    files_dropped = Signal(list)               # list of file paths dropped on maps list
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # ── Upper list: Maps ──
        maps_label = QLabel("Maps:")
        maps_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(maps_label)
        
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
        
        layout.addWidget(self.maps_list)
        
        # ── Lower list: Spectra from selected map ──
        spectra_label = QLabel("Spectrum(s):")
        spectra_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(spectra_label)
        
        self.spectra_list = QListWidget()
        self.spectra_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # Connect selection change
        self.spectra_list.itemSelectionChanged.connect(self._on_spectra_selection_changed)
        
        layout.addWidget(self.spectra_list, stretch=1)
    
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
