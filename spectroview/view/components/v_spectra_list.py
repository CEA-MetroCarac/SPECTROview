# spectroview/view/components/v_spectra_list.py
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from PySide6.QtCore import Qt, Signal


class VSpectraList(QListWidget):
    # ───── View → ViewModel signals ─────
    selection_changed = Signal(list)     # list of selected row indices
    order_changed = Signal(list)          # new order of row indices
    files_dropped = Signal(list)          # list of file paths
    item_activated = Signal(int)          # double-clicked row

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setAcceptDrops(True)
        self.setDefaultDropAction(Qt.MoveAction)

        # Internal state to detect reorder
        self._order_before_drag = []

        # Qt signals → our semantic signals
        self.itemSelectionChanged.connect(self._emit_selection_changed)
        self.itemDoubleClicked.connect(self._on_item_activated)

    # ───── Public API (used by ViewModel) ────────────────────────
    def set_spectra_names(self, spectra: list):
        """Replace entire list (ViewModel-driven)."""
        self.clear()
        for i, spectrum in enumerate(spectra):
            item = QListWidgetItem(spectrum.fname)
            item.setData(Qt.UserRole, i)  # model index -> used when dragging/reordering
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Set checkbox state from spectrum.is_active
            item.setCheckState(Qt.Checked if spectrum.is_active else Qt.Unchecked)
            self.addItem(item)
            
            # Connect checkbox state change to update spectrum.is_active
            # Store reference to spectrum object
            item.setData(Qt.UserRole + 1, id(spectrum))  # Store spectrum ID for lookup
    
    def itemChanged(self, item):
        """Handle checkbox state change to update spectrum.is_active."""
        # This will be connected from the workspace view
        pass

    def selected_rows(self) -> list[int]:
        return [self.row(i) for i in self.selectedItems()]
    
    def select_all(self):
        """Select all items in the list."""
        self.selectAll()
    
    def get_checked_spectra_indices(self) -> list[int]:
        """Return list of checked spectrum indices."""
        checked = []
        for i in range(self.count()):
            item = self.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(i)
        return checked
    
    def check_all_spectra(self, checked: bool):
        """Check or uncheck all spectra."""
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.count()):
            self.item(i).setCheckState(state)
    
    # ───── Drag & Drop handling ──────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        # External files dropped
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
            return

        # Internal reorder
        super().dropEvent(event)

        order_after = self._current_order()
        if order_after != self._order_before_drag:
            self.order_changed.emit(order_after)

            
    def startDrag(self, supportedActions):
        self._order_before_drag = self._current_order()
        super().startDrag(supportedActions)

    # ───── Helpers ───────────────────────────────────────────────
    def _current_order(self) -> list[int]:
        """Return model indices in current visual order."""
        return [
            self.item(row).data(Qt.UserRole)
            for row in range(self.count())
        ]


    def selected_model_indices(self) -> list[int]:
        """Return list of selected spectra indices in the Model."""
        return [
            item.data(Qt.UserRole)
            for item in self.selectedItems()
        ]

    def _emit_selection_changed(self):
        """Emit selection_changed signal with model indices of selected items."""
        self.selection_changed.emit(self.selected_model_indices())

    def _on_item_activated(self, item: QListWidgetItem):
        """Emit item_activated signal with row index."""
        self.item_activated.emit(self.row(item))
