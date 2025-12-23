from PySide6.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from PySide6.QtCore import Qt, Signal


class SpectraList(QListWidget):
    """
    View component: list of spectra.
    - Supports multi-selection
    - Internal drag & drop reordering
    - External file drop
    - Emits MVVM-friendly signals
    """

    # ───── MVVM signals ──────────────────────────────────────────
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
    def set_spectra_names(self, names: list[str]):
        """Replace entire list (ViewModel-driven)."""
        self.clear()
        for name in names:
            self.addItem(QListWidgetItem(name))

    def selected_rows(self) -> list[int]:
        return [self.row(i) for i in self.selectedItems()]

    # ───── Drag & Drop handling ──────────────────────────────────
    def startDrag(self, supportedActions):
        self._order_before_drag = self._current_order()
        super().startDrag(supportedActions)

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

    # ───── Helpers ───────────────────────────────────────────────
    def _current_order(self) -> list[int]:
        """Return list of row indices in current visual order."""
        return list(range(self.count()))

    def _emit_selection_changed(self):
        self.selection_changed.emit(self.selected_rows())

    def _on_item_activated(self, item: QListWidgetItem):
        self.item_activated.emit(self.row(item))
