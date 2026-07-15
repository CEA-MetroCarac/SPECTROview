# spectroview/view/components/v_spectra_list.py
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QStyledItemDelegate
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from spectroview.viewmodel.utils import set_spectrum_item_color


class SpectrumItemDelegate(QStyledItemDelegate):
    """Custom delegate to draw background colors that Qt Stylesheets hide."""
    def paint(self, painter, option, index):
        from PySide6.QtWidgets import QStyle
        is_selected = option.state & QStyle.State_Selected
        
        bg_brush = index.data(Qt.BackgroundRole)
        # Only draw the custom state background if the item is NOT selected.
        # This allows the bright blue selection highlight to be fully visible.
        if bg_brush and not is_selected:
            color = bg_brush.color() if hasattr(bg_brush, 'color') else bg_brush
            if hasattr(color, 'alpha') and color.alpha() > 0:
                painter.save()
                painter.setPen(Qt.NoPen)
                painter.setBrush(bg_brush)
                # Match the margin: 1px 2px and border-radius: 4px from stylesheet
                painter.drawRoundedRect(option.rect.adjusted(2, 1, -2, -1), 4, 4)
                painter.restore()
        super().paint(painter, option, index)

class VSpectraList(QListWidget):
    # ───── View → ViewModel signals ─────
    selection_changed = Signal(list)     # list of selected row indices
    order_changed = Signal(list)          # new order of row indices

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(SpectrumItemDelegate(self))

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setAcceptDrops(True)
        self.setDefaultDropAction(Qt.MoveAction)
        
        # Set text to inform users about drag-and-drop feature
        self._update_placeholder()

        # Internal state to detect reorder
        self._order_before_drag = []

        # Qt signals → our semantic signals
        self.itemSelectionChanged.connect(self._emit_selection_changed)
    
    def _update_placeholder(self):
        """Update placeholder text based on list state."""
        if self.count() == 0:
            # Add 6 empty lines before placeholder for spacing
            for _ in range(6):
                spacer = QListWidgetItem("")
                spacer.setFlags(Qt.NoItemFlags)
                self.addItem(spacer)
            
            # Add the centered placeholder item with larger text
            placeholder = QListWidgetItem("📂 Drag and drop file(s) anywhere to open")
            placeholder.setFlags(Qt.NoItemFlags)  # Make it non-selectable and non-editable
            placeholder.setForeground(Qt.gray)
            placeholder.setTextAlignment(Qt.AlignCenter)  # Center the text horizontally
            
            # Set larger font size
            
            font = QFont()
            font.setPointSize(12)  # Increase font size
            placeholder.setFont(font)
            
            self.addItem(placeholder)
            
            self._has_placeholder = True
        else:
            # Remove all placeholder items if they exist
            if hasattr(self, '_has_placeholder') and self._has_placeholder:
                # Clear all items with NoItemFlags (placeholders and spacers)
                i = 0
                while i < self.count():
                    if self.item(i).flags() == Qt.NoItemFlags:
                        self.takeItem(i)
                    else:
                        i += 1
                self._has_placeholder = False

    # ───── Public API (used by ViewModel) ────────────────────────
    def set_spectra_names(self, spectra: list):
        """Replace entire list (ViewModel-driven)."""
        # Save current selection by fname (more robust than indices)
        selected_fnames = []
        for item in self.selectedItems():
            selected_fnames.append(item.text())
        
        # Block signals to prevent selection change cascade
        self.blockSignals(True)
        
        self.clear()
        self._has_placeholder = False  # Reset placeholder flag

        items_by_fname = {}
        for i, spectrum in enumerate(spectra):
            fname = spectrum["fname"]
            is_active = spectrum["is_active"]

            item = QListWidgetItem(fname)
            item.setData(Qt.UserRole, i)  # model index -> used when dragging/reordering
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Set checkbox state from spectrum.is_active
            item.setCheckState(Qt.Checked if is_active else Qt.Unchecked)

            # Set background color based on spectrum status
            set_spectrum_item_color(item, spectrum)

            self.addItem(item)

            # Connect checkbox state change to update spectrum.is_active
            # Store reference to fname instead of ID
            item.setData(Qt.UserRole + 1, fname)
            items_by_fname[fname] = item

        # Restore selection by matching fnames (O(1) lookup per fname instead
        # of an O(count) scan, which made this loop quadratic overall)
        selection_restored = False
        for fname in selected_fnames:
            item = items_by_fname.get(fname)
            if item is not None:
                item.setSelected(True)
                selection_restored = True

        # If no selection was restored and list is not empty, select first item
        if not selection_restored and self.count() > 0:
            self.item(0).setSelected(True)
        
        # Unblock signals and manually emit selection changed
        self.blockSignals(False)
        self._emit_selection_changed()
        
        # Update placeholder only if list is empty after adding items
        if len(spectra) == 0:
            self._update_placeholder()
    
    def select_all(self):
        """Select all items in the list."""
        self.selectAll()

    # ───── Drag & Drop handling ──────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.ignore()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.ignore()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        # External files dropped - ignore them so they propagate to parent
        if event.mimeData().hasUrls():
            event.ignore()
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
