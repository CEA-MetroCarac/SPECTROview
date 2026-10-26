from PySide6.QtWidgets import (
    QListWidget, QListWidgetItem, QAbstractItemView, QStyledItemDelegate,
    QStyleOptionViewItem,
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QFont, QPainter, QColor

from spectroview.viewmodel.utils import set_spectrum_item_color, MetadataTooltipManager
from spectroview.view.components.v_range_selector import (
    VRangeSelectorPanel,
    RangeSelectorManager,
    RangeSelectorState,
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    SELECTOR_KEYS,
)

__all__ = [
    "VSpectraList",
    "SpectrumItemDelegate",
    "VRangeSelectorPanel",
    "RangeSelectorManager",
    "RangeSelectorState",
    "SELECTOR_RED",
    "SELECTOR_BLUE",
    "SELECTOR_GREEN",
    "SELECTOR_KEYS",
]


class SpectrumItemDelegate(QStyledItemDelegate):
    """Custom delegate to draw status indicator dots on the right side of spectra items.

    Instead of filling the entire item background with color, this delegate renders
    a neat circular or square indicator dot on the right side of each item, preserving
    theme backgrounds and selection styling while clearly displaying processing status.
    """

    def __init__(self, parent=None, shape: str = "circle", size: int = 8, margin_right: int = 10):
        super().__init__(parent)
        self.shape = shape
        self.size = size
        self.margin_right = margin_right

    def paint(self, painter, option, index):
        # Anchor to visible viewport width so dots stay pinned on the right even with long names
        parent = self.parent()
        visible_right = parent.viewport().width() if (parent and hasattr(parent, 'viewport')) else option.rect.right()
        right_edge = min(option.rect.right(), visible_right)
        reserve = self.size + self.margin_right + 6

        # Reserve space on right for indicator so long filename text does not collide
        opt = QStyleOptionViewItem(option)
        if opt.rect.right() > right_edge - reserve:
            opt.rect.setRight(right_edge - reserve)
        super().paint(painter, opt, index)

        bg_brush = index.data(Qt.BackgroundRole)
        if not bg_brush:
            return
        color = bg_brush.color() if hasattr(bg_brush, 'color') else QColor(bg_brush)
        if not (hasattr(color, 'alpha') and color.alpha() > 0):
            return

        painter.save()
        painter.setClipping(False)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        # Draw status indicator badge with full opacity
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 255))

        s = self.size
        x = right_edge - self.margin_right - s
        y = option.rect.center().y() - s / 2.0

        if self.shape == "circle":
            painter.drawEllipse(QRectF(x, y, s, s))
        elif self.shape == "rounded_square":
            painter.drawRoundedRect(QRectF(x, y, s, s), 2.0, 2.0)
        elif self.shape == "square":
            painter.drawRect(QRectF(x, y, s, s))
        painter.restore()


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

        # Metadata tooltip support
        self._metadata_provider = None
        self._tooltip_manager = MetadataTooltipManager(self, self._get_item_metadata)

        # Qt signals → our semantic signals
        self.itemSelectionChanged.connect(self._emit_selection_changed)

    def set_metadata_provider(self, provider):
        """Set a custom metadata provider callable for tooltips."""
        self._metadata_provider = provider

    def set_indicator_shape(self, shape: str):
        """Set status indicator shape ('circle', 'square', or 'rounded_square')."""
        delegate = self.itemDelegate()
        if isinstance(delegate, SpectrumItemDelegate):
            delegate.shape = shape
            self.viewport().update()

    def _get_item_metadata(self, item):
        """Retrieve metadata for a spectrum item."""
        if self._metadata_provider:
            try:
                meta = self._metadata_provider(item)
            except TypeError:
                meta = self._metadata_provider(item.text())
            if meta:
                return meta
        return item.data(Qt.UserRole + 2) or {}
    
    def _update_placeholder(self):
        """Update placeholder text based on list state."""
        if self.count() == 0:
            # Add 6 empty lines before placeholder for spacing
            for _ in range(6):
                spacer = QListWidgetItem("")
                spacer.setFlags(Qt.NoItemFlags)
                self.addItem(spacer)
            
            # Add the centered placeholder item with larger text
            placeholder = QListWidgetItem("Drag and drop file(s) anywhere to open")
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
            item.setData(Qt.UserRole + 2, spectrum.get("metadata", {}))
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
