"""Virtualized spectrum list widget using QListView + QAbstractListModel."""

from __future__ import annotations

from PySide6.QtCore import (
    QAbstractListModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QPoint,
    QRect,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from spectroview.i18n import on_language_changed, tr
from spectroview.view.components.lists.v_metadata_tooltip import MetadataTooltipManager

__all__ = [
    "SpectraListModel",
    "SpectrumItemDelegate",
    "VSpectraList",
]

_PLACEHOLDER_SPACERS = 6
_PLACEHOLDER_ROW = _PLACEHOLDER_SPACERS
_PLACEHOLDER_COUNT = _PLACEHOLDER_SPACERS + 1
_USER_ROLE = int(Qt.UserRole)
_USER_ROLE_1 = _USER_ROLE + 1
_USER_ROLE_2 = _USER_ROLE + 2
_BG_ROLE = int(Qt.BackgroundRole)


def _spectrum_status_color(spectrum_info: dict) -> QColor:
    """Compute list-item status indicator color based on processing state.

    Status hierarchy:
    1. Fitted (has_fit): converged -> success tint, unconverged -> error tint
    2. Baselined (has_baseline): accent tint
    3. Cropped (is_cropped): muted tint
    4. Original/Reinit: transparent
    """
    is_cropped = spectrum_info.get("is_cropped", False)
    has_baseline = spectrum_info.get("has_baseline", False)
    has_fit = spectrum_info.get("has_fit", False)
    fit_success = spectrum_info.get("fit_success", False)

    if not (has_fit or has_baseline or is_cropped):
        return QColor(0, 0, 0, 0)

    from spectroview.view.theme import current_tokens

    tok = current_tokens()

    if has_fit:
        color = QColor(tok.success if fit_success else tok.error)
        color.setAlpha(120)
        return color
    if has_baseline:
        color = QColor(tok.accent)
        color.setAlpha(120)
        return color
    color = QColor(tok.text_muted)
    color.setAlpha(100)
    return color


class _VirtualSpectrumItem:
    """Lightweight on-demand proxy providing QListWidgetItem-compatible row access."""

    __slots__ = ("_row", "_view")

    def __init__(self, list_view: VSpectraList, row: int):
        self._view = list_view
        self._row = row

    def text(self) -> str:
        return self._view._model.item_text(self._row)

    def data(self, role: int = Qt.UserRole):
        return self._view._model.item_data(self._row, role)

    def setData(self, role: int, value):
        self._view._model.set_item_data(self._row, role, value)

    def flags(self) -> Qt.ItemFlag:
        idx = self._view._model.index(self._row, 0)
        return self._view._model.flags(idx)

    def checkState(self) -> Qt.CheckState:
        return self._view._model.item_check_state(self._row)

    def setCheckState(self, state: Qt.CheckState):
        checked = state == Qt.Checked or state == Qt.Checked.value
        self._view._model.set_item_checked(
            self._row, checked, emit_change=not self._view.signalsBlocked()
        )

    def background(self) -> QBrush:
        return QBrush(self._view._model.status_color(self._row))

    def setBackground(self, color_or_brush):
        brush = color_or_brush if isinstance(color_or_brush, QBrush) else QBrush(color_or_brush)
        self._view._model.set_item_data(self._row, Qt.BackgroundRole, brush)

    def isSelected(self) -> bool:
        sel = self._view.selectionModel()
        if sel is None:
            return False
        return sel.isSelected(self._view._model.index(self._row, 0))

    def setSelected(self, selected: bool):
        sel = self._view.selectionModel()
        if sel is None:
            return
        flag = QItemSelectionModel.Select if selected else QItemSelectionModel.Deselect
        sel.select(self._view._model.index(self._row, 0), flag)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _VirtualSpectrumItem)
            and self._view is other._view
            and self._row == other._row
        )

    def __hash__(self) -> int:
        return hash((id(self._view), self._row))


class SpectraListModel(QAbstractListModel):
    """Virtualized Qt list model for discrete spectra with status indicators."""

    item_check_toggled = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spectra: list[dict] = []
        self._fnames: list[str] = []
        self._is_active: list[bool] = []
        self._model_indices: list[int] = []
        self._custom_roles: dict[tuple[int, int], object] = {}
        self.fname_to_row: dict[str, int] = {}
        self._has_placeholder: bool = True

        self._placeholder_font = QFont()
        self._placeholder_font.setPointSize(12)
        self._placeholder_fg = QBrush(Qt.gray)

    @property
    def has_placeholder(self) -> bool:
        return self._has_placeholder

    @property
    def fnames(self) -> list[str]:
        return self._fnames

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        if self._has_placeholder:
            return _PLACEHOLDER_COUNT
        return len(self._spectra)

    def set_spectra(self, spectra: list[dict]):
        """Replace model contents atomically using model reset."""
        self.beginResetModel()
        self._custom_roles.clear()
        if not spectra:
            self._has_placeholder = True
            self._spectra = []
            self._fnames = []
            self._is_active = []
            self._model_indices = []
            self.fname_to_row = {}
        else:
            self._has_placeholder = False
            self._spectra = list(spectra)
            self._fnames = [s["fname"] for s in spectra]
            self._is_active = [bool(s.get("is_active", True)) for s in spectra]
            self._model_indices = list(range(len(spectra)))
            self.fname_to_row = {fname: i for i, fname in enumerate(self._fnames)}
        self.endResetModel()

    def set_all_active(self, checked: bool):
        """Update checkbox state across all spectra in O(N) Python without Qt widget churn."""
        if self._has_placeholder or not self._spectra:
            return
        state = bool(checked)
        self._is_active = [state] * len(self._spectra)
        for spec in self._spectra:
            spec["is_active"] = state
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._spectra) - 1, 0)
        self.dataChanged.emit(top_left, bottom_right, [Qt.CheckStateRole])

    def notify_placeholder_retranslated(self):
        """Notify view to repaint placeholder text on UI language change."""
        if self._has_placeholder:
            idx = self.index(_PLACEHOLDER_ROW, 0)
            self.dataChanged.emit(idx, idx, [Qt.DisplayRole])

    def item_text(self, row: int) -> str:
        if self._has_placeholder:
            if row == _PLACEHOLDER_ROW:
                return tr("Drag and drop file(s) anywhere to open")
            return ""
        if 0 <= row < len(self._fnames):
            custom = self._custom_roles.get((row, Qt.DisplayRole))
            if custom is not None:
                return str(custom)
            return self._fnames[row]
        return ""

    def item_check_state(self, row: int) -> Qt.CheckState:
        if not self._has_placeholder and 0 <= row < len(self._is_active):
            return Qt.Checked if self._is_active[row] else Qt.Unchecked
        return Qt.Unchecked

    def set_item_checked(self, row: int, checked: bool, emit_change: bool = True) -> bool:
        if self._has_placeholder or not (0 <= row < len(self._is_active)):
            return False
        state = bool(checked)
        if self._is_active[row] == state:
            return False
        self._is_active[row] = state
        self._spectra[row]["is_active"] = state
        idx = self.index(row, 0)
        self.dataChanged.emit(idx, idx, [Qt.CheckStateRole])
        if emit_change:
            self.item_check_toggled.emit(row)
        return True

    def model_index(self, row: int) -> int:
        if 0 <= row < len(self._model_indices):
            if self._custom_roles:
                custom = self._custom_roles.get((row, _USER_ROLE))
                if custom is not None:
                    return int(custom)
            return self._model_indices[row]
        return row

    def status_color(self, row: int) -> QColor:
        if self._has_placeholder or not (0 <= row < len(self._spectra)):
            return QColor(0, 0, 0, 0)
        if self._custom_roles:
            custom = self._custom_roles.get((row, _BG_ROLE))
            if custom is not None:
                return custom.color() if hasattr(custom, "color") else QColor(custom)
        return _spectrum_status_color(self._spectra[row])

    def item_data(self, row: int, role: int):
        if self._has_placeholder:
            if role == Qt.DisplayRole:
                return self.item_text(row)
            if row == _PLACEHOLDER_ROW:
                if role == Qt.ForegroundRole:
                    return self._placeholder_fg
                if role == Qt.TextAlignmentRole:
                    return Qt.AlignCenter
                if role == Qt.FontRole:
                    return self._placeholder_font
            return None

        if not (0 <= row < len(self._spectra)):
            return None

        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self.item_text(row)
        if role == Qt.CheckStateRole:
            return Qt.Checked if self._is_active[row] else Qt.Unchecked
        if role == Qt.BackgroundRole:
            color = self.status_color(row)
            if color.alpha() > 0:
                return QBrush(color)
            return None
        if role == Qt.UserRole:
            return self.model_index(row)
        if not self._custom_roles:
            if role == Qt.UserRole + 1:
                return self._fnames[row]
            if role == Qt.UserRole + 2:
                return self._spectra[row].get("metadata", {})
            return None
        role_int = int(role)
        if role_int == _USER_ROLE_1:
            custom = self._custom_roles.get((row, _USER_ROLE_1))
            return custom if custom is not None else self._fnames[row]
        if role_int == _USER_ROLE_2:
            custom = self._custom_roles.get((row, _USER_ROLE_2))
            return custom if custom is not None else self._spectra[row].get("metadata", {})
        return self._custom_roles.get((row, role_int))

    def set_item_data(self, row: int, role: int, value) -> bool:
        if self._has_placeholder or not (0 <= row < len(self._spectra)):
            return False
        if role == Qt.CheckStateRole:
            checked = value == Qt.Checked or value == Qt.Checked.value
            return self.set_item_checked(row, checked, emit_change=True)
        self._custom_roles[(row, int(role))] = value
        idx = self.index(row, 0)
        self.dataChanged.emit(idx, idx, [role])
        return True

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        return self.item_data(index.row(), role)

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:
        if not index.isValid():
            return False
        return self.set_item_data(index.row(), role, value)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if self._has_placeholder:
            return Qt.NoItemFlags
        if not index.isValid():
            return Qt.ItemIsDropEnabled
        return (
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsDragEnabled
        )

    def supportedDropActions(self) -> Qt.DropAction:
        return Qt.MoveAction

    def reorder_rows(self, source_rows: list[int], dest_row: int) -> list[int]:
        """Move selected rows to ``dest_row`` and return their new row positions."""
        if self._has_placeholder or not source_rows:
            return []

        n = len(self._spectra)
        source_set = set(r for r in source_rows if 0 <= r < n)
        if not source_set:
            return []

        ordered_sources = sorted(source_set)
        dest_row = max(0, min(dest_row, n))
        insert_pos = dest_row - sum(1 for r in ordered_sources if r < dest_row)

        moving_indices = ordered_sources
        remaining_indices = [r for r in range(n) if r not in source_set]
        new_perm = remaining_indices[:insert_pos] + moving_indices + remaining_indices[insert_pos:]
        if new_perm == list(range(n)):
            return ordered_sources

        self.layoutAboutToBeChanged.emit()
        self._spectra = [self._spectra[i] for i in new_perm]
        self._fnames = [self._fnames[i] for i in new_perm]
        self._is_active = [self._is_active[i] for i in new_perm]
        self._model_indices = [self._model_indices[i] for i in new_perm]
        self.fname_to_row = {fname: i for i, fname in enumerate(self._fnames)}

        if self._custom_roles:
            old_to_new = {old_r: new_r for new_r, old_r in enumerate(new_perm)}
            self._custom_roles = {
                (old_to_new[old_r], role): val
                for (old_r, role), val in self._custom_roles.items()
                if old_r in old_to_new
            }

        self.layoutChanged.emit()
        return list(range(insert_pos, insert_pos + len(moving_indices)))


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
        visible_right = parent.viewport().width() if (parent and hasattr(parent, "viewport")) else option.rect.right()
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
        color = bg_brush.color() if hasattr(bg_brush, "color") else QColor(bg_brush)
        if not (hasattr(color, "alpha") and color.alpha() > 0):
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


class VSpectraList(QListView):
    # ───── View -> ViewModel signals ─────
    selection_changed = Signal(list)     # list of selected row indices
    order_changed = Signal(list)         # new order of row indices
    itemChanged = Signal(object)         # _VirtualSpectrumItem when check state changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = SpectraListModel(self)
        self.setModel(self._model)
        self.setItemDelegate(SpectrumItemDelegate(self))

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setUniformItemSizes(False)

        # Internal state to detect reorder
        self._order_before_drag: list[int] = []

        # Metadata tooltip support
        self._metadata_provider = None
        self._tooltip_manager = MetadataTooltipManager(self, self._get_item_metadata)

        # Placeholder localization
        self._update_placeholder()
        on_language_changed(self, self._retranslate_placeholder)

        # Qt signals -> semantic signals
        self._model.item_check_toggled.connect(self._on_item_check_toggled)
        sel_model = self.selectionModel()
        if sel_model is not None:
            sel_model.selectionChanged.connect(lambda _sel, _desel: self._emit_selection_changed())

    # ───── QListWidget Compatibility Helpers ────────────────────
    def count(self) -> int:
        return self._model.rowCount()

    def clear(self):
        self._model.set_spectra([])
        self.setUniformItemSizes(False)

    def item(self, row: int) -> _VirtualSpectrumItem | None:
        if 0 <= row < self._model.rowCount():
            return _VirtualSpectrumItem(self, row)
        return None

    def itemAt(self, pos_or_x, y: int | None = None) -> _VirtualSpectrumItem | None:
        pos = QPoint(pos_or_x, y) if y is not None else pos_or_x
        idx = self.indexAt(pos)
        if idx.isValid():
            return _VirtualSpectrumItem(self, idx.row())
        return None

    def visualItemRect(self, item: _VirtualSpectrumItem) -> QRect:
        if isinstance(item, _VirtualSpectrumItem) and 0 <= item._row < self._model.rowCount():
            return self.visualRect(self._model.index(item._row, 0))
        return QRect()

    def selectedItems(self) -> list[_VirtualSpectrumItem]:
        if self._model.has_placeholder:
            return []
        return [_VirtualSpectrumItem(self, r) for r in self._selected_rows()]

    def _selected_rows(self) -> list[int]:
        """Return selected visual row indices in O(ranges) via QItemSelection."""
        if self._model.has_placeholder:
            return []
        sel_model = self.selectionModel()
        if sel_model is None:
            return []
        selection = sel_model.selection()
        if not selection:
            return []
        if len(selection) == 1:
            rng = selection[0]
            return list(range(rng.top(), rng.bottom() + 1))
        rows: list[int] = []
        seen: set[int] = set()
        for rng in selection:
            for r in range(rng.top(), rng.bottom() + 1):
                if r not in seen:
                    seen.add(r)
                    rows.append(r)
        return rows

    def _select_rows(self, rows: list[int]):
        """Select a list of row indices using coalesced QItemSelection ranges."""
        sel_model = self.selectionModel()
        if sel_model is None or not rows:
            return
        sorted_rows = sorted(set(rows))
        selection = QItemSelection()
        start = prev = sorted_rows[0]
        for r in sorted_rows[1:]:
            if r == prev + 1:
                prev = r
            else:
                selection.select(self._model.index(start, 0), self._model.index(prev, 0))
                start = prev = r
        selection.select(self._model.index(start, 0), self._model.index(prev, 0))
        sel_model.select(selection, QItemSelectionModel.ClearAndSelect)
        sel_model.setCurrentIndex(
            self._model.index(rows[0], 0), QItemSelectionModel.NoUpdate
        )

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
        """Update placeholder state when the list is empty."""
        if self._model.has_placeholder:
            self.setUniformItemSizes(False)

    def _retranslate_placeholder(self):
        self._model.notify_placeholder_retranslated()

    def _on_item_check_toggled(self, row: int):
        """Emit QListWidget-compatible itemChanged signal when a checkbox toggles."""
        item = self.item(row)
        if item is not None:
            self.itemChanged.emit(item)

    # ───── Public API (used by ViewModel) ────────────────────────
    def set_spectra_names(self, spectra: list):
        """Replace entire list in O(N) Python without allocating C++ QListWidgetItems."""
        selected_fnames: list[str] = []
        if not self._model.has_placeholder:
            fnames = self._model.fnames
            n_fnames = len(fnames)
            for row in self._selected_rows():
                if 0 <= row < n_fnames:
                    selected_fnames.append(fnames[row])

        sel_model = self.selectionModel()
        was_blocked = self.blockSignals(True)
        sel_was_blocked = sel_model.blockSignals(True) if sel_model is not None else False
        try:
            self.setUniformItemSizes(len(spectra) > 0)
            self._model.set_spectra(spectra)

            if spectra:
                fname_to_row = self._model.fname_to_row
                target_rows = [
                    fname_to_row[fname]
                    for fname in selected_fnames
                    if fname in fname_to_row
                ]
                if not target_rows:
                    target_rows = [0]
                self._select_rows(target_rows)
        finally:
            if sel_model is not None:
                sel_model.blockSignals(sel_was_blocked)
            self.blockSignals(was_blocked)

        self._emit_selection_changed()

    def set_all_checked(self, checked: bool):
        """Check or uncheck all spectra without triggering per-item signals."""
        was_blocked = self.blockSignals(True)
        try:
            self._model.set_all_active(checked)
        finally:
            self.blockSignals(was_blocked)

    def select_all(self):
        """Select all items in the list in O(1) via a single QItemSelection range."""
        if self._model.has_placeholder:
            return
        n = self._model.rowCount()
        sel_model = self.selectionModel()
        if n <= 0 or sel_model is None:
            return
        selection = QItemSelection(self._model.index(0, 0), self._model.index(n - 1, 0))
        sel_model.select(selection, QItemSelectionModel.ClearAndSelect)

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

        if self._model.has_placeholder or event.source() is not self:
            super().dropEvent(event)
            return

        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        idx = self.indexAt(pos)
        drop_pos = self.dropIndicatorPosition()
        if not idx.isValid() or drop_pos == QAbstractItemView.OnViewport:
            target_row = self._model.rowCount()
        elif drop_pos == QAbstractItemView.BelowItem:
            target_row = idx.row() + 1
        else:
            target_row = idx.row()

        selected_rows = sorted(self._selected_rows())
        if not selected_rows:
            event.ignore()
            return

        order_before = self._order_before_drag or self._current_order()
        sel_model = self.selectionModel()
        sel_was_blocked = sel_model.blockSignals(True) if sel_model is not None else False
        try:
            new_rows = self._model.reorder_rows(selected_rows, target_row)
            if new_rows:
                self._select_rows(new_rows)
        finally:
            if sel_model is not None:
                sel_model.blockSignals(sel_was_blocked)

        event.setDropAction(Qt.MoveAction)
        event.accept()

        order_after = self._current_order()
        if order_after != order_before:
            self.order_changed.emit(order_after)

    def startDrag(self, supportedActions):
        self._order_before_drag = self._current_order()
        super().startDrag(supportedActions)

    # ───── Helpers ───────────────────────────────────────────────
    def _current_order(self) -> list[int]:
        """Return model indices in current visual order."""
        if self._model.has_placeholder:
            return []
        if not self._model._custom_roles:
            return list(self._model._model_indices)
        return [self._model.model_index(row) for row in range(self._model.rowCount())]

    def selected_model_indices(self) -> list[int]:
        """Return list of selected spectra indices in the Model."""
        if self._model.has_placeholder:
            return []
        model = self._model
        if not model._custom_roles:
            sel_model = self.selectionModel()
            if sel_model is None:
                return []
            selection = sel_model.selection()
            if not selection:
                return []
            model_indices = model._model_indices
            n = len(model_indices)
            if len(selection) == 1:
                rng = selection[0]
                top = rng.top()
                bottom = rng.bottom()
                if 0 <= top and bottom < n:
                    return model_indices[top : bottom + 1]
            rows = self._selected_rows()
            return [model_indices[r] if 0 <= r < n else r for r in rows]
        return [model.model_index(row) for row in self._selected_rows()]

    def _emit_selection_changed(self):
        """Emit selection_changed signal with model indices of selected items."""
        self.selection_changed.emit(self.selected_model_indices())
