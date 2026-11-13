# spectroview/view/components/v_map_list.py
"""View component for Maps list and spectrum action controls."""
from __future__ import annotations
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QAbstractItemView, QPushButton,
)
from PySide6.QtCore import Qt, Signal, QItemSelectionModel
from PySide6.QtGui import QFont

from spectroview.viewmodel.utils import MetadataTooltipManager
from spectroview.view.theme import Icons, current_tokens


class VMapsList(QWidget):
    """Maps list widget with multi-selection and spectrum/map action controls.

    Supports ``ExtendedSelection`` (Click, Ctrl+Click, Shift+Click) and emits
    both the legacy ``map_selection_changed(int)`` (focused/primary row) and
    ``maps_selection_changed(selected_names, primary_name)``.
    """

    # ═══ View → ViewModel signals ═══
    map_selection_changed = Signal(int)               # Focused/primary map index
    maps_selection_changed = Signal(list, object)     # (selected_names, primary_name)
    view_map_requested = Signal()                     # View map DataFrame
    delete_map_requested = Signal()                   # Delete selected map(s)
    save_requested = Signal()                         # Save map to Excel / CSV
    select_all_requested = Signal()                   # Select all spectra
    reinitialize_requested = Signal()                 # Reinitialize selected spectra
    send_to_spectra_requested = Signal()              # Send to Spectra workspace

    def __init__(self, parent=None):
        super().__init__(parent)
        self._has_maps_placeholder = False
        self._items_by_name: dict[str, QListWidgetItem] = {}
        self._last_emitted_selection: Optional[tuple[tuple[str, ...], Optional[str]]] = None

        # Metadata tooltip support
        self._map_metadata_provider = None

        self._init_ui()
        self._maps_tooltip_manager = MetadataTooltipManager(self.maps_list, self._get_map_metadata)
        self._update_maps_placeholder()  # Show placeholder initially

    def set_map_metadata_provider(self, provider):
        """Set metadata provider for the maps list."""
        self._map_metadata_provider = provider

    def _get_map_metadata(self, item):
        """Retrieve metadata for a map item."""
        if self._map_metadata_provider:
            try:
                meta = self._map_metadata_provider(item)
            except TypeError:
                meta = self._map_metadata_provider(item.text())
            if meta:
                return meta
        return item.data(Qt.UserRole + 2) or {}
        
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        maps_label = QLabel("Maps list:")
        main_layout.addWidget(maps_label)
        
        # ══════════════════════════════════════════════════════════════
        # MAPS LIST (EXPANDED) WITH VERTICAL ACTION BUTTON STRIP
        # ══════════════════════════════════════════════════════════════
        maps_container = QHBoxLayout()
        maps_container.setSpacing(4)
        
        # Left: Maps list (occupies remaining height; Ctrl/Shift multi-select supported)
        self.maps_list = QListWidget()
        self.maps_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.maps_list.setMinimumHeight(80)
        self.maps_list.itemSelectionChanged.connect(self._on_map_selected)
        maps_container.addWidget(self.maps_list, stretch=1)
        
        # Right: Action buttons vertical strip (Spectra actions on top, Maps actions below)
        buttons_layout = QVBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(4)
        
        # --- Spectra action buttons (operating on heatmap-selected spectra) ---
        self.btn_select_all = QPushButton()
        self.btn_select_all.setIcon(Icons.check_circle())
        self.btn_select_all.setToolTip("Select all spectra in current map")
        self.btn_select_all.setFixedSize(32, 32)
        self.btn_select_all.clicked.connect(self.select_all_requested.emit)
        
        self.btn_reinit = QPushButton()
        self.btn_reinit.setIcon(Icons.undo())
        self.btn_reinit.setToolTip("Reinitialize selected spectra (Ctrl for all maps)")
        self.btn_reinit.setFixedSize(32, 32)
        self.btn_reinit.clicked.connect(self.reinitialize_requested.emit)
        
        self.btn_send_to_spectra = QPushButton()
        self.btn_send_to_spectra.setIcon(Icons.send())
        self.btn_send_to_spectra.setToolTip("Send selected spectra to 'Spectra' Workspace")
        self.btn_send_to_spectra.setFixedSize(32, 32)
        self.btn_send_to_spectra.clicked.connect(self.send_to_spectra_requested.emit)
        
        buttons_layout.addWidget(self.btn_select_all)
        buttons_layout.addWidget(self.btn_reinit)
        buttons_layout.addWidget(self.btn_send_to_spectra)
        
        # Visual spacing between spectra actions and map actions
        buttons_layout.addSpacing(6)
        
        # --- Map action buttons ---
        self.btn_view = QPushButton()
        self.btn_view.setIcon(Icons.search())
        self.btn_view.setToolTip("View selected map")
        self.btn_view.setFixedSize(32, 32)
        self.btn_view.clicked.connect(self.view_map_requested.emit)
        
        self.btn_delete = QPushButton()
        self.btn_delete.setIcon(Icons.trash())
        self.btn_delete.setToolTip("Delete selected map")
        self.btn_delete.setFixedSize(32, 32)
        self.btn_delete.clicked.connect(self.delete_map_requested.emit)
        
        self.btn_save = QPushButton()
        self.btn_save.setIcon(Icons.save())
        self.btn_save.setToolTip("Save the selected map")
        self.btn_save.setFixedSize(32, 32)
        self.btn_save.clicked.connect(self.save_requested.emit)
        
        buttons_layout.addWidget(self.btn_view)
        buttons_layout.addWidget(self.btn_delete)
        buttons_layout.addWidget(self.btn_save)
        buttons_layout.addStretch()
        
        maps_container.addLayout(buttons_layout)
        main_layout.addLayout(maps_container, stretch=1)
    
    def _update_maps_placeholder(self):
        """Update placeholder text for maps list based on state."""
        if self.maps_list.count() == 0:
            for _ in range(4):
                spacer = QListWidgetItem("")
                spacer.setFlags(Qt.NoItemFlags)
                self.maps_list.addItem(spacer)
            
            placeholder = QListWidgetItem("Drag and drop file(s) anywhere to open")
            placeholder.setFlags(Qt.NoItemFlags)
            placeholder.setForeground(Qt.gray)
            placeholder.setTextAlignment(Qt.AlignCenter)
            
            font = QFont()
            font.setPointSize(12)
            placeholder.setFont(font)
            
            self.maps_list.addItem(placeholder)
            self._has_maps_placeholder = True

    def _selectable_selected_items(self) -> list[QListWidgetItem]:
        """Return currently selected non-placeholder items in visual list order."""
        items: list[QListWidgetItem] = []
        for row in range(self.maps_list.count()):
            item = self.maps_list.item(row)
            if item is not None and bool(item.flags() & Qt.ItemIsSelectable) and item.isSelected():
                items.append(item)
        return items

    def _resolve_selection(self) -> tuple[list[str], Optional[str], int]:
        """Resolve ordered selected names, Primary Map name, and Primary row index.

        When Ctrl+Click deselects the currently focused item while other maps
        remain selected, promotes the last remaining selected item as Primary
        so ``primary_name`` is always an actually selected map when non-empty.
        """
        selected_items = self._selectable_selected_items()
        if not selected_items:
            return [], None, -1

        selected_names = [it.text() for it in selected_items]
        curr = self.maps_list.currentItem()
        if curr is not None and curr in selected_items:
            primary_item = curr
        else:
            primary_item = selected_items[-1]
            was_blocked = self.maps_list.blockSignals(True)
            try:
                self.maps_list.setCurrentItem(primary_item, QItemSelectionModel.NoUpdate)
            finally:
                self.maps_list.blockSignals(was_blocked)

        return selected_names, primary_item.text(), self.maps_list.row(primary_item)

    # ───── Public API (ViewModel → View) ─────────────────────────
    def set_maps_names(self, names: list[str]):
        """Replace entire maps list while preserving multi-selection by map name."""
        prev_selected = self.get_selected_map_names()
        prev_primary = self.get_primary_map_name()

        was_blocked = self.maps_list.blockSignals(True)
        try:
            self.maps_list.clear()
            self._has_maps_placeholder = False
            self._items_by_name.clear()

            for i, name in enumerate(names):
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, i)
                self._items_by_name[name] = item
                self.maps_list.addItem(item)

            if len(names) == 0:
                self._update_maps_placeholder()
            else:
                surviving = [n for n in prev_selected if n in self._items_by_name]
                if surviving:
                    for n in surviving:
                        self._items_by_name[n].setSelected(True)
                    target_primary = prev_primary if prev_primary in self._items_by_name else surviving[-1]
                    self.maps_list.setCurrentItem(
                        self._items_by_name[target_primary], QItemSelectionModel.NoUpdate
                    )
                else:
                    first_item = self._items_by_name[names[0]]
                    first_item.setSelected(True)
                    self.maps_list.setCurrentItem(first_item, QItemSelectionModel.NoUpdate)
        finally:
            self.maps_list.blockSignals(was_blocked)

        selected_names, primary_name, _ = self._resolve_selection()
        self._last_emitted_selection = (tuple(selected_names), primary_name)

    def get_selected_map_index(self) -> int:
        """Return currently focused/primary selected map index, or -1 if none."""
        _, _, row = self._resolve_selection()
        return row

    def get_selected_map_names(self) -> list[str]:
        """Return ordered list of all currently selected map names."""
        return [it.text() for it in self._selectable_selected_items()]

    def get_primary_map_name(self) -> Optional[str]:
        """Return the Primary (focused) selected map name, or None."""
        _, primary_name, _ = self._resolve_selection()
        return primary_name

    def set_selected_map_name(self, name: str):
        """Select a single map by name cleanly without emitting duplicate signals."""
        self.set_selected_maps([name] if name else [], primary_name=name or None)

    def set_selected_maps(self, names: list[str], primary_name: Optional[str] = None):
        """Programmatically highlight *names* and set *primary_name* without emitting."""
        valid = [n for n in names if n in self._items_by_name]
        if primary_name not in self._items_by_name or (valid and primary_name not in valid):
            resolved_primary = valid[-1] if valid else None
        else:
            resolved_primary = primary_name

        # Skip redundant programmatic resets if selection & primary already match
        curr_sel = self.get_selected_map_names()
        curr_item = self.maps_list.currentItem()
        curr_prim = curr_item.text() if (curr_item and bool(curr_item.flags() & Qt.ItemIsSelectable)) else None
        if curr_sel == valid and curr_prim == resolved_primary:
            return

        was_blocked = self.maps_list.blockSignals(True)
        try:
            self.maps_list.clearSelection()
            for n in valid:
                self._items_by_name[n].setSelected(True)
            if resolved_primary and resolved_primary in self._items_by_name:
                self.maps_list.setCurrentItem(
                    self._items_by_name[resolved_primary], QItemSelectionModel.NoUpdate
                )
        finally:
            self.maps_list.blockSignals(was_blocked)

        self._last_emitted_selection = (tuple(valid), resolved_primary)

    # ───── Internal signal handlers ──────────────────────────────
    def _on_map_selected(self):
        """Emit deduplicated legacy index and multi-selection signals."""
        selected_names, primary_name, primary_row = self._resolve_selection()
        sig_key = (tuple(selected_names), primary_name)
        if sig_key == self._last_emitted_selection:
            return
        self._last_emitted_selection = sig_key
        if primary_row >= 0:
            self.map_selection_changed.emit(primary_row)
        self.maps_selection_changed.emit(selected_names, primary_name)

    def apply_theme(self, theme: str):
        """Update icons based on the global application theme."""
        tok = current_tokens()
        icon_color = tok.text_primary

        # Map action buttons
        self.btn_view.setIcon(Icons.search(color=icon_color))
        self.btn_delete.setIcon(Icons.trash(color=icon_color))
        self.btn_save.setIcon(Icons.save(color=icon_color))

        # Spectra action buttons
        self.btn_select_all.setIcon(Icons.check_circle(color=icon_color))
        self.btn_reinit.setIcon(Icons.undo(color=icon_color))
        self.btn_send_to_spectra.setIcon(Icons.send(color=icon_color))
