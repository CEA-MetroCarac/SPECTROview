# spectroview/view/components/v_map_list.py
"""View component for Maps list and spectrum action controls."""
import os 
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QAbstractItemView, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from spectroview import ICON_DIR
from spectroview.viewmodel.utils import MetadataTooltipManager, get_tinted_icon


class VMapsList(QWidget):
    """Maps list widget with map actions and spectrum selection action buttons.
    
    Provides navigation across loaded maps (view, delete, save) and actions
    operating on spectra selected via the 2D Map Viewer (select all, reinit,
    stats, send to Spectra workspace).
    """
    
    # ═══ View → ViewModel signals ═══
    map_selection_changed = Signal(int)           # Selected map index
    view_map_requested = Signal()                 # View map DataFrame
    delete_map_requested = Signal()               # Delete selected map
    save_requested = Signal()                     # Save map to Excel / CSV
    select_all_requested = Signal()               # Select all spectra
    reinitialize_requested = Signal()             # Reinitialize selected spectra
    stats_requested = Signal()                    # Show fitting statistics
    send_to_spectra_requested = Signal()          # Send to Spectra workspace
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._has_maps_placeholder = False  # Track placeholder state

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
        main_layout.setSpacing(6)
        
        # ══════════════════════════════════════════════════════════════
        # MAPS LIST WITH ACTION BUTTONS
        # ══════════════════════════════════════════════════════════════
        maps_container = QHBoxLayout()
        maps_container.setSpacing(4)
        
        # Left: Maps list
        maps_list_layout = QVBoxLayout()
        maps_list_layout.setContentsMargins(0, 0, 0, 0)
        maps_list_layout.setSpacing(2)
    
        maps_label = QLabel("Maps list:")
        maps_list_layout.addWidget(maps_label)
        
        self.maps_list = QListWidget()
        self.maps_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.maps_list.setMaximumHeight(160)
        self.maps_list.setMinimumHeight(80)
        self.maps_list.itemSelectionChanged.connect(self._on_map_selected)
        
        maps_list_layout.addWidget(self.maps_list)
        maps_container.addLayout(maps_list_layout, stretch=1)
        
        # Right: Map action buttons
        map_buttons_layout = QVBoxLayout()
        map_buttons_layout.setSpacing(4)
        
        self.btn_view = QPushButton()
        self.btn_view.setToolTip("View selected map")
        self.btn_view.setFixedSize(32, 32)
        self.btn_view.clicked.connect(self.view_map_requested.emit)
        
        self.btn_delete = QPushButton()
        self.btn_delete.setToolTip("Delete selected map")
        self.btn_delete.setFixedSize(32, 32)
        self.btn_delete.clicked.connect(self.delete_map_requested.emit)
        
        self.btn_save = QPushButton()
        self.btn_save.setToolTip("Save the selected map")
        self.btn_save.setFixedSize(32, 32)
        self.btn_save.clicked.connect(self.save_requested.emit)
        
        map_buttons_layout.addStretch()
        map_buttons_layout.addWidget(self.btn_view)
        map_buttons_layout.addWidget(self.btn_delete)
        map_buttons_layout.addWidget(self.btn_save)
        map_buttons_layout.addStretch()
        
        maps_container.addLayout(map_buttons_layout)
        main_layout.addLayout(maps_container)
        
        # ══════════════════════════════════════════════════════════════
        # SPECTRA ACTIONS (operating on heatmap-selected spectra)
        # ══════════════════════════════════════════════════════════════
        spectra_actions_container = QVBoxLayout()
        spectra_actions_container.setSpacing(3)
        
        spectra_actions_label = QLabel("Spectra actions:")
        spectra_actions_container.addWidget(spectra_actions_label)
        
        spectra_buttons_layout = QHBoxLayout()
        spectra_buttons_layout.setSpacing(6)
        
        self.btn_select_all = QPushButton()
        self.btn_select_all.setToolTip("Select all spectra in current map")
        self.btn_select_all.setFixedSize(32, 32)
        self.btn_select_all.clicked.connect(self.select_all_requested.emit)
        
        self.btn_reinit = QPushButton()
        self.btn_reinit.setToolTip("Reinitialize selected spectra (Ctrl for all maps)")
        self.btn_reinit.setFixedSize(32, 32)
        self.btn_reinit.clicked.connect(self.reinitialize_requested.emit)
        
        self.btn_stats = QPushButton()
        self.btn_stats.setToolTip("Show fitting statistics of selected spectrum")
        self.btn_stats.setFixedSize(32, 32)
        self.btn_stats.clicked.connect(self.stats_requested.emit)
        
        self.btn_send_to_spectra = QPushButton()
        self.btn_send_to_spectra.setToolTip("Send selected spectra to 'Spectra' Workspace")
        self.btn_send_to_spectra.setFixedSize(32, 32)
        self.btn_send_to_spectra.clicked.connect(self.send_to_spectra_requested.emit)
        
        spectra_buttons_layout.addWidget(self.btn_select_all)
        spectra_buttons_layout.addWidget(self.btn_reinit)
        spectra_buttons_layout.addWidget(self.btn_stats)
        spectra_buttons_layout.addWidget(self.btn_send_to_spectra)
        spectra_buttons_layout.addStretch()
        
        spectra_actions_container.addLayout(spectra_buttons_layout)
        main_layout.addLayout(spectra_actions_container)
    
    def _update_maps_placeholder(self):
        """Update placeholder text for maps list based on state."""
        if self.maps_list.count() == 0:
            for _ in range(2):
                spacer = QListWidgetItem("")
                spacer.setFlags(Qt.NoItemFlags)
                self.maps_list.addItem(spacer)
            
            placeholder = QListWidgetItem("📂 Drag and drop file(s) anywhere to open")
            placeholder.setFlags(Qt.NoItemFlags)
            placeholder.setForeground(Qt.gray)
            placeholder.setTextAlignment(Qt.AlignCenter)
            
            font = QFont()
            font.setPointSize(12)
            placeholder.setFont(font)
            
            self.maps_list.addItem(placeholder)
            self._has_maps_placeholder = True

    # ───── Public API (ViewModel → View) ─────────────────────────
    def set_maps_names(self, names: list[str]):
        """Replace entire maps list (ViewModel-driven)."""
        current_selection = self.get_selected_map_index()
        
        self.maps_list.clear()
        self._has_maps_placeholder = False  # Reset placeholder flag
        
        for i, name in enumerate(names):
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, i)  # Store index
            self.maps_list.addItem(item)
        
        # Auto-select first map if no selection and maps exist
        if current_selection < 0 and len(names) > 0:
            self.maps_list.setCurrentRow(0)
        elif 0 <= current_selection < len(names):
            self.maps_list.setCurrentRow(current_selection)
        
        # Update placeholder only if list is empty after adding items
        if len(names) == 0:
            self._update_maps_placeholder()
    
    def get_selected_map_index(self) -> int:
        """Return currently selected map index, or -1 if none."""
        selected = self.maps_list.selectedItems()
        if selected:
            return self.maps_list.row(selected[0])
        return -1

    # ───── Internal signal handlers ──────────────────────────────
    def _on_map_selected(self):
        """Emit signal when map selection changes."""
        idx = self.get_selected_map_index()
        if idx >= 0:
            self.map_selection_changed.emit(idx)

    def apply_theme(self, theme: str):
        """Update icons based on the global application theme."""
        icon_color = "#404040" if theme != "dark" else "#F0F0F0"

        # Map action buttons
        self.btn_view.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "view.png"), icon_color))
        self.btn_delete.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "trash.png"), icon_color))
        self.btn_save.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "save.png"), icon_color))

        # Spectra action buttons
        self.btn_select_all.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "select-all.png"), icon_color))
        self.btn_reinit.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "undo2.png"), icon_color))
        self.btn_stats.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "stats.png"), icon_color))
        self.btn_send_to_spectra.setIcon(get_tinted_icon(os.path.join(ICON_DIR, "send.png"), icon_color))
