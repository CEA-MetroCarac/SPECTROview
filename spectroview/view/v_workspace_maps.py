"""View for Maps Workspace - extends Spectra Workspace with map-specific features."""
import os
import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QComboBox, QFrame,
    QGroupBox, QLineEdit, QDoubleSpinBox, QSpacerItem, QSizePolicy,
    QScrollArea, QDialog
)

from spectroview import ICON_DIR
from spectroview.model.m_settings import MSettings
from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.view.components.v_map_list import VMapsList
from spectroview.view.components.v_map_viewer import VMapViewer
from spectroview.view.components.v_map_viewer_dialog import VMapViewerDialog
from spectroview.view.components.v_dataframe_table import VDataframeTable
from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
from spectroview.viewmodel.utils import show_toast_notification


class VWorkspaceMaps(VWorkspaceSpectra):
    """Maps Workspace View - inherits from Spectra Workspace and adds map visualization."""
    
    # ───── Additional signals for Maps ─────
    map_type_changed = Signal(str)
    send_to_spectra_requested = Signal()
    
    def __init__(self, parent=None):
        # Temporarily store original setup_connections to prevent parent from calling it
        self._skip_parent_setup = True
        
        # Call parent constructor - init UI but skip connections
        super().__init__(parent)
        
        # Replace parent's ViewModel with Maps-specific ViewModel
        self.vm = VMWorkspaceMaps(self.m_settings)
        
        # Track additional viewer dialogs
        self.viewer_dialogs = []  # List of VMapViewerDialog instances
        self.next_viewer_number = 2  # Counter for dialog titles
        
        # Centralized map type - all viewers reference this
        self.selected_map_type = '2Dmap'
        
        # Shared data references to avoid duplication
        self._current_map_df = None
        self._current_fit_results = None
        
        # Re-inject the fit model builder dependency (required for apply_loaded_fit_model)
        self.vm.set_fit_model_builder(self.vm_fit_model_builder)
        
        # Now set up ALL connections with the correct VM
        self._skip_parent_setup = False
        self.setup_connections()
        
        self._add_maps_panel()
        self._connect_signals()
    
    @staticmethod
    def _extract_coords_from_fname(fname: str) -> tuple[float, float] | None:
        """Extract (x, y) coordinates from spectrum fname."""
        if '(' in fname and ')' in fname:
            coords_str = fname[fname.rfind('(')+1:fname.rfind(')')]
            try:
                x_str, y_str = coords_str.split(',')
                return (float(x_str.strip()), float(y_str.strip()))
            except (ValueError, AttributeError):
                pass
        return None
    
    def setup_connections(self):
        """Override to prevent double connection during parent init, then connect properly."""
        if hasattr(self, '_skip_parent_setup') and self._skip_parent_setup:
            return  # Skip during parent's __init__
        
        # Call parent's setup_connections which will connect to self.vm (now VMWorkspaceMaps)
        super().setup_connections()
    
    def _add_maps_panel(self):
        """Replace the simple spectra list sidebar with Maps-specific controls."""
        main_layout = self.layout()
        main_splitter = main_layout.itemAt(0).widget()  # QSplitter
        
        # Remove the old right widget and create new one
        old_right = main_splitter.widget(1)
        if old_right:
            old_right.setParent(None)
            old_right.deleteLater()
        
        # Create new right panel with Maps features
        right_widget = self._create_maps_right_panel()
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([900, 400])
    
    def _create_maps_right_panel(self):
        """Create the right panel with map viewer and controls in a scroll area."""
        # Main panel wrapper
        panel = QFrame()
        panel.setMaximumWidth(450)
        panel.setFrameShape(QFrame.StyledPanel)
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for all controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Scrollable content widget
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        
        # ══════════════════════════════════════════════════════════════
        # MAP VIEWER WIDGET (with matplotlib canvas and controls)
        # ══════════════════════════════════════════════════════════════
        self.v_map_viewer = VMapViewer()
        layout.addWidget(self.v_map_viewer)
        
        # ══════════════════════════════════════════════════════════════
        # MAPS AND SPECTRA LIST (VMapsList widget with action buttons)
        # ══════════════════════════════════════════════════════════════
        self.v_maps_list = VMapsList()
        layout.addWidget(self.v_maps_list, stretch=1)
        
        # Set scroll content and add to main panel
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # ── Footer: count + progress bar + stop button (outside scroll area) ──
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(4)
        
        self.lbl_count.setText("0 spectra loaded")
        self.progress_bar.setFixedHeight(15)
        
        # Stop button is inherited from parent but needs to be added to layout
        self.btn_stop_fit.setFixedHeight(15)
        self.btn_stop_fit.setFixedWidth(50)
        
        footer_layout.addWidget(self.lbl_count)
        footer_layout.addWidget(self.progress_bar)
        footer_layout.addWidget(self.btn_stop_fit)
        main_layout.addLayout(footer_layout)
        
        return panel
    
    def _on_spectra_list_changed(self, spectra: list):
        """Handle spectra list update from ViewModel."""
        self.v_maps_list.set_spectra_names(spectra)
    
    def _on_checkbox_changed(self, item):
        """Update spectrum.is_active when checkbox state changes."""
        if item is None or not self.vm.current_map_name:
            return
        
        idx = self.v_maps_list.spectra_list.row(item)
        
        # Get current map's spectra
        fname_prefix = f"{self.vm.current_map_name}_("
        map_spectra = [s for s in self.vm.spectra if s.fname.startswith(fname_prefix)]
        
        if 0 <= idx < len(map_spectra):
            is_checked = item.checkState() == Qt.Checked
            map_spectra[idx].is_active = is_checked
            
            # Invalidate fit results cache and trigger map data update
            self.vm._fit_results_cache_dirty = True
            # Get updated DataFrame and emit
            updated_df = self.vm.get_current_map_dataframe()
            self.vm.map_data_updated.emit(updated_df)
    
    def _on_check_all_toggled(self, checked: bool):
        """Handle check all checkbox toggle for current map."""
        if not self.vm.current_map_name:
            return
        
        # Block signals temporarily to avoid triggering individual checkbox handlers
        self.v_maps_list.spectra_list.blockSignals(True)
        
        # Get current map's spectra
        fname_prefix = f"{self.vm.current_map_name}_("
        map_spectra = [s for s in self.vm.spectra if s.fname.startswith(fname_prefix)]
        
        # Update all checkboxes in the list
        for i in range(self.v_maps_list.spectra_list.count()):
            item = self.v_maps_list.spectra_list.item(i)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        
        # Update all current map spectra is_active state
        for spectrum in map_spectra:
            spectrum.is_active = checked
        
        self.v_maps_list.spectra_list.blockSignals(False)
        
        # Invalidate cache and trigger map data update
        self.vm._fit_results_cache_dirty = True
        updated_df = self.vm.get_current_map_dataframe()
        self.vm.map_data_updated.emit(updated_df)
    
    def _connect_signals(self):
        """Connect Maps-specific signals between View and ViewModel."""
        # ── VMapsList → ViewModel connections ──
        self.v_maps_list.files_dropped.connect(self.vm.load_map_files)
        self.v_maps_list.map_selection_changed.connect(self._on_map_selected)
        self.v_maps_list.spectra_selection_changed.connect(self._on_spectra_list_selection)
        
        # ── VMapsList button connections ──
        self.v_maps_list.view_map_requested.connect(self._on_view_map_requested)
        self.v_maps_list.delete_map_requested.connect(self.vm.delete_current_map)
        self.v_maps_list.save_requested.connect(self._on_save_map_requested)
        self.v_maps_list.select_all_requested.connect(self._on_select_all_spectra)
        self.v_maps_list.reinitialize_requested.connect(self._on_reinit_spectra)
        self.v_maps_list.send_to_spectra_requested.connect(self._on_send_to_spectra)
        
        # ── VMapViewer → ViewModel connections ──
        self.v_map_viewer.spectra_selected.connect(self._on_map_viewer_selection)
        self.v_map_viewer.multi_viewer_requested.connect(self._on_add_viewer_requested)
        self.v_map_viewer.cbb_map_type.currentTextChanged.connect(self._on_map_type_changed)
        self.v_map_viewer.extract_profile_requested.connect(self._on_extract_profile_requested)
        
        # ── ViewModel → VMapsList connections ──
        self.vm.maps_list_changed.connect(self.v_maps_list.set_maps_names)
        
        # Check all checkbox (now in VMapsList)
        self.v_maps_list.check_all_toggled.connect(self._on_check_all_toggled)
        
        # The spectra list is updated via inherited VMWorkspaceSpectra signals
        # when vm.select_map() calls _extract_spectra_from_map()
        self.vm.spectra_list_changed.connect(self._on_spectra_list_changed)
        self.v_maps_list.spectra_list.itemChanged.connect(self._on_checkbox_changed)
        
        # Clear griddata cache when map data changes (after fitting)
        self.vm.clear_map_cache_requested.connect(self.v_map_viewer.clear_cache_for_map)
        
        # ── ViewModel → VMapViewer connections ──
        # Note: map_selected signal removed to avoid duplicate calls to _on_map_data_changed
        self.vm.map_data_updated.connect(self._on_map_data_changed)
        
        # Connect spectra selection to viewer (inherited from parent)
        self.vm.spectra_selection_changed.connect(self.v_spectra_viewer.set_plot_data)
        
        # Connect send to spectra workspace signal to parent's spectra workspace
        self.vm.send_spectra_to_workspace.connect(self._receive_spectra_from_maps)
    
    def _on_map_viewer_selection(self, selected_points: list):
        """Handle spectrum selection from map viewer."""
        if not selected_points:
            self.v_maps_list.spectra_list.clearSelection()
            return
        
        # Build fnames from coordinates
        selected_fnames = [
            f"{self.vm.current_map_name}_({x}, {y})"
            for x, y in selected_points
        ]
        
        # Get list of current map's spectra
        fname_prefix = f"{self.vm.current_map_name}_("
        current_map_fnames = [
            s.fname for s in self.vm.spectra 
            if s.fname.startswith(fname_prefix)
        ]
        
        # Find list indices
        list_indices = [
            i for i, fname in enumerate(current_map_fnames)
            if fname in selected_fnames
        ]
        
        if not list_indices:
            return
        
        self.v_maps_list.spectra_list.blockSignals(True)
        self.v_maps_list.spectra_list.clearSelection()
        
        # Set selection in list widget
        for list_idx in list_indices:
            if 0 <= list_idx < self.v_maps_list.spectra_list.count():
                self.v_maps_list.spectra_list.item(list_idx).setSelected(True)
        
        # Auto-scroll
        if list_indices:
            self.v_maps_list.spectra_list.scrollToItem(
                self.v_maps_list.spectra_list.item(list_indices[0])
            )
        
        self.v_maps_list.spectra_list.blockSignals(False)
        self.vm.set_selected_fnames(selected_fnames)
        
        # Sync selected points to all dialog viewers
        for dialog in self.viewer_dialogs:
            dialog.set_selected_points(selected_points)
    
    def _on_add_viewer_requested(self, _count: int):
        """Handle request to open a new floating map viewer window."""
        # Create a new dialog viewer
        dialog = VMapViewerDialog(self, viewer_number=self.next_viewer_number)
        self.next_viewer_number += 1
        
        # Track the dialog
        self.viewer_dialogs.append(dialog)
        
        # Connect signals from dialog viewer
        dialog.spectra_selected.connect(self._on_dialog_viewer_selection)
        dialog.extract_profile_requested.connect(lambda name, d=dialog: self._on_extract_profile_from_dialog(name, d))
        
        # Connect map type change to centralized handler
        dialog.map_viewer.cbb_map_type.currentTextChanged.connect(self._on_map_type_changed)
        
        # Set current map data in dialog (shared references - no duplication)
        if self._current_map_df is not None and not self._current_map_df.empty and self.vm.current_map_name:
            dialog.set_map_data(self._current_map_df, self.vm.current_map_name, self._current_fit_results)
        
        # Sync map type with centralized value
        dialog.map_viewer.cbb_map_type.blockSignals(True)
        dialog.map_viewer.cbb_map_type.setCurrentText(self.selected_map_type)
        dialog.map_viewer.cbb_map_type.blockSignals(False)
        
        # Clean up when dialog is closed
        dialog.finished.connect(lambda: self._on_dialog_closed(dialog))
        
        # Show the dialog (non-modal - user can interact with both windows)
        dialog.show()
    
    def _on_dialog_viewer_selection(self, selected_points: list):
        """Handle selection from a dialog viewer - sync with main viewer and list."""
        # Same logic as main viewer selection (updates list and ViewModel)
        self._on_map_viewer_selection(selected_points)
        
        # Update main viewer with selected points
        self.v_map_viewer.set_selected_points(selected_points)
        
        # Update all dialog viewers
        for dialog in self.viewer_dialogs:
            dialog.set_selected_points(selected_points)
    
    def _on_dialog_closed(self, dialog):
        """Clean up when a dialog viewer is closed."""
        if dialog in self.viewer_dialogs:
            self.viewer_dialogs.remove(dialog)
    
    def _on_map_type_changed(self, map_type: str):
        """Handle map type change - update centralized value and sync all comboboxes."""
        # Update centralized value
        self.selected_map_type = map_type
        
        # Sync main viewer combobox and trigger replot
        self.v_map_viewer.cbb_map_type.blockSignals(True)
        self.v_map_viewer.cbb_map_type.setCurrentText(map_type)
        self.v_map_viewer.cbb_map_type.blockSignals(False)
        self.v_map_viewer.plot_heatmap()  # Manually trigger plot update
        
        # Sync all dialog viewers comboboxes and trigger replots
        for dialog in self.viewer_dialogs:
            dialog.map_viewer.cbb_map_type.blockSignals(True)
            dialog.map_viewer.cbb_map_type.setCurrentText(map_type)
            dialog.map_viewer.cbb_map_type.blockSignals(False)
            dialog.map_viewer.plot_heatmap()  # Manually trigger plot update
    
    def _on_spectra_list_selection(self, list_indices: list):
        """Handle spectra selection from list."""
        # Update heatmap highlights
        self._update_heatmap_selection()
        
        if not list_indices:
            self.vm.set_selected_fnames([])
            return
        
        # Get list of current map's spectra
        fname_prefix = f"{self.vm.current_map_name}_("
        current_map_fnames = [
            s.fname for s in self.vm.spectra 
            if s.fname.startswith(fname_prefix)
        ]
        
        # Convert list indices to fnames
        selected_fnames = [
            current_map_fnames[i]
            for i in list_indices
            if 0 <= i < len(current_map_fnames)
        ]
        
        if not selected_fnames:
            return
        
        # Auto-scroll to first selected item
        if 0 <= list_indices[0] < self.v_maps_list.spectra_list.count():
            self.v_maps_list.spectra_list.scrollToItem(
                self.v_maps_list.spectra_list.item(list_indices[0])
            )
        
        self.vm.set_selected_fnames(selected_fnames)
    
    def _on_map_data_changed(self):
        """Update map viewer when map data changes."""
        # Get current map data and fit results ONCE (shared references)
        self._current_map_df = self.vm.get_current_map_dataframe()
        self._current_fit_results = self.vm.get_fit_results_dataframe()
        
        # Update main map viewer
        old_state = self.v_map_viewer.blockSignals(True)
        try:
            if self._current_map_df is not None and not self._current_map_df.empty and self.vm.current_map_name:
                self.v_map_viewer.set_map_data(self._current_map_df, self.vm.current_map_name, self._current_fit_results)
            else:
                # Clear map plot when no map data (all maps deleted or map removed)
                self.v_map_viewer.set_map_data(None, None, None)
        finally:
            # Restore signal state
            self.v_map_viewer.blockSignals(old_state)
        
        # Update all dialog viewers (they share the same dataframe references - no duplication)
        for dialog in self.viewer_dialogs:
            dialog_old_state = dialog.blockSignals(True)
            try:
                if self._current_map_df is not None and not self._current_map_df.empty and self.vm.current_map_name:
                    dialog.set_map_data(self._current_map_df, self.vm.current_map_name, self._current_fit_results)
                else:
                    dialog.set_map_data(None, None, None)
            finally:
                dialog.blockSignals(dialog_old_state)
        
        # After map data is loaded, sync heatmap highlights with current selection
        self._sync_heatmap_with_selection()
    
    def _sync_heatmap_with_selection(self):
        """Synchronize heatmap highlights with current spectra list selection."""
        self._update_heatmap_selection()
    
    def _update_heatmap_selection(self):
        """Update heatmap highlights based on current list selection (common logic)."""
        if not self.vm.current_map_name:
            return
        
        # Get currently selected indices from the list widget
        selected_indices = self.v_maps_list.get_selected_spectra_indices()
        
        if not selected_indices:
            self.v_map_viewer.set_selected_points([])
            return
        
        # Get list of current map's spectra
        fname_prefix = f"{self.vm.current_map_name}_("
        current_map_fnames = [
            s.fname for s in self.vm.spectra 
            if s.fname.startswith(fname_prefix)
        ]
        
        # Convert indices to coordinates
        selected_points = []
        for idx in selected_indices:
            if 0 <= idx < len(current_map_fnames):
                coords = self._extract_coords_from_fname(current_map_fnames[idx])
                if coords:
                    selected_points.append(coords)
        
        # Update heatmap highlights in main viewer
        if selected_points:
            self.v_map_viewer.set_selected_points(selected_points)
        
        # Update all dialog viewers
        for dialog in self.viewer_dialogs:
            dialog.set_selected_points(selected_points if selected_points else [])
    
    def _on_map_selected(self, index: int):
        """Handle map selection - convert index to map name and call ViewModel."""
        if index >= 0 and index < len(self.vm.maps):
            map_names = list(self.vm.maps.keys())
            map_name = map_names[index]
            self.vm.select_map(map_name)
    
    def _on_view_map_requested(self):
        """Display the current map's DataFrame in a table dialog."""
        df = self.vm.get_current_map_dataframe()
        if df is None:
            return
        
        # Limit to first 50 rows and 50 columns to avoid freezing with large datasets
        max_rows = 50
        max_cols = 50
        df_limited = df.iloc[:max_rows, :max_cols]
        
        # Create dialog to show the DataFrame
        dialog = QDialog(self)
        title = f"Map Data: {self.vm.current_map_name}"
        if len(df) > max_rows or len(df.columns) > max_cols:
            title += f" (showing {len(df_limited)} of {len(df)} rows, {len(df_limited.columns)} of {len(df.columns)} columns)"
        dialog.setWindowTitle(title)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        table = VDataframeTable(layout)
        table.show(df_limited, fill_colors=False)  # Don't color-code map data
        
        dialog.exec()
    
    def _on_save_map_requested(self):
        """Save the current map to an Excel file."""
        from PySide6.QtWidgets import QFileDialog
        
        if not self.vm.current_map_name:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Map Data",
            f"{self.vm.current_map_name}.xlsx",
            "Excel Files (*.xlsx)"
        )
        
        if file_path:
            self.vm.save_current_map_to_excel(file_path)
    
    def _on_select_all_spectra(self):
        """Select all spectra in the current map."""
        self.vm.select_all_current_map_spectra()
        # Update the list widget to show all selected (both checked and highlighted)
        self.v_maps_list.check_all_spectra(True)
        self.v_maps_list.select_all_spectra()
    
    def _on_reinit_spectra(self):
        """Reinitialize selected spectra (Ctrl for all maps)."""
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        
        # Check if Ctrl is held
        modifiers = QApplication.keyboardModifiers()
        apply_all = bool(modifiers & Qt.ControlModifier)
        
        self.vm.reinit_current_map_spectra(apply_all)
    
    def _on_send_to_spectra(self):
        """Send selected spectra to the Spectra workspace."""
        self.vm.send_selected_spectra_to_spectra_workspace()
    
    def _receive_spectra_from_maps(self, spectra_list: list):
        """Receive spectra from Maps workspace and add to parent's Spectra workspace."""
        # Access the parent window's Spectra workspace
        parent_window = self.window()
        if not hasattr(parent_window, 'v_spectra_workspace'):
            return
        
        # Add spectra to the Spectra workspace
        spectra_workspace = parent_window.v_spectra_workspace
        for spectrum in spectra_list:
            spectra_workspace.vm.spectra.add(spectrum)
        
        # Update the Spectra workspace view
        spectra_workspace.vm._emit_list_update()
        
        # Switch to Spectra tab
        if hasattr(parent_window, 'tabWidget'):
            parent_window.tabWidget.setCurrentWidget(spectra_workspace)
    
    def _on_extract_profile_requested(self, profile_name: str):
        """Handle profile extraction request from map viewer."""
        # Extract profile data from map viewer
        profile_df = self.v_map_viewer._extract_profile()
        
        if profile_df is None or profile_df.empty:
            show_toast_notification(
                parent=self,
                message="Select exactly 2 points in 2D map mode to extract a profile.",
                duration=3000
            )
            return
        
        # Send to ViewModel which will emit signal to Graphs workspace
        self.vm.extract_and_send_profile_to_graphs(profile_name, profile_df)
    
    def _on_extract_profile_from_dialog(self, profile_name: str, dialog):
        """Handle profile extraction request from dialog viewer."""
        # Extract profile data from dialog's map viewer
        profile_df = dialog.map_viewer._extract_profile()
        
        if profile_df is None or profile_df.empty:
            show_toast_notification(
                parent=self,
                message="Select exactly 2 points in 2D map mode to extract a profile.",
                duration=3000
            )
            return
        
        # Send to ViewModel which will emit signal to Graphs workspace
        self.vm.extract_and_send_profile_to_graphs(profile_name, profile_df)
    
    def _show_toast_notification(self, message: str):
        """Show auto-dismissing toast notification."""
        show_toast_notification(
            parent=self,
            message=message,
            duration=3000
        )
    
    def clear_workspace(self):
        """Clear the Maps workspace (called from main window)."""
        # Clear griddata cache in map viewer
        if hasattr(self, 'v_map_viewer'):
            if self.v_map_viewer:
                self.v_map_viewer._griddata_cache.clear()
        
        # Clear cache in all dialog viewers
        for dialog in self.viewer_dialogs:
            dialog.clear_cache_for_map("")
        
        # Close all dialog viewers
        for dialog in list(self.viewer_dialogs):  # Copy list to avoid modification during iteration
            dialog.close()
        self.viewer_dialogs.clear()
        self.next_viewer_number = 2
        
        # Delegate to ViewModel for data clearing
        self.vm.clear_workspace()
    
    def save_work(self):
        """Trigger save work in ViewModel."""
        self.vm.save_work()

    def load_work(self, file_path: str):
        """Trigger load work in ViewModel."""
        from PySide6.QtCore import QTimer
        
        # Delegate to ViewModel
        self.vm.load_work(file_path)
        
        # Clear griddata cache since we loaded new data
        if hasattr(self, 'v_map_viewer'):
            if self.v_map_viewer:
                self.v_map_viewer._griddata_cache.clear()
        
        # Clear cache in all dialog viewers
        for dialog in self.viewer_dialogs:
            dialog.clear_cache_for_map("")
        
        # Delay fit results collection to ensure UI is ready (matches legacy)
        self.vm.collect_fit_results()