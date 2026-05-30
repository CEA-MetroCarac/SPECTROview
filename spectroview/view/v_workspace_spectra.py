"""View for Spectra Workspace - main UI coordinator for spectral analysis."""
import os

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QApplication,
    QPushButton, QCheckBox, QProgressBar, QSplitter, QTabWidget,
    QFrame
)

from spectroview import ICON_DIR
from spectroview.model.m_settings import MSettings

from spectroview.viewmodel.utils import show_toast_notification
from spectroview.view.components.v_fit_model_builder import VFitModelBuilder
from spectroview.view.components.v_fit_results import VFitResults
from spectroview.view.components.v_moretab  import VMoreTab
from spectroview.view.components.v_mva import VMVA
from spectroview.view.components.v_spectra_list import VSpectraList
from spectroview.view.components.v_spectra_viewer import VSpectraViewer
from spectroview.viewmodel.vm_fit_model_builder import VMFitModelBuilder
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.viewmodel.vm_mva import VMMVA


class VWorkspaceSpectra(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_settings = MSettings()
        self.vm = VMWorkspaceSpectra(self.m_settings) # To bind View to ViewModel
        
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Main vertical splitter
        main_splitter = QSplitter(Qt.Horizontal, self)
        main_layout.addWidget(main_splitter)

        # LEFT SIDE (Viewer + Tabs)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_splitter = QSplitter(Qt.Vertical, left_widget)
        left_layout.addWidget(left_splitter)

        # --- Upper: SpectraViewer (placeholder for now)
        self.v_spectra_viewer = VSpectraViewer(parent=self)
        self.v_spectra_viewer.setMinimumHeight(200)
        
        # Shortcut for rescale - only works when this workspace is active
        self.shortcut_rescale = QShortcut(QKeySequence("Ctrl+R"), self)
        self.shortcut_rescale.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_rescale.activated.connect(self.v_spectra_viewer._rescale)
        
        # --- Lower: TabWidget
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setMinimumHeight(150)
        self.v_fit_model_builder = VFitModelBuilder()
        self.v_fit_results = VFitResults()
        self.v_more_tab = VMoreTab()
        self.v_mva = VMVA()
        self.vm_fit_model_builder = VMFitModelBuilder(self.m_settings)
        self.vm_mva = VMMVA(self.m_settings)
        self.vm.set_fit_model_builder(self.vm_fit_model_builder) # 🔑 inject dependency
        
        self.bottom_tabs.addTab(self.v_fit_model_builder, "Fit Model Builder")
        self.bottom_tabs.addTab(self.v_fit_results, "Fit Results")
        self.bottom_tabs.addTab(self.v_more_tab, "More")
        self.bottom_tabs.addTab(self.v_mva, "MVA")
        
        left_splitter.addWidget(self.v_spectra_viewer)
        left_splitter.addWidget(self.bottom_tabs)
        left_splitter.setSizes([530, 470])

        # RIGHT SIDE (Sidebar)
        right_widget = QFrame()
        right_widget.setMinimumWidth(200)
        right_widget.setMaximumWidth(450)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        # --- Top buttons row
        buttons_layout = QHBoxLayout()

        self.btn_select_all = QPushButton()
        self.btn_select_all.setIcon(QIcon(os.path.join(ICON_DIR, "select-all.png")))
        self.btn_select_all.setToolTip("Select all spectra")
        
        self.btn_remove = QPushButton()
        self.btn_remove.setIcon(QIcon(os.path.join(ICON_DIR, "trash.png")))
        self.btn_remove.setToolTip("Remove selected spectra")

        self.btn_reinit = QPushButton("Reinit")
        self.btn_reinit.setIcon(QIcon(os.path.join(ICON_DIR, "undo2.png")))
        self.btn_reinit.setToolTip("Reinitialize selected spectra to original values")
        self.btn_stats = QPushButton("Stats")
        self.btn_stats.setIcon(QIcon(os.path.join(ICON_DIR, "stats.png")))
        self.btn_stats.setToolTip("View fitting statistics of selected spectra")

        self.btn_save_spectra_data = QPushButton("Save Data")
        self.btn_save_spectra_data.setIcon(QIcon(os.path.join(ICON_DIR, "save.png")))
        self.btn_save_spectra_data.setToolTip("Save selected spectra data to TXT file")

        for btn in (self.btn_select_all, self.btn_remove, self.btn_reinit, self.btn_stats, self.btn_save_spectra_data):
            buttons_layout.addWidget(btn)

        right_layout.addLayout(buttons_layout)

        # --- Check all checkbox
        self.cb_check_all = QCheckBox("Check All")
        self.cb_check_all.setChecked(True)  # Checked by default
        right_layout.addWidget(self.cb_check_all)

        # --- Spectra list
        self.v_spectra_list = VSpectraList()
        right_layout.addWidget(self.v_spectra_list, stretch=1)

        # --- Footer: count + progress + stop button
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(4)
        
        self.lbl_count = QLabel("0 spectra")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(100)
        self.progress_bar.setFixedHeight(17)
        
        self.btn_stop_fit = QPushButton("Stop")
        self.btn_stop_fit.setIcon(QIcon(os.path.join(ICON_DIR, "stop.png")))
        self.btn_stop_fit.setFixedHeight(17)
        self.btn_stop_fit.setFixedWidth(50)
        self.btn_stop_fit.setVisible(False)  # Hidden by default

        footer_layout.addWidget(self.lbl_count)
        footer_layout.addWidget(self.progress_bar)
        footer_layout.addWidget(self.btn_stop_fit)
        right_layout.addLayout(footer_layout)

        # Assemble main splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([900, 400])

    def _apply_with_ctrl(self, fn):
        apply_all = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        fn(apply_all)

    def _update_progress_bar(self, current: int, total: int, percentage: int, elapsed_time: float, converged: int = 0):
        """Update progress bar with fitting progress and elapsed time."""
        if total > 0:
            self.progress_bar.setValue(percentage)
            # Format elapsed time in seconds with 2 decimal places
            time_str = f"{elapsed_time:.2f}s"
            self.progress_bar.setFormat(f"({percentage}%) converged {converged}/{total} - {time_str}")
        else:
            # Reset to default state
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("")
            self.progress_bar.setToolTip("")
    
    def _on_spectra_list_changed(self, spectra: list):
        """Handle spectra list update from ViewModel."""
        self.v_spectra_list.set_spectra_names(spectra)
    
    def _on_checkbox_changed(self, item):
        """Update spectrum.is_active when checkbox state changes."""
        if item is None:
            return
        
        fname = item.data(Qt.UserRole + 1)
        if fname:
            is_checked = item.checkState() == Qt.Checked
            md = self.vm.store.get_map_data(fname)
            if md:
                md.is_active[0] = is_checked
    
    def _on_check_all_toggled(self, checked: bool):
        """Handle check all checkbox toggle."""
        # Block signals temporarily to avoid triggering individual checkbox handlers
        self.v_spectra_list.blockSignals(True)
        
        # Update all checkboxes in the list
        for i in range(self.v_spectra_list.count()):
            item = self.v_spectra_list.item(i)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            
            # Update store
            fname = item.data(Qt.UserRole + 1)
            if fname:
                md = self.vm.store.get_map_data(fname)
                if md:
                    md.is_active[0] = checked
                    
        # Restore signals
        self.v_spectra_list.blockSignals(False)

    def setup_connections(self):
        """Connect ViewModel signals and slots to the View components."""
        vm = self.vm # VMWorkspaceSpectra
        
        self.btn_select_all.clicked.connect(self.v_spectra_list.select_all)
        self.btn_remove.clicked.connect(vm.remove_selected_spectra)
        self.btn_reinit.clicked.connect(lambda: self._apply_with_ctrl(vm.reinit_spectra))
        self.btn_reinit.clicked.connect(self.v_spectra_viewer._rescale)  # Auto-rescale after reinit
        self.btn_stats.clicked.connect(lambda: vm.view_stats(parent_widget=self))
        self.btn_save_spectra_data.clicked.connect(lambda: vm.save_spectra_data(parent_widget=self))

        # Connection with VMWorkspaceSpectra (vm)
        self.v_spectra_list.selection_changed.connect(vm.set_selected_indices) # V Notify VM of selection change
        self.v_spectra_list.order_changed.connect(vm.reorder_spectra)

        self.v_spectra_viewer.peak_add_requested.connect(vm.add_peak_at)
        self.v_spectra_viewer.peak_remove_requested.connect(vm.remove_peak_at)
        self.v_spectra_viewer.baseline_add_requested.connect(vm.add_baseline_point)
        self.v_spectra_viewer.baseline_remove_requested.connect(vm.remove_baseline_point)
        self.v_spectra_viewer.copy_data_requested.connect(vm.copy_spectrum_data_to_clipboard)
        self.v_spectra_viewer.spectrumCustomized.connect(vm._emit_selected_spectra)
        self.v_spectra_viewer.spectrumCustomized.connect(vm._emit_list_update)
        
        # Peak dragging
        self.v_spectra_viewer.peak_dragged.connect(vm.update_dragged_peak)
        self.v_spectra_viewer.peak_drag_finished.connect(vm.finalize_peak_drag)

        self.v_fit_model_builder.btn_xcorrect.clicked.connect(lambda: vm.apply_x_correction(self.v_fit_model_builder.spin_xcorr.value()))
        self.v_fit_model_builder.btn_undo_corr.clicked.connect(vm.undo_x_correction)
        self.v_fit_model_builder.spectral_range_apply_requested.connect(vm.apply_spectral_range)
        self.v_fit_model_builder.spectral_range_apply_requested.connect(self.v_spectra_viewer._rescale)  # Auto-rescale after crop
        self.v_fit_model_builder.baseline_settings_changed.connect(vm.set_baseline_settings)
        self.v_fit_model_builder.baseline_preview_requested.connect(vm.preview_baseline)

        self.v_fit_model_builder.baseline_copy_requested.connect(vm.copy_baseline)
        self.v_fit_model_builder.baseline_paste_requested.connect(vm.paste_baseline)
        self.v_fit_model_builder.baseline_subtract_requested.connect(vm.subtract_baseline)
        self.v_fit_model_builder.baseline_subtract_requested.connect(self.v_spectra_viewer._rescale)  # Auto-rescale after subtract
        self.v_fit_model_builder.baseline_delete_requested.connect(vm.delete_baseline)

        self.v_fit_model_builder.peaks_copy_requested.connect(vm.copy_peaks)
        self.v_fit_model_builder.peaks_paste_requested.connect(vm.paste_peaks)
        self.v_fit_model_builder.peaks_delete_requested.connect(vm.delete_peaks)
        self.v_fit_model_builder.peak_shape_changed.connect(self.vm.set_peak_shape)

        #Fit control
        self.v_fit_model_builder.fit_requested.connect(vm.fit)
        self.v_fit_model_builder.fitmodel_copy_requested.connect(vm.copy_fit_model)
        self.v_fit_model_builder.fitmodel_paste_requested.connect(vm.paste_fit_model)
        self.v_fit_model_builder.fitmodel_paste_requested.connect(self.v_spectra_viewer._rescale)  # Auto-rescale after paste
        self.v_fit_model_builder.fitmodel_save_requested.connect(vm.save_fit_model)

        # Check all checkbox
        self.cb_check_all.toggled.connect(self._on_check_all_toggled)
        
        # SpectraList connection: ViewModel → View
        vm.spectra_list_changed.connect(self._on_spectra_list_changed)
        self.v_spectra_list.itemChanged.connect(self._on_checkbox_changed)
        vm.spectra_selection_changed.connect(self.v_spectra_viewer.set_plot_data)
        vm.spectra_selection_changed.connect(self._update_metadata_display)
        vm.spectra_selection_changed.connect(self.v_fit_model_builder.update_baseline_ui)
        vm.spectra_selection_changed.connect(self._update_peak_table)
        vm.count_changed.connect(lambda n: self.lbl_count.setText(f"{n} spectra"))
        vm.notify.connect(self._show_toast_notification)

        # MetaData connections
        self.v_more_tab.normalize_requested.connect(
            lambda factor: self._apply_with_ctrl(lambda apply_all: vm.apply_y_normalization(factor, apply_all))
        )
        self.v_more_tab.undo_normalization_requested.connect(
            lambda: self._apply_with_ctrl(vm.undo_y_normalization)
        )
        self.v_more_tab.cosmic_ray_requested.connect(vm.cosmic_ray_detection)

        # ═════════════════════════════════════════════════════════════════
        # MVA connections
        # ═════════════════════════════════════════════════════════════════
        self.vm_mva.set_store(self.vm.store)  # inject SpectraStore reference
        self.v_mva.run_pca_requested.connect(self.vm_mva.run_pca)
        self.v_mva.run_nmf_requested.connect(self.vm_mva.run_nmf)
        self.v_mva.send_to_graphs_requested.connect(self.vm_mva.send_to_graphs)
        self.vm_mva.pca_results_ready.connect(self.v_mva.display_pca_results)
        self.vm_mva.nmf_results_ready.connect(self.v_mva.display_nmf_results)
        self.vm_mva.send_df_to_graphs.connect(self._send_df_to_graphs)
        self.vm_mva.notify.connect(self._show_toast_notification)
        self.vm_mva.notify.connect(self.v_mva.set_status)

        vm.show_xcorrection_value.connect(self.v_fit_model_builder.set_xcorrection_value)        
        vm.spectral_range_changed.connect(self.v_fit_model_builder.set_spectral_range)
        vm.fit_in_progress.connect(lambda in_progress: self.v_fit_model_builder.set_fit_buttons_enabled(not in_progress))
        vm.fit_in_progress.connect(self.btn_stop_fit.setVisible)  # Show/hide Stop button
        vm.fit_progress_updated.connect(self._update_progress_bar)
        vm.fit_timings_ready.connect(self.progress_bar.setToolTip)
        
        # Send DataFrame to Graphs workspace
        vm.send_df_to_graphs.connect(self._send_df_to_graphs)
        
        # Stop button → ViewModel
        self.btn_stop_fit.clicked.connect(vm.stop_fit)

        # V_FitModelBuilder <-> VM_FitModelBuilder
        self.v_fit_model_builder.refresh_fit_models_requested.connect(self.vm_fit_model_builder.refresh_models)
        self.v_fit_model_builder.load_fit_models_requested.connect(self.vm_fit_model_builder.pick_and_load_model)
        self.v_fit_model_builder.cbb_model.currentTextChanged.connect(self.vm_fit_model_builder.set_current_model)
        self.v_fit_model_builder.apply_loaded_fit_model_requested.connect(vm.apply_fit_model)
        self.v_fit_model_builder.apply_loaded_fit_model_requested.connect(self.v_spectra_viewer._rescale)  # Auto-rescale after apply

        self.vm_fit_model_builder.models_changed.connect(self.v_fit_model_builder.cbb_model.clear)
        self.vm_fit_model_builder.models_changed.connect(self.v_fit_model_builder.cbb_model.addItems)
        self.vm_fit_model_builder.model_selected.connect(self.v_fit_model_builder.cbb_model.setCurrentText)
        self.vm_fit_model_builder.refresh_models() # Initial load of models


        # PeakTable → VMWorkspaceSpectra
        pt = self.v_fit_model_builder.peak_table
        pt.peak_label_changed.connect(vm.update_peak_label)
        pt.peak_model_changed.connect(vm.update_peak_model)
        pt.peak_param_changed.connect(vm.update_peak_param)
        pt.peak_deleted.connect(vm.delete_peak)

        # Selection → PeakTable
        vm.spectra_selection_changed.connect(self._update_peak_table)
        
        # ═════════════════════════════════════════════════════════════════
        # Fit Results connections
        # ═════════════════════════════════════════════════════════════════
        self.v_fit_results.collect_results_requested.connect(vm.collect_fit_results)
        self.v_fit_results.split_fname_requested.connect(vm.split_filename)
        self.v_fit_results.add_column_requested.connect(vm.add_column_from_filename)
        self.v_fit_results.compute_column_requested.connect(vm.compute_column_from_expression)
        self.v_fit_results.save_results_requested.connect(vm.save_fit_results)
        self.v_fit_results.send_to_viz_requested.connect(vm.send_results_to_graphs)
        
        # ViewModel → View for fit results
        vm.fit_results_updated.connect(self._update_fit_results)
        vm.split_parts_updated.connect(self.v_fit_results.populate_split_combobox)

    def _update_fit_results(self, df):
        """Update fit results table display."""
        if df is not None and not df.empty:
            self.v_fit_results.show_results(df)
        else:
            self.v_fit_results.clear_results()
    
    def _update_metadata_display(self, selected_spectra):
        """Update metadata display with first selected spectrum's metadata."""
        specs = selected_spectra.get("proxies", []) if isinstance(selected_spectra, dict) else selected_spectra
        if specs and len(specs) > 0:
            spectrum = specs[0]
            self.v_more_tab.show_metadata(spectrum)
        else:
            self.v_more_tab.clear_metadata()
    
    def save_work(self):
        """Trigger save work in ViewModel."""
        self.vm.save_work()

    def load_work(self, file_path: str):
        """Trigger load work in ViewModel."""
        self.vm.load_work(file_path)

    def clear_workspace(self):
        """Trigger workspace clear in ViewModel."""
        self.vm.clear_workspace()
    

    def _send_df_to_graphs(self, df_name: str, df: pd.DataFrame):
        """Forward DataFrame to Graphs workspace"""
        # Access the parent window's Graphs workspace
        parent_window = self.window()
        if not hasattr(parent_window, 'v_graphs_workspace'):
            return
        
        # Add DataFrame to Graphs workspace via its ViewModel
        graphs_workspace = parent_window.v_graphs_workspace
        graphs_workspace.vm.add_dataframe(df_name, df)
        
    def _show_toast_notification(self, message: str):
        """Show auto-dismissing toast notification."""
        show_toast_notification(
            parent=self,
            message=message,
            duration=3000
        )

    def _update_peak_table(self, spectra):
        """Update the peak table with the first selected spectrum."""
        if not spectra:
            self.v_fit_model_builder.peak_table.clear()
            return
        
        if isinstance(spectra, dict):
            # Tensor dictionary payload
            self.v_fit_model_builder.peak_table.set_spectrum(spectra)
        elif isinstance(spectra, list) and len(spectra) > 0:
            # Legacy list of objects
            self.v_fit_model_builder.peak_table.set_spectrum(spectra[0])
        else:
            self.v_fit_model_builder.peak_table.clear()

