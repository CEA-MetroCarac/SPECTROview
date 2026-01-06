"""View for Spectra Workspace - main UI coordinator for spectral analysis."""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QApplication,
    QPushButton, QCheckBox, QProgressBar, QSplitter, QTabWidget,
    QMessageBox, QFrame
)

from spectroview import ICON_DIR
from spectroview.model.m_settings import MSettings
from spectroview.view.components.v_fit_model_builder import VFitModelBuilder
from spectroview.view.components.v_fit_results import VFitResults
from spectroview.view.components.v_spectra_list import VSpectraList
from spectroview.view.components.v_spectra_viewer import VSpectraViewer
from spectroview.viewmodel.vm_fit_model_builder import VMFitModelBuilder
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra


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
        self.vm_fit_model_builder = VMFitModelBuilder(self.m_settings)
        self.vm.set_fit_model_builder(self.vm_fit_model_builder) # 🔑 inject dependency
        
        self.bottom_tabs.addTab(self.v_fit_model_builder, "Fit Model Builder")
        self.bottom_tabs.addTab(self.v_fit_results, "Fit Results")
        
        left_splitter.addWidget(self.v_spectra_viewer)
        left_splitter.addWidget(self.bottom_tabs)
        left_splitter.setSizes([550, 450])

        # RIGHT SIDE (Sidebar)
        right_widget = QFrame()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setFrameShape(QFrame.StyledPanel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        # --- Top buttons row
        buttons_layout = QHBoxLayout()

        self.btn_select_all = QPushButton()
        self.btn_select_all.setIcon(QIcon(os.path.join(ICON_DIR, "select-all.png")))
        
        self.btn_remove = QPushButton()
        self.btn_remove.setIcon(QIcon(os.path.join(ICON_DIR, "trash.png")))
        self.btn_reinit = QPushButton("Reinit")
        self.btn_stats = QPushButton("Stats")

        for btn in (self.btn_select_all, self.btn_remove, self.btn_reinit, self.btn_stats):
            buttons_layout.addWidget(btn)

        right_layout.addLayout(buttons_layout)

        # --- Check all checkbox
        self.cb_check_all = QCheckBox("Check All")
        right_layout.addWidget(self.cb_check_all)

        # --- Spectra list
        self.v_spectra_list = VSpectraList()
        right_layout.addWidget(self.v_spectra_list, stretch=1)

        # --- Footer: count + progress
        self.lbl_count = QLabel("Loaded spectra: 0")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(100)
        self.progress_bar.setFixedHeight(15)

        right_layout.addWidget(self.lbl_count)
        right_layout.addWidget(self.progress_bar)

        # Assemble main splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([900, 200])

    def _apply_with_ctrl(self, fn):
        apply_all = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        fn(apply_all)

    def _update_progress_bar(self, current: int, total: int, percentage: int, elapsed_time: float):
        """Update progress bar with fitting progress and elapsed time."""
        if total > 0:
            self.progress_bar.setValue(percentage)
            # Format elapsed time in seconds with 2 decimal places
            time_str = f"{elapsed_time:.2f}s"
            self.progress_bar.setFormat(f"Fitting: {current}/{total} ({percentage}%) - {time_str}")
        else:
            # Reset to default state
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("")

    def setup_connections(self):
        """Connect ViewModel signals and slots to the View components."""
        vm = self.vm # VMWorkspaceSpectra
        
        self.btn_select_all.clicked.connect(self.v_spectra_list.select_all)
        self.btn_remove.clicked.connect(vm.remove_selected_spectra)
        self.btn_reinit.clicked.connect(lambda: self._apply_with_ctrl(vm.reinit_spectra))

        # Connection with VMWorkspaceSpectra (vm)
        self.v_spectra_list.selection_changed.connect(vm.set_selected_indices) # V Notify VM of selection change
        self.v_spectra_list.files_dropped.connect(vm.load_files)
        self.v_spectra_list.order_changed.connect(vm.reorder_spectra)

        self.v_spectra_viewer.peak_add_requested.connect(vm.add_peak_at)
        self.v_spectra_viewer.peak_remove_requested.connect(vm.remove_peak_at)
        self.v_spectra_viewer.baseline_add_requested.connect(vm.add_baseline_point)
        self.v_spectra_viewer.baseline_remove_requested.connect(vm.remove_baseline_point)
        self.v_spectra_viewer.copy_data_requested.connect(vm.copy_spectrum_data_to_clipboard)
        
        # Peak dragging
        self.v_spectra_viewer.peak_dragged.connect(vm.update_dragged_peak)
        self.v_spectra_viewer.peak_drag_finished.connect(vm.finalize_peak_drag)

        self.v_fit_model_builder.btn_xcorrect.clicked.connect(lambda: vm.apply_x_correction(self.v_fit_model_builder.spin_xcorr.value()))
        self.v_fit_model_builder.btn_undo_corr.clicked.connect(vm.undo_x_correction)
        self.v_fit_model_builder.spectral_range_apply_requested.connect(vm.apply_spectral_range)
        self.v_fit_model_builder.baseline_settings_changed.connect(vm.set_baseline_settings)

        self.v_fit_model_builder.baseline_copy_requested.connect(vm.copy_baseline)
        self.v_fit_model_builder.baseline_paste_requested.connect(vm.paste_baseline)
        self.v_fit_model_builder.baseline_subtract_requested.connect(vm.subtract_baseline)
        #self.v_fit_model_builder.baseline_delete_requested.connect(vm.apply_spectral_range)

        self.v_fit_model_builder.peaks_copy_requested.connect(vm.copy_peaks)
        self.v_fit_model_builder.peaks_paste_requested.connect(vm.paste_peaks)
        self.v_fit_model_builder.peaks_delete_requested.connect(vm.delete_peaks)
        self.v_fit_model_builder.peak_shape_changed.connect(self.vm.set_peak_shape)

        #Fit control
        self.v_fit_model_builder.fit_requested.connect(vm.fit)
        self.v_fit_model_builder.fitmodel_copy_requested.connect(vm.copy_fit_model)
        self.v_fit_model_builder.fitmodel_paste_requested.connect(vm.paste_fit_model)
        self.v_fit_model_builder.fitmodel_save_requested.connect(vm.save_fit_model)

        # SpectraList connection: ViewModel → View
        vm.spectra_list_changed.connect(self.v_spectra_list.set_spectra_names)
        vm.spectra_selection_changed.connect(self.v_spectra_viewer.set_plot_data)
        vm.count_changed.connect(lambda n: self.lbl_count.setText(f"{n} spectra loaded"))
        vm.notify.connect(lambda msg: QMessageBox.information(self, "Spectra already loaded", msg))

        vm.show_xcorrection_value.connect(self.v_fit_model_builder.set_xcorrection_value)        
        vm.spectral_range_changed.connect(self.v_fit_model_builder.set_spectral_range)
        vm.fit_in_progress.connect(lambda in_progress: self.v_fit_model_builder.set_fit_buttons_enabled(not in_progress))
        vm.fit_progress_updated.connect(self._update_progress_bar)

        # V_FitModelBuilder <-> VM_FitModelBuilder
        self.v_fit_model_builder.refresh_fit_models_requested.connect(self.vm_fit_model_builder.refresh_models)
        self.v_fit_model_builder.load_fit_models_requested.connect(self.vm_fit_model_builder.pick_and_load_model)
        self.v_fit_model_builder.cbb_model.currentTextChanged.connect(self.vm_fit_model_builder.set_current_model)
        self.v_fit_model_builder.apply_loaded_fit_model_requested.connect(vm.apply_loaded_fit_model)

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
        vm.spectra_selection_changed.connect(
            lambda specs:
            pt.set_spectrum(specs[0] if specs else None)
        )
        
        # ═════════════════════════════════════════════════════════════════
        # Fit Results connections
        # ═════════════════════════════════════════════════════════════════
        self.v_fit_results.collect_results_requested.connect(vm.collect_fit_results)
        self.v_fit_results.split_fname_requested.connect(vm.split_filename)
        self.v_fit_results.add_column_requested.connect(vm.add_column_from_filename)
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
    
    def save_work(self):
        """Trigger save work in ViewModel."""
        self.vm.save_work()

    def load_work(self, file_path: str):
        """Trigger load work in ViewModel."""
        self.vm.load_work(file_path)

    def clear_workspace(self):
        """Trigger workspace clear in ViewModel."""
        self.vm.clear_workspace()

