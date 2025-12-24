# view/workspace_spectra.py
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout,QHBoxLayout, QLabel,
    QPushButton, QCheckBox,QProgressBar,QSplitter,QTabWidget, QMessageBox, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from spectroview.view.components.v_spectra_list import VSpectraList
from spectroview.view.components.v_spectra_viewer import VSpectraViewer
from spectroview.view.components.v_fit_model_builder import VFitModelBuilder

from spectroview.viewmodel.vm_spectra import VMSpectra

from spectroview import ICON_DIR


class WorkspaceSpectra(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vm = VMSpectra()

        self.init_ui()
        self.connect_vm()

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
        self.spectra_viewer = VSpectraViewer(parent=self)
        self.spectra_viewer.setMinimumHeight(200)
        
        # --- Lower: TabWidget
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setMinimumHeight(150)
        self.fit_model_builder = VFitModelBuilder()
        self.bottom_tabs.addTab(self.fit_model_builder, "Fit Model Builder")
        self.bottom_tabs.addTab(QWidget(), "Fit Results")
        
        left_splitter.addWidget(self.spectra_viewer)
        left_splitter.addWidget(self.bottom_tabs)
        left_splitter.setSizes([500, 500])

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
        self.spectra_list = VSpectraList()
        right_layout.addWidget(self.spectra_list, stretch=1)

        # --- Footer: count + progress
        self.lbl_count = QLabel("Loaded spectra: 0")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(100)
        self.progress_bar.setFixedHeight(12)

        right_layout.addWidget(self.lbl_count)
        right_layout.addWidget(self.progress_bar)

        # Assemble main splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([900, 200])

    def connect_vm(self):
        """Connect ViewModel signals and slots to the View components."""
        v = self.spectra_viewer
        vm = self.vm 
        
        # SpectraList connection: View → ViewModel
        self.spectra_list.selection_changed.connect(vm.set_selected_indices) # V Notify VM of selection change
        self.btn_select_all.clicked.connect(self.spectra_list.select_all)
        self.btn_remove.clicked.connect(vm.remove_selected_spectra)
        self.spectra_list.files_dropped.connect(vm.load_files)

        # SpectraViewer connections: View → ViewModel
        v.peak_add_requested.connect(vm.add_peak_at)
        v.peak_remove_requested.connect(vm.remove_peak_at)
        v.baseline_add_requested.connect(vm.add_baseline_point)
        v.baseline_remove_requested.connect(vm.remove_baseline_point)

        # Fit Model Builder connections : view → viewmodel
        
        self.fit_model_builder.btn_correct.clicked.connect(lambda: vm.apply_x_correction(self.fit_model_builder.spin_xcorr.value()))
        self.fit_model_builder.btn_undo_corr.clicked.connect(vm.undo_x_correction)
        
        
        # SpectraList connection: ViewModel → View
        vm.spectra_list_changed.connect(self.spectra_list.set_spectra_names)
        vm.spectra_selection_changed.connect(self.spectra_viewer.set_plot_data)
        vm.count_changed.connect(lambda n: self.lbl_count.setText(f"{n} spectra loaded"))
        vm.notify.connect(lambda msg: QMessageBox.information(self, "Spectra already loaded", msg))

        vm.show_xcorrection_value.connect(self.fit_model_builder.set_xcorrection_value)        

