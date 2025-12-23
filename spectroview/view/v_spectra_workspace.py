# view/spectra_workspace.py
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout,QHBoxLayout, QLabel,
    QPushButton, QCheckBox,QProgressBar,QSplitter,QTabWidget, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from spectroview.view.components.v_spectra_list import SpectraList
from spectroview.view.components.v_spectra_viewer import SpectraViewer

from spectroview.viewmodel.vm_spectra import SpectraVM

from spectroview import ICON_DIR


class SpectraWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
       
        
        # ✅ Single ViewModel
        self.vm = SpectraVM()

        self.init_ui()
        self.connect_vm()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # =========================
        # Main vertical splitter
        # =========================
        main_splitter = QSplitter(Qt.Horizontal, self)
        main_layout.addWidget(main_splitter)

        # ======================================================
        # LEFT SIDE (Viewer + Tabs)
        # ======================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_splitter = QSplitter(Qt.Vertical, left_widget)
        left_layout.addWidget(left_splitter)

        # --- Upper: SpectraViewer (placeholder for now)
        self.spectra_viewer = SpectraViewer(parent=self)
        self.spectra_viewer.setMinimumHeight(200)
        
        # --- Lower: TabWidget
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setMinimumHeight(150)
        self.bottom_tabs.addTab(QWidget(), "Fit Model Builder")
        self.bottom_tabs.addTab(QWidget(), "Fit Results")
        
        left_splitter.addWidget(self.spectra_viewer)
        left_splitter.addWidget(self.bottom_tabs)
        left_splitter.setSizes([600, 500])

        # ======================================================
        # RIGHT SIDE (Sidebar)
        # ======================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
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
        self.spectra_list = SpectraList()
        right_layout.addWidget(self.spectra_list, stretch=1)

        # --- Footer: count + progress
        self.lbl_count = QLabel("Loaded spectra: 0")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(15)

        right_layout.addWidget(self.lbl_count)
        right_layout.addWidget(self.progress_bar)

        # ======================================================
        # Assemble main splitter
        # ======================================================
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([900, 200])
        # main_splitter.setStretchFactor(0, 2)
        # main_splitter.setStretchFactor(1, 1)


    def connect_vm(self):
        # View → ViewModel
        self.spectra_list.selection_changed.connect(
            self.vm.set_selected_indices
        )
      
        self.spectra_list.files_dropped.connect(self.vm.load_files)
        
        self.vm.notify.connect(lambda msg: QMessageBox.information(self, "Spectra already loaded", msg))
        
        # ViewModel → View
        self.vm.spectra_list_changed.connect(
            self.spectra_list.set_spectra_names
        )
        self.vm.spectra_selection_changed.connect(
            self.spectra_viewer.set_plot_data
        )
        self.vm.count_changed.connect(
            lambda n: self.lbl_count.setText(f"Loaded spectra: {n}")
        )


