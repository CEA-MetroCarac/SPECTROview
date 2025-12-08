import os
from functools import partial
import logging
from spectroview import PEAK_MODELS, ICON_DIR
from spectroview.modules.file_converter import FileConverter
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets  import QWidget


logger = logging.getLogger(__name__)

class UiConnector:
    """Centralize all signal wiring / UI connections here. """
    def __init__(self, app_settings, ui, main_app, maps_tab, spectrums_tab, graphs_tab, setting_panel):
        
        self.app_settings = app_settings
        self.setting_panel = setting_panel
        
        self.ui = ui
        self.main_app = main_app
        self.maps = maps_tab
        self.spectrums = spectrums_tab
        self.graphs = graphs_tab

        # sync helper
        self.sync_settings = partial(self.app_settings.sync_app_settings, self.ui)
        self.connect_all_signals()

    def _attach_sync_to_widget(self, widget):
        """Attach sync_settings to common signal names if they exist on widget."""
        for sig_name in ("valueChanged", "stateChanged", "textChanged", "currentIndexChanged", "toggled"):
            sig = getattr(widget, sig_name, None)
            if sig:
                sig.connect(self.sync_settings)

    def connect_main_tab_signals(self):
        self.ui.actionOpen.triggered.connect(lambda: self.main_app.open())
        self.ui.actionSave.triggered.connect(self.main_app.save)
        self.ui.actionClear_env.triggered.connect(self.main_app.clear_env)
        self.ui.actionDarkLight.triggered.connect(lambda: self.main_app.toggle_dark_light_mode())
        self.ui.actionAbout.triggered.connect(self.main_app.show_about)
        self.ui.actionHelps.triggered.connect(self.main_app.open_manual)
        
        ### DRAG & DROP to open files 
        self.ui.spectrums_listbox.files_dropped.connect(self.main_app.open)
        self.ui.spectra_listbox.files_dropped.connect(self.main_app.open)


    def connect_maps_tab_signals(self):
        # Remove wafer
        self.ui.btn_remove_wafer.clicked.connect(self.maps.remove_map)

        # Selection shortcuts
        self.ui.btn_sel_all.clicked.connect(self.maps.select_all_spectra)
        self.ui.btn_sel_verti.clicked.connect(self.maps.select_verti)
        self.ui.btn_sel_horiz.clicked.connect(self.maps.select_horiz)
        self.ui.btn_sel_q1.clicked.connect(self.maps.select_Q1)
        self.ui.btn_sel_q2.clicked.connect(self.maps.select_Q2)
        self.ui.btn_sel_q3.clicked.connect(self.maps.select_Q3)
        self.ui.btn_sel_q4.clicked.connect(self.maps.select_Q4)

        # Model & fit actions
        # self.ui.btn_load_model.clicked.connect(self.maps.load_fit_model)
        # self.ui.btn_apply_model.clicked.connect(self.maps.apply_model_fnc_handler)
        self.ui.btn_init.clicked.connect(self.maps.reinit_fnc_handler)
        self.ui.btn_collect_results.clicked.connect(self.maps.collect_results)
        self.ui.btn_view_df_2.clicked.connect(self.maps.view_fit_results_df)
        self.ui.btn_show_stats.clicked.connect(self.maps.view_stats)
        self.ui.btn_save_fit_results.clicked.connect(self.maps.save_fit_results)
        self.ui.btn_view_wafer.clicked.connect(self.maps.view_map_df)

        # self.ui.btn_cosmis_ray.clicked.connect(self.maps.cosmis_ray_detection)

        self.ui.btn_split_fname_2.clicked.connect(self.maps.split_fname)
        self.ui.btn_add_col_2.clicked.connect(self.maps.add_column)

        # Range / baseline
        self.ui.range_max.returnPressed.connect(self.maps.set_x_range)
        self.ui.range_min.returnPressed.connect(self.maps.set_x_range)
        self.ui.range_apply.clicked.connect(self.maps.set_x_range_handler)

        # Fit / models / peaks
        self.ui.btn_fit.clicked.connect(self.maps.fit_fnc_handler)
        self.ui.save_model.clicked.connect(self.maps.save_fit_model)
        self.ui.clear_peaks.clicked.connect(self.maps.clear_peaks_handler)
        self.ui.btn_copy_fit_model.clicked.connect(self.maps.copy_fit_model)
        self.ui.btn_copy_peaks.clicked.connect(self.maps.copy_fit_model)
        self.ui.btn_paste_fit_model.clicked.connect(self.maps.paste_fit_model_fnc_handler)
        self.ui.btn_paste_peaks.clicked.connect(self.maps.paste_peaks_fnc_handler)
        self.ui.cbb_fit_models.addItems(PEAK_MODELS)

        self.ui.btn_undo_baseline.clicked.connect(self.maps.set_x_range_handler)

        self.ui.btn_send_to_compare.clicked.connect(self.maps.send_spectrum_to_compare)
        
        # self.settings_dialog.btn_model_folder.clicked.connect(self.maps.set_default_model_folder)
       

        # For sync settings
        for widget_name in (
            "cb_attached",
            "rbtn_linear",
            "rbtn_polynomial",
            "noise",
            "degre",
        ):
            widget = getattr(self.ui, widget_name, None)
            if widget is not None:
                self._attach_sync_to_widget(widget)
            else:
                logger.warning("Maps tab widget '%s' not found on UI", widget_name)


    def connect_spectra_tab_signals(self):
        self.ui.cbb_fit_models_2.addItems(PEAK_MODELS)
        self.ui.range_apply_2.clicked.connect(self.spectrums.set_x_range_handler)
        self.ui.range_max_2.returnPressed.connect(self.spectrums.set_x_range)
        self.ui.range_min_2.returnPressed.connect(self.spectrums.set_x_range)

        self.ui.sub_baseline_2.clicked.connect(self.spectrums.subtract_baseline_handler)
        self.ui.btn_undo_baseline_2.clicked.connect(self.spectrums.set_x_range_handler)
        self.ui.clear_peaks_2.clicked.connect(self.spectrums.clear_peaks_handler)
        self.ui.btn_fit_3.clicked.connect(self.spectrums.fit_fnc_handler)
        self.ui.btn_copy_fit_model_2.clicked.connect(self.spectrums.copy_fit_model)
        self.ui.btn_copy_peaks_2.clicked.connect(self.spectrums.copy_fit_model)
        self.ui.btn_paste_fit_model_2.clicked.connect(self.spectrums.paste_fit_model_fnc_handler)
        self.ui.btn_paste_peaks_2.clicked.connect(self.spectrums.paste_peaks_fnc_handler)
        self.ui.save_model_2.clicked.connect(self.spectrums.save_fit_model)

        # self.ui.btn_load_model_3.clicked.connect(self.spectrums.load_fit_model)
        # self.ui.btn_apply_model_3.clicked.connect(self.spectrums.apply_model_fnc_handler)
        # self.ui.btn_cosmis_ray_3.clicked.connect(self.spectrums.cosmis_ray_detection)
        self.ui.btn_init_3.clicked.connect(self.spectrums.reinit_fnc_handler)
        self.ui.btn_show_stats_3.clicked.connect(self.spectrums.view_stats)
        self.ui.btn_sel_all_3.clicked.connect(self.spectrums.select_all_spectra)
        self.ui.btn_remove_spectrum.clicked.connect(self.spectrums.remove_spectrum)
        self.ui.btn_collect_results_3.clicked.connect(self.spectrums.collect_results)
        self.ui.btn_view_df_5.clicked.connect(self.spectrums.view_fit_results_df)
        self.ui.btn_save_fit_results_3.clicked.connect(self.spectrums.save_fit_results)

        self.ui.btn_split_fname.clicked.connect(self.spectrums.split_fname)
        self.ui.btn_add_col.clicked.connect(self.spectrums.add_column)

        # self.settings_dialog.btn_model_folder.clicked.connect(self.spectrums.set_default_model_folder)
        
        #For sync settings
        for widget_name in (
            
            "cb_attached_2",
            "rbtn_linear_2",
            "rbtn_polynomial_2",
            "noise_2",
            "degre_2",
        ):
            widget = getattr(self.ui, widget_name, None)
            if widget is not None:
                self._attach_sync_to_widget(widget)

    def connect_visu_tab_signals(self):
        """Visualization/graph watchers."""
        #For sync settings
        for widget_name in (
            "cb_grid",
        ):
            widget = getattr(self.ui, widget_name, None)
            if widget is not None:
                self._attach_sync_to_widget(widget)

    def connect_file_converter(self):
        """To launch the FilConverter Tool"""
        self.file_converter = FileConverter(self.app_settings.qsettings, parent=self.ui)
            
        #Create button
        icon_path = os.path.join(ICON_DIR, "FileConvert.png")
        action = QAction(self.ui.toolBar, icon = QIcon(icon_path))
        action.setToolTip("Open File Converter Tool")
        action.setStatusTip("File Converter")

        #Connect
        action.triggered.connect(self.file_converter.launch)
        
        # Add the action to the toolbar
        spacer = QWidget()
        spacer.setFixedWidth(50)
        self.ui.toolBar.addWidget(spacer)        
        self.ui.toolBar.addSeparator()
        self.ui.toolBar.addAction(action)
        
    def connect_settings_panel(self):
        """To launch the FilConverter Tool"""
        #Create "Settings Panel" button
        icon_path = os.path.join(ICON_DIR, "settings.png")
        action = QAction(self.ui.toolBar, icon = QIcon(icon_path))
        action.setToolTip("Settings")
        action.setStatusTip("Settings")

        #Connect
        action.triggered.connect(self.setting_panel.launch)
        
        # Add the action to the toolbar
        spacer = QWidget()
        spacer.setFixedWidth(50)
        self.ui.toolBar.addWidget(spacer)        
        self.ui.toolBar.addSeparator()
        self.ui.toolBar.addAction(action)

    def connect_all_signals(self):
        """Convenience to hook everything in one call."""
        self.connect_main_tab_signals()
        self.connect_maps_tab_signals()
        self.connect_spectra_tab_signals()
        self.connect_visu_tab_signals()
        self.connect_file_converter()
        self.connect_settings_panel()
