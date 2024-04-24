# main.py module
import sys
import os
import datetime

from PySide6.QtWidgets import QApplication, QListWidget, QComboBox, QMessageBox
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QSettings
from PySide6.QtGui import QDoubleValidator, QIcon

from utils import dark_palette, light_palette, view_markdown, show_alert, \
    PEAK_MODELS

from ui import resources_new
from dataframe import Dataframe
from visualization import Vizualisation
from maps import Maps
from spectrums import Spectrums
from workspace import SaveLoadWorkspace

DIRNAME = os.path.dirname(__file__)
UI_FILE = os.path.join(DIRNAME, "ui", "gui.ui")
ICON_APPLI = os.path.join(DIRNAME, "ui", "iconpack", "icon3.png")
HELP_DFQUERY = os.path.join(DIRNAME, "resources", "pandas_df_query.md")
ABOUT = os.path.join(DIRNAME, "resources", "about.md")


class Main:
    def __init__(self):
        # Load the UI file
        loader = QUiLoader()
        ui_file = QFile(UI_FILE)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file)
        ui_file.close()

        # Initialize QSettings
        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.settings = QSettings("CEA-Leti", "SPECTROview")

        if self.settings.value("mode") == "light":
            self.toggle_light_mode()
        else:
            self.toggle_dark_mode()  # Toggle to dark mode by default
        self.ui.actionDarkMode.triggered.connect(self.toggle_dark_mode)
        self.ui.actionLightMode.triggered.connect(self.toggle_light_mode)

        # Create an instance of Dataframe and pass the self.ui object
        self.dataframe = Dataframe(self.settings, self.ui)
        self.visualization = Vizualisation(self.settings, self.ui,
                                           self.dataframe)
        self.workspace = SaveLoadWorkspace(self.settings, self.ui,
                                           self.dataframe,
                                           self.visualization)
        self.maps = Maps(self.settings, self.ui, self.dataframe)
        self.spectrums = Spectrums(self.settings, self.ui, self.dataframe)

        # DATAFRAME
        self.ui.btn_open_df.clicked.connect(
            lambda event: self.dataframe.action_open_df())

        self.listbox_dfs = self.ui.findChild(QListWidget, 'listbox_dfs')
        # Connect the itemClicked signal to select_dataframe
        self.listbox_dfs.itemClicked.connect(self.dataframe.select_df)

        self.combo_xaxis = self.ui.findChild(QComboBox, 'combo_xaxis')
        self.combo_yaxis = self.ui.findChild(QComboBox, 'combo_yaxis')
        self.combo_hue = self.ui.findChild(QComboBox, 'combo_hue')
        self.combo_hue_2 = self.ui.findChild(QComboBox, 'combo_hue_2')

        # Save/load workspace
        self.ui.btn_clearworkspace.clicked.connect(
            self.visualization.clear_workspace)
        self.ui.btn_save_work.clicked.connect(
            lambda event: self.workspace.save_workspace())
        self.ui.btn_open_work.clicked.connect(
            lambda event: self.workspace.load_workspace())
        self.ui.btn_plot_recipe.clicked.connect(
            lambda event: self.workspace.load_recipe())

        # Populate df columns to comboboxes
        self.combo_xaxis.activated.connect(
            self.dataframe.set_selected_x_column)
        self.combo_yaxis.activated.connect(
            self.dataframe.set_selected_y_column)
        self.combo_hue.activated.connect(
            self.dataframe.set_selected_hue_column)
        self.combo_hue_2.activated.connect(
            self.dataframe.set_selected_hue_column)
        # Bidirectional synchronization between two comboboax
        self.ui.combo_hue.currentIndexChanged.connect(self.update_combo_hue_2)
        self.ui.combo_hue_2.currentIndexChanged.connect(self.update_combo_hue)

        # REMOVE, SAVE DataFrames ###
        self.ui.btn_view_df.clicked.connect(self.dataframe.view_df)
        self.ui.btn_remove_df.clicked.connect(self.dataframe.remove_df)
        self.ui.btn_save_all_df.clicked.connect(
            self.dataframe.save_all_df_handler)

        # Concantenate dataframes in listbox to a global dataframe
        self.ui.merge_dfs.clicked.connect(self.dataframe.concat_dfs)

        # df FILTERS:
        self.ui.ent_filter_query.returnPressed.connect(
            self.dataframe.add_filter)
        # ADD filter
        self.ui.btn_add_filter.clicked.connect(self.dataframe.add_filter)
        # REMOVE filter
        self.ui.btn_remove_filters.clicked.connect(
            self.dataframe.remove_selected_filters)
        # APPLY filter
        self.ui.btn_apply_filters.clicked.connect(
            self.dataframe.apply_filters)

        # PLOT STYLING
        self.ui.combo_plot_style.addItems(self.visualization.plot_styles)

        self.ui.combo_plot_style.currentIndexChanged.connect(
            self.visualization.set_selected_plot_style)

        self.ui.combo_palette_color.addItems(self.visualization.palette_colors)
        self.ui.combo_palette_color.currentIndexChanged.connect(
            self.visualization.set_selected_palette_colors)

        self.ui.btn_add_a_plot.clicked.connect(self.visualization.add_a_plot)
        self.ui.btn_add_wafer_plot.clicked.connect(
            self.visualization.add_wafer_plots)

        # Set default values for plot width & height
        self.ui.ent_plotwidth.setText("470")
        self.ui.ent_plotheight.setText("400")
        self.ui.ent_plot_display_dpi.setText("80")
        self.ui.ent_plot_save_dpi.setText("300")
        self.ui.ent_xlabel_rot.setText("0")  # x_label rotation
        self.ui.lineEdit_2.setText("1")  # data point transparency

        self.ui.checkBox.setChecked(True)  # include stats values in wafer plot
        self.ui.checkBox_2.setChecked(False)  # legend inside or outside
        self.ui.checkBox_3.setChecked(False)  # Grid

        self.ui.btn_clear_all_entry.clicked.connect(
            self.visualization.clear_entries)

        # Save fig function:
        self.ui.btn_save_all_figs.clicked.connect(
            self.visualization.save_all_figs)

        # Define limits of axis, accept only floats numbers (not text)
        validator = QDoubleValidator()
        self.ui.ent_xmax.setValidator(validator)
        self.ui.ent_ymax.setValidator(validator)
        self.ui.ent_xmin.setValidator(validator)
        self.ui.ent_ymin.setValidator(validator)
        self.ui.ent_colorscale_max.setValidator(validator)
        self.ui.ent_colorscale_min.setValidator(validator)
        self.ui.ent_xlabel_rot.setValidator(validator)
        self.ui.lineEdit_2.setValidator(validator)

        # Set default number of plot per row
        self.ui.spinBox_plot_per_row.setValue(3)

        # Help : document about pandas_df_query
        self.ui.actionHelps.triggered.connect(self.open_doc_df_query)

        # About dialog
        self.ui.actionAbout.triggered.connect(self.show_about)
        # Toggle to switch Dark/light mode
        # self.ui.radio_darkmode.clicked.connect(self.change_style)

        ########################################################
        ############## GUI for Wafer Processing tab #############
        ########################################################

        self.ui.btn_open_wafers.clicked.connect(self.maps.open_data)

        self.ui.btn_remove_wafer.clicked.connect(self.maps.remove_wafer)

        self.ui.btn_copy_fig.clicked.connect(self.maps.copy_fig)
        self.ui.btn_copy_fig_wafaer.clicked.connect(self.maps.copy_fig_wafer)
        self.ui.btn_copy_fig_graph.clicked.connect(self.maps.copy_fig_graph)

        self.ui.btn_sel_all.clicked.connect(self.maps.select_all_spectra)
        self.ui.btn_sel_verti.clicked.connect(self.maps.select_verti)
        self.ui.btn_sel_horiz.clicked.connect(self.maps.select_horiz)

        self.ui.btn_load_model.clicked.connect(self.maps.open_fit_model)
        self.ui.btn_fit.clicked.connect(self.maps.fit_fnc_handler)
        self.ui.btn_init.clicked.connect(self.maps.reinit_fnc_handler)
        self.ui.btn_collect_results.clicked.connect(self.maps.collect_results)
        self.ui.btn_view_df_2.clicked.connect(self.maps.view_fit_results_df)
        self.ui.btn_show_stats.clicked.connect(self.maps.view_stats)
        self.ui.btn_save_fit_results.clicked.connect(
            self.maps.save_fit_results)
        self.ui.btn_view_wafer.clicked.connect(self.maps.view_wafer_data)

        self.ui.btn_plot_wafer.clicked.connect(self.maps.plot3)
        self.ui.btn_plot_graph.clicked.connect(self.maps.plot4)
        self.ui.cbb_color_pallete.addItems(self.visualization.palette_colors)
        self.ui.btn_open_fitspy.clicked.connect(self.maps.fitspy_launcher)
        self.ui.btn_cosmis_ray.clicked.connect(self.maps.cosmis_ray_detection)
        self.ui.btn_open_fit_results.clicked.connect(
            self.maps.load_fit_results)

        self.ui.cbb_plot_style.addItems(self.maps.plot_styles)
        self.ui.btn_sw.clicked.connect(self.maps.save_work)
        self.ui.btn_lw.clicked.connect(self.maps.load_work)
        self.ui.btn_split_fname_2.clicked.connect(self.maps.split_fname)
        self.ui.btn_add_col_2.clicked.connect(self.maps.add_column)

        self.ui.range_max.returnPressed.connect(self.maps.set_x_range)
        self.ui.range_min.returnPressed.connect(self.maps.set_x_range)
        self.ui.range_apply.clicked.connect(self.maps.set_x_range_handler)
        self.ui.sub_baseline.clicked.connect(self.maps.subtract_baseline)
        self.ui.btn_fit_2.clicked.connect(self.maps.apply_fit_model_handler)
        self.ui.save_model.clicked.connect(self.maps.save_fit_model)
        self.ui.clear_peaks.clicked.connect(self.maps.clear_all_peaks)
        self.ui.cbb_fit_models.addItems(PEAK_MODELS)
        # self.ui.del_bl_points.clicked.connect(
        # self.maps.delete_baseline_points)
        self.ui.btn_undo_baseline.clicked.connect(self.maps.set_x_range)

        ########################################################
        ############## GUI for Spectrums Processing tab #############
        ########################################################
        self.ui.btn_open_spectrums.clicked.connect(self.spectrums.open_data)

        self.ui.btn_load_model_3.clicked.connect(self.spectrums.open_model)
        self.ui.btn_fit_3.clicked.connect(self.spectrums.fit_fnc_handler)
        self.ui.btn_open_fitspy_3.clicked.connect(
            self.spectrums.fitspy_launcher)
        self.ui.btn_cosmis_ray_3.clicked.connect(
            self.spectrums.cosmis_ray_detection)
        self.ui.btn_init_3.clicked.connect(self.spectrums.reinit_fnc_handler)
        self.ui.btn_show_stats_3.clicked.connect(self.spectrums.view_stats)
        self.ui.btn_sel_all_3.clicked.connect(self.spectrums.select_all_spectra)
        self.ui.btn_remove_spectrum.clicked.connect(
            self.spectrums.remove_spectrum)
        self.ui.btn_collect_results_3.clicked.connect(
            self.spectrums.collect_results)
        self.ui.btn_view_df_5.clicked.connect(
            self.spectrums.view_fit_results_df)
        self.ui.btn_save_fit_results_3.clicked.connect(
            self.spectrums.save_fit_results)
        self.ui.btn_open_fit_results_3.clicked.connect(
            self.spectrums.load_fit_results)

        self.ui.btn_sw_3.clicked.connect(self.spectrums.save_work)
        self.ui.btn_lw_3.clicked.connect(self.spectrums.load_work)
        self.ui.cbb_plot_style_3.addItems(self.spectrums.plot_styles)
        self.ui.cbb_plot_style_7.addItems(self.spectrums.plot_styles)
        self.ui.btn_plot_graph_3.clicked.connect(self.spectrums.plot2)
        self.ui.btn_plot_graph_7.clicked.connect(self.spectrums.plot3)

        self.ui.btn_split_fname.clicked.connect(self.spectrums.split_fname)
        self.ui.btn_add_col.clicked.connect(self.spectrums.add_column)

        self.ui.btn_add_filter_2.clicked.connect(self.spectrums.add_filter)
        self.ui.ent_filter_query_2.returnPressed.connect(
            self.spectrums.add_filter)
        self.ui.btn_apply_filters_2.clicked.connect(
            self.spectrums.apply_filters)
        self.ui.btn_remove_filters_2.clicked.connect(
            self.spectrums.remove_filter)

        self.ui.btn_copy_fig_3.clicked.connect(self.spectrums.copy_fig)
        self.ui.btn_copy2_3.clicked.connect(self.spectrums.copy_fig_graph1)
        self.ui.btn_copy2_7.clicked.connect(self.spectrums.copy_fig_graph2)

    def toggle_dark_mode(self):
        self.ui.setPalette(dark_palette())
        self.settings.setValue("mode", "dark")  # Save to settings

    def toggle_light_mode(self):
        self.ui.setPalette(light_palette())
        self.settings.setValue("mode", "light")  # Save to settings

    def update_combo_hue(self, index):
        selected_text = self.ui.combo_hue_2.itemText(index)
        self.ui.combo_hue.setCurrentIndex(
            self.ui.combo_hue.findText(selected_text))

    def update_combo_hue_2(self, index):
        selected_text = self.ui.combo_hue.itemText(index)
        self.ui.combo_hue_2.setCurrentIndex(
            self.ui.combo_hue_2.findText(selected_text))

    def open_doc_df_query(self):
        """Open doc detail about query function of pandas dataframe"""
        title = "Data filtering"
        view_markdown(self.ui, title, HELP_DFQUERY, 550, 650)

    def show_about(self):
        """Show about dialog """
        view_markdown(self.ui, "About", ABOUT, 500, 300)


# expiration_date = datetime.datetime(2024, 10, 1)
#
# def launcher():
#     # Check if the current date is past the expiration date
#     if datetime.datetime.now() > expiration_date:
#         text = f"This version of the application has expired on " \
#                f"{expiration_date}, so can not be used anymore. Please " \
#                f"contact the developer for an " \
#                f"updated version"
#         # If expired, disable the central widget
#         app = QApplication(sys.argv)
#         app.setWindowIcon(QIcon(ICON_APPLI))
#         window = Main()
#         window.ui.centralwidget.setEnabled(False)
#         app.setStyle("Fusion")
#         window.ui.show()
#         show_alert(text)
#         sys.exit(app.exec())
#
#     # If not expired, continue launching the application as usual
#     app = QApplication(sys.argv)
#     app.setWindowIcon(QIcon(ICON_APPLI))
#     window = Main()
#     window.ui.centralwidget.setEnabled(True)
#     app.setStyle("Fusion")
#     window.ui.show()
#     sys.exit(app.exec())
# if __name__ == "__main__":
#     launcher()

def launcher2(file_paths=None, fname_json=None):
    app = QApplication()
    app.setWindowIcon(QIcon(ICON_APPLI))
    window = Main()
    app.setStyle("Fusion")
    if file_paths is not None:
        window.maps.open_data(file_paths=file_paths)
    if fname_json is not None:
        window.maps.open_fit_model(fname_json=fname_json)
    window.ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    DIRNAME = os.path.dirname(__file__)
    DATA = os.path.join(DIRNAME, "data_test")
    DATA_MAPS = os.path.join(DATA, "RAW 2Dmaps")
    fname1 = os.path.join(DATA_MAPS, 'D23S2204.2_09.csv')
    fname2 = os.path.join(DATA_MAPS, 'D23S2204.2_19.csv')
    fname3 = os.path.join(DATA_MAPS, 'D23S2204.2_25.csv')
    fname_json1 = os.path.join(DATA_MAPS,
                               'MoS2_325-490_8cm-shifted_simple-labels.json')

    launcher2([fname1, fname2, fname3], fname_json1)
