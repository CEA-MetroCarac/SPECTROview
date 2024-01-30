# main.py module
import sys
import os
from PySide6.QtWidgets import QApplication, QDialog, QListWidget, QComboBox, \
    QTextBrowser, QVBoxLayout, QSplitter
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt, QFile
from PySide6.QtGui import QDoubleValidator, QTextCursor, QIcon, QPalette, QColor

from ui import resources_new

from callbacks_df import CallbacksDf
from callbacks_plot import CallbacksPlot
from callbacks_spectre import CallbacksSpectre
from workspace import SaveLoadWorkspace

DIRNAME = os.path.dirname(__file__)
UI_FILE = os.path.join(DIRNAME, "ui", "gui.ui")
ICON_APPLI = os.path.join(DIRNAME, "ui", "iconpack", "icon.png")
HELP_DFQUERY = os.path.join(DIRNAME, "resources", "pandas_df_query.md")


class MainWindow:
    def __init__(self):
        # Load the UI file
        loader = QUiLoader()
        ui_file = QFile(UI_FILE)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file)
        ui_file.close()

        # Create an instance of CallbacksDf and pass the self.ui object
        self.callbacks_df = CallbacksDf(self.ui)
        self.callbacks_spectre = CallbacksSpectre(self.ui)
        self.callbacks_plot = CallbacksPlot(self.ui, self.callbacks_df)
        self.workspace = SaveLoadWorkspace(self.ui, self.callbacks_df,
                                           self.callbacks_plot)
        ## Tét commit
        # DATAFRAME
        self.ui.btn_open_df.clicked.connect(
            lambda event: self.callbacks_df.action_open_df())

        self.listbox_dfs = self.ui.findChild(QListWidget, 'listbox_dfs')
        # Connect the itemClicked signal to select_dataframe
        self.listbox_dfs.itemClicked.connect(self.callbacks_df.select_df)

        self.combo_xaxis = self.ui.findChild(QComboBox, 'combo_xaxis')
        self.combo_yaxis = self.ui.findChild(QComboBox, 'combo_yaxis')
        self.combo_hue = self.ui.findChild(QComboBox, 'combo_hue')
        self.combo_hue_2 = self.ui.findChild(QComboBox, 'combo_hue_2')

        # Save/load workspace
        self.ui.btn_clearworkspace.clicked.connect(
            self.callbacks_plot.clear_workspace)
        self.ui.btn_save_work.clicked.connect(
            lambda event: self.workspace.save_workspace())
        self.ui.btn_open_work.clicked.connect(
            lambda event: self.workspace.load_workspace())
        self.ui.btn_plot_recipe.clicked.connect(
            lambda event: self.workspace.load_recipe())

        # Populate df columns to comboboxes
        self.combo_xaxis.activated.connect(
            self.callbacks_df.set_selected_x_column)
        self.combo_yaxis.activated.connect(
            self.callbacks_df.set_selected_y_column)
        self.combo_hue.activated.connect(
            self.callbacks_df.set_selected_hue_column)
        self.combo_hue_2.activated.connect(
            self.callbacks_df.set_selected_hue_column)
        # Bidirectional synchronization between two comboboax
        self.ui.combo_hue.currentIndexChanged.connect(self.update_combo_hue_2)
        self.ui.combo_hue_2.currentIndexChanged.connect(self.update_combo_hue)

        ### REMOVE, SAVE DataFrames   qsd ###
        self.ui.btn_view_df.clicked.connect(self.callbacks_df.view_df)
        self.ui.btn_remove_df.clicked.connect(self.callbacks_df.remove_df)
        # self.ui.btn_save_df.clicked.connect(self.callbacks_df.save_df)
        self.ui.btn_save_all_df.clicked.connect(
            self.callbacks_df.save_all_df_handler)

        # Concantenate dataframes in listbox to a global dataframe
        self.ui.merge_dfs.clicked.connect(self.callbacks_df.concat_dfs)

        # df FILTERS:
        self.ui.ent_filter_query.returnPressed.connect(
            self.callbacks_df.add_filter)
        # ADD filter
        self.ui.btn_add_filter.clicked.connect(self.callbacks_df.add_filter)
        # REMOVE filter
        self.ui.btn_remove_filters.clicked.connect(
            self.callbacks_df.remove_selected_filters)
        # APPLY filter
        self.ui.btn_apply_filters.clicked.connect(
            self.callbacks_df.apply_selected_filters)

        # PLOT STYLING
        self.ui.combo_plot_style.addItems(self.callbacks_plot.plot_styles)
        self.ui.combo_plot_style.currentIndexChanged.connect(
            self.callbacks_plot.set_selected_plot_style)

        self.ui.combo_palette_color.addItems(self.callbacks_plot.palette_colors)
        self.ui.combo_palette_color.currentIndexChanged.connect(
            self.callbacks_plot.set_selected_palette_colors)

        self.ui.btn_add_a_plot.clicked.connect(self.callbacks_plot.add_a_plot)
        self.ui.btn_add_wafer_plot.clicked.connect(
            self.callbacks_plot.add_wafer_plots)

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
            self.callbacks_plot.clear_entries)

        # Save fig function:
        self.ui.btn_save_all_figs.clicked.connect(
            self.callbacks_plot.save_all_figs)

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
        self.ui.actionHelps.triggered.connect(self.open_documentation)

        # About dialog
        self.ui.actionabout.triggered.connect(self.show_about_dialog)

        # Toggle to switch Dark/light mode
        self.ui.radio_darkmode.clicked.connect(self.change_style)

        self.darkmode = True
        dark_palette = self.dark_palette()
        self.ui.setPalette(dark_palette)

        ########################################################
        ############## GUI for Data Processing tab #############
        ########################################################

        self.ui.btn_open_wafers.clicked.connect(
            self.callbacks_spectre.open_csv)
        self.ui.btn_remove_wafer.clicked.connect(
            self.callbacks_spectre.remove_wafer)

        self.ui.btn_copy_fig.clicked.connect(
            self.callbacks_spectre.copy_fig)

        self.ui.btn_sel_all.clicked.connect(
            self.callbacks_spectre.select_all_spectra)
        self.ui.btn_sel_vertical.clicked.connect(
            self.callbacks_spectre.select_spectra_vertical)
        self.ui.btn_sel_horizontal.clicked.connect(
            self.callbacks_spectre.select_spectra_horizontal)

        self.ui.btn_load_model.clicked.connect(
            self.callbacks_spectre.open_model)
        self.ui.btn_fit.clicked.connect(
            self.callbacks_spectre.fitting_sel_spectrum)
        self.ui.btn_fit_all_wafers.clicked.connect(
            self.callbacks_spectre.fitting_all_wafer)

    def update_combo_hue(self, index):
        selected_text = self.ui.combo_hue_2.itemText(index)
        self.ui.combo_hue.setCurrentIndex(
            self.ui.combo_hue.findText(selected_text))

    def update_combo_hue_2(self, index):
        selected_text = self.ui.combo_hue.itemText(index)
        self.ui.combo_hue_2.setCurrentIndex(
            self.ui.combo_hue_2.findText(selected_text))

    def change_style(self):
        if not self.darkmode:
            self.darkmode = True
            dark_palette = self.dark_palette()
            self.ui.setPalette(dark_palette)
        else:
            self.darkmode = False
            light_palette = self.light_palette()
            self.ui.setPalette(light_palette)

    def dark_palette(self):
        # Get the dark color palette of the application
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(65, 65, 65))
        dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(64, 64, 64))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.white)
        # Placeholder text color
        dark_palette.setColor(QPalette.PlaceholderText, QColor(140, 140, 140))
        return dark_palette

    def light_palette(self):
        # Get the light color palette of the application
        light_palette = QPalette()
        # Light gray background
        light_palette.setColor(QPalette.Window, QColor(235, 235, 235))
        light_palette.setColor(QPalette.WindowText, Qt.black)
        # White background
        light_palette.setColor(QPalette.Base, QColor(239, 239, 239))
        # Light gray alternate background
        light_palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        light_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        light_palette.setColor(QPalette.ToolTipText, Qt.black)
        light_palette.setColor(QPalette.Text, Qt.black)
        # Light gray button color
        light_palette.setColor(QPalette.Button, QColor(251, 251, 251))
        light_palette.setColor(QPalette.ButtonText, Qt.black)
        light_palette.setColor(QPalette.BrightText, Qt.red)
        light_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        light_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        light_palette.setColor(QPalette.HighlightedText, Qt.black)

        light_palette.setColor(QPalette.PlaceholderText, QColor(150, 150, 150))
        # Light gray placeholder text color
        return light_palette

    def open_documentation(self):
        markdown_file_path = HELP_DFQUERY

        # Create a QDialog to display the Markdown content
        markdown_viewer = QDialog(self.ui.tabWidget)
        markdown_viewer.setWindowTitle("Markdown Viewer")
        markdown_viewer.setGeometry(100, 100, 800, 600)

        # Create a QTextBrowser to display the Markdown content
        text_browser = QTextBrowser(markdown_viewer)
        text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        text_browser.setOpenExternalLinks(
            True)  # Allow opening links in a web browser

        # Load and display the Markdown file
        with open(markdown_file_path, 'r', encoding='utf-8') as markdown_file:
            markdown_content = markdown_file.read()
            text_browser.setMarkdown(markdown_content)

        text_browser.moveCursor(QTextCursor.Start)  # Scroll to top of document

        layout = QVBoxLayout(markdown_viewer)
        layout.addWidget(text_browser)
        markdown_viewer.exec()  # Show the Markdown viewer dialog

    def show_about_dialog(self):
        about_text = """
        <h3>DaProViz</h3>
        <p>Tool for Data Processing and Visualization</p>
        <p>Version: 0.10 / Build 240129 </p>
        <p>Contact: Van-Hoan Le (van-hoan.le@cea.fr)</p>
        """
        about_dialog = QDialog(self.ui)
        about_dialog.setWindowTitle("About")

        text_browser = QTextBrowser(about_dialog)
        text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        text_browser.setOpenExternalLinks(
            True)  # Allow opening links in a web browser
        text_browser.setHtml(about_text)

        layout = QVBoxLayout(about_dialog)
        layout.addWidget(text_browser)

        about_dialog.exec()


# def launcher():
#     app = QApplication()
#     app.setWindowIcon(QIcon(ICON_APPLI))
#     window = MainWindow()
#     app.setStyle("Fusion")
#     window.ui.show()
#     sys.exit(app.exec())
#
#
# if __name__ == "__main__":
#     launcher()

def launcher2(file_paths=None, fname_json=None):
    app = QApplication()
    app.setWindowIcon(QIcon(ICON_APPLI))
    window = MainWindow()
    app.setStyle("Fusion")
    if file_paths is not None:
        window.callbacks_spectre.open_csv(file_paths=file_paths)

    if fname_json is not None:
        window.callbacks_spectre.open_model(fname_json=fname_json)

        # window.saveloadws.save_workspace(fname_json='toto.json')
    # window.saveloadws.load_workspace(fname_json='toto.json')
    window.ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    dirname = r"C:\Users\VL251876\PycharmProjects\workspace\projects\Raman" \
              r"\P_23_10_DaProViz\data_test\RAW spectrum"
    fname1 = os.path.join(dirname, 'D23S2204.2_05.csv')
    fname2 = os.path.join(dirname, 'D23S2204.2_10.csv')
    fname3 = os.path.join(dirname, 'D23S2204.2_19.csv')
    fname4 = os.path.join(dirname, 'D23S2204.2_07.csv')
    fname_json = os.path.join(dirname, 'MoS2_5peaks_Test.json')
    launcher2([fname1, fname2, fname4, fname3], fname_json)
