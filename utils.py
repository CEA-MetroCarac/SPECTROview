"""
Module contains all utilities functions and common functions
"""
import time
import markdown
import os
from copy import deepcopy

try:
    import win32clipboard
except:
    pass
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PySide6.QtWidgets import QMessageBox, QDialog, QTableWidget, \
    QTableWidgetItem, QVBoxLayout, QHBoxLayout, QTextBrowser, QLabel, \
    QLineEdit, \
    QPushButton, \
    QComboBox, QCheckBox
from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QPalette, QColor, QTextCursor

from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

DIRNAME = os.path.dirname(__file__)
RELPATH = os.path.join(DIRNAME, "resources")

PEAK_MODELS = ["Lorentzian", "Gaussian", "PseudoVoigt", "GaussianAsym",
               "LorentzianAsym"]
FIT_PARAMS = {'method': 'leastsq', 'fit_negative': False, 'fit_outliers': False,
              'max_ite': 200, 'coef_noise': 1, 'xtol': 1.e-4, 'ncpus': 'auto'}
FIT_METHODS = {'Leastsq': 'leastsq', 'Least_squares': 'least_squares',
               'Nelder-Mead': 'nelder', 'SLSQP': 'slsqp'}
NCPUS = ['auto', '1', '2', '3', '4', '6', '8', '10', '12', '14', '16', '20',
         '24', '28', '32']


def plot_graph(ax, dfr, x, y, z, style, xmin, xmax, ymin, ymax, title, x_text,
               y_text, xlabel_rot):
    """Plot graph """

    ax.clear()
    if style == "box plot":
        sns.boxplot(data=dfr, x=x, y=y, hue=z, dodge=True, ax=ax)
    elif style == "point plot":
        sns.pointplot(data=dfr, x=x, y=y, hue=z, linestyle='none',
                      dodge=True, capsize=0.00, ax=ax)
    elif style == "scatter plot":
        sns.scatterplot(data=dfr, x=x, y=y, hue=z, s=100, ax=ax)
    elif style == "bar plot":
        sns.barplot(data=dfr, x=x, y=y, hue=z, errorbar='sd', ax=ax)

    if xmin and xmax:
        ax.set_xlim(float(xmin), float(xmax))
    if ymin and ymax:
        ax.set_ylim(float(ymin), float(ymax))

    xlabel = x if not x_text else x_text
    ylabel = y if not y_text else y_text
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc='upper right')
    plt.setp(ax.get_xticklabels(), rotation=xlabel_rot, ha="right",
             rotation_mode="anchor")
    return ax


def reinit_spectrum(fnames, spectra_fs):
    """Reinitilize a FITSPY spectrum object"""
    for fname in fnames:
        spectrum, _ = spectra_fs.get_objects(fname)
        spectrum.range_min = None
        spectrum.range_max = None
        spectrum.x = spectrum.x0.copy()
        spectrum.y = spectrum.y0.copy()
        spectrum.norm_mode = None
        spectrum.result_fit = lambda: None
        spectrum.remove_models()
        spectrum.baseline.points = [[], []]
        spectrum.baseline.is_subtracted = False


def clear_layout(layout):
    """Clear everything within a given Qlayout"""
    if layout is not None:
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if isinstance(item.widget(),
                          (FigureCanvas, NavigationToolbar2QT)):
                widget = item.widget()
                layout.removeWidget(widget)
                widget.close()


def translate_param(model_fs, param):
    """Translate parameter names to plot title"""
    peak_labels = model_fs["peak_labels"]
    param_unit_mapping = {"ampli": "Intensity", "fwhm": "FWHM",
                          "fwhm_l": "FWHM_left", "fwhm_r": "FWHM_right",
                          "alpha": "L/G ratio",
                          "x0": "Position"}
    if "_" in param:
        prefix, param = param.split("_", 1)
        if param in param_unit_mapping:
            if param == "alpha":
                unit = ""  # Set unit to empty string for "alpha"
            else:
                unit = "(a.u)" if param == "ampli" else "(cm⁻¹)"
            label = param_unit_mapping[param]
            # Convert prefix to peak_label
            peak_index = int(prefix[1:]) - 1
            if 0 <= peak_index < len(peak_labels):
                peak_label = peak_labels[peak_index]
                return f"{label} of peak {peak_label} {unit}"
    return param


def quadrant(row):
    """Define 4 quadrant of a wafer"""
    if row['X'] < 0 and row['Y'] < 0:
        return 'Q1'
    elif row['X'] < 0 and row['Y'] > 0:
        return 'Q2'
    elif row['X'] > 0 and row['Y'] > 0:
        return 'Q3'
    elif row['X'] > 0 and row['Y'] < 0:
        return 'Q4'
    else:
        return np.nan


def zone(row, diameter):
    """Define 3 zones (Center, Mid-Rayon, Edge) based on X and Y coordinates."""
    rad = diameter / 2
    x = row['X']
    y = row['Y']
    distance_to_center = np.sqrt(x ** 2 + y ** 2)
    if distance_to_center <= rad * 0.35:
        return 'Center'
    elif distance_to_center > rad * 0.35 and distance_to_center < rad * 0.8:
        return 'Mid-Rayon'
    elif distance_to_center >= 0.8 * rad:
        return 'Edge'
    else:
        return np.nan


def copy_fig_to_clb(canvas):
    """Function to copy canvas figure to clipboard"""
    if canvas:
        figure = canvas.figure
        with BytesIO() as buf:
            figure.savefig(buf, format='png', dpi=400)
            data = buf.getvalue()
        format_id = win32clipboard.RegisterClipboardFormat('PNG')
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(format_id, data)
        win32clipboard.CloseClipboard()
    else:
        QMessageBox.critical(None, "Error", "No plot to copy.")


def show_alert(message):
    """Show alert"""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Alert")
    msg_box.setText(message)
    msg_box.exec_()


def view_df(tabWidget, df):
    """View selected dataframe"""
    df_viewer = QDialog(tabWidget.parent())
    df_viewer.setWindowTitle("DataFrame Viewer")
    df_viewer.setWindowFlags(df_viewer.windowFlags() & ~Qt.WindowStaysOnTopHint)
    # Create a QTableWidget and populate it with data from the DataFrame
    table_widget = QTableWidget(df_viewer)
    table_widget.setColumnCount(df.shape[1])
    table_widget.setRowCount(df.shape[0])
    table_widget.setHorizontalHeaderLabels(df.columns)
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            item = QTableWidgetItem(str(df.iat[row, col]))
            table_widget.setItem(row, col, item)
    table_widget.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
    layout = QVBoxLayout(df_viewer)
    layout.addWidget(table_widget)
    df_viewer.show()


def display_df_in_table(table_widget, df_results):
    """Display pandas DataFrame in QTableWidget in GUI"""
    table_widget.setRowCount(df_results.shape[0])
    table_widget.setColumnCount(df_results.shape[1])
    table_widget.setHorizontalHeaderLabels(df_results.columns)
    for row in range(df_results.shape[0]):
        for col in range(df_results.shape[1]):
            item = QTableWidgetItem(str(df_results.iat[row, col]))
            table_widget.setItem(row, col, item)
    table_widget.resizeColumnsToContents()


def view_text(ui, title, text):
    """ Create a QTextBrowser to display a text content"""
    report_viewer = QDialog(ui)
    report_viewer.setWindowTitle(title)
    report_viewer.setGeometry(100, 100, 800, 600)
    text_browser = QTextBrowser(report_viewer)
    text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    text_browser.setOpenExternalLinks(True)
    text_browser.setPlainText(text)
    text_browser.moveCursor(QTextCursor.Start)
    layout = QVBoxLayout(report_viewer)
    layout.addWidget(text_browser)
    report_viewer.show()


def view_markdown(ui, title, fname, x, y):
    """To convert MD file to html format and display them in GUI"""
    with open(fname, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    html_content = markdown.markdown(markdown_content)
    DIRNAME = os.path.dirname(__file__)
    html_content = html_content.replace('src="',
                                        f'src="'
                                        f'{os.path.join(DIRNAME, "resources/")}')
    about_dialog = QDialog(ui)
    about_dialog.setWindowTitle(title)
    about_dialog.resize(x, y)
    text_browser = QTextBrowser(about_dialog)
    text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    text_browser.setOpenExternalLinks(True)
    text_browser.setHtml(html_content)
    layout = QVBoxLayout(about_dialog)
    layout.addWidget(text_browser)
    about_dialog.setLayout(layout)
    about_dialog.show()


def dark_palette():
    """Palette color for dark mode of the appli's GUI"""
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(70, 70, 70))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base,
                          QColor(65, 65, 65))  # QlineEdit Listbox bg
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
    dark_palette.setColor(QPalette.PlaceholderText, QColor(140, 140, 140))
    return dark_palette


def light_palette():
    """Palette color for light mode of the appli's GUI"""
    light_palette = QPalette()
    light_palette.setColor(QPalette.Window, QColor(225, 225, 225))
    light_palette.setColor(QPalette.WindowText, Qt.black)
    light_palette.setColor(QPalette.Base, QColor(215, 215, 215))
    light_palette.setColor(QPalette.AlternateBase, QColor(230, 230, 230))
    light_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    light_palette.setColor(QPalette.ToolTipText, Qt.black)
    light_palette.setColor(QPalette.Text, Qt.black)
    light_palette.setColor(QPalette.Button, QColor(230, 230, 230))
    light_palette.setColor(QPalette.ButtonText, Qt.black)
    light_palette.setColor(QPalette.BrightText, Qt.red)
    light_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    light_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    light_palette.setColor(QPalette.HighlightedText, Qt.black)
    light_palette.setColor(QPalette.PlaceholderText, QColor(150, 150, 150))
    return light_palette


class FitThread(QThread):
    """ Class to perform fitting in a separate Thread to avoid GUI
    freezing/lagging"""
    fit_progress_changed = Signal(int)  # To update progress bar
    fit_progress = Signal(int, float)  # TO display number and elapsed time
    fit_completed = Signal()

    def __init__(self, spectra_fs, model_fs, fnames):
        super().__init__()
        self.spectra_fs = spectra_fs
        self.model_fs = model_fs
        self.fnames = fnames

    def run(self):
        start_time = time.time()  # Record start time
        num = 0
        for index, fname in enumerate(self.fnames):
            progress = int((index + 1) / len(self.fnames) * 100)
            self.fit_progress_changed.emit(progress)
            fit_model = deepcopy(self.model_fs)
            self.spectra_fs.apply_model(fit_model, fnames=[fname],
                                        show_progressbar=None)
            num += 1
            elapsed_time = time.time() - start_time
            self.fit_progress.emit(num, elapsed_time)
        self.fit_progress_changed.emit(100)
        self.fit_completed.emit()


class ShowParameters:
    def __init__(self, main_layout, sel_spectrum, cb_limits, cb_expr, update):
        self.main_layout = main_layout
        self.sel_spectrum = sel_spectrum
        self.cb_limits = cb_limits
        self.cb_expr = cb_expr
        self.update = update

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def show_peak_table(self, main_layout, sel_spectrum, cb_limits, cb_expr):
        """ To show all fitted parameters in GUI"""
        self.clear_layout(main_layout)

        header_labels = ["Delete", "Label", "Model"]
        param_hint_order = ['x0', 'fwhm', 'ampli', 'alpha', 'fwhm_l', 'fwhm_r']

        # Create and add headers to list
        for param_hint_key in param_hint_order:
            if any(param_hint_key in peak_model.param_hints for peak_model in
                   sel_spectrum.peak_models):
                header_labels.append(param_hint_key.title())
                header_labels.append(f"fix {param_hint_key.title()}")
                if cb_limits.isChecked():
                    header_labels.append(f"min {param_hint_key.title()}")
                    header_labels.append(f"max {param_hint_key.title()}")
                if cb_expr.isChecked():
                    header_labels.append(f"expression {param_hint_key.title()}")

        # Create vertical layouts for each column type
        delete_layout = QVBoxLayout()
        label_layout = QVBoxLayout()
        model_layout = QVBoxLayout()
        param_hint_layouts = {param_hint: {var: QVBoxLayout() for var in
                                           ['value', 'min', 'max', 'expr',
                                            'vary']} for
                              param_hint in param_hint_order}

        # Add header labels to each layout
        for header_label in header_labels:
            label = QLabel(header_label)
            label.setAlignment(Qt.AlignCenter)
            if header_label == "Delete":
                delete_layout.addWidget(label)
            elif header_label == "Label":
                label_layout.addWidget(label)
            elif header_label == "Model":
                model_layout.addWidget(label)
            elif header_label.startswith("fix"):
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['vary'].addWidget(label)
            elif "min" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['min'].addWidget(label)
            elif "max" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['max'].addWidget(label)
            elif "expression" in header_label:
                param_hint_key = header_label.split()[1].lower()
                param_hint_layouts[param_hint_key]['expr'].addWidget(label)
            else:
                param_hint_key = header_label.lower()
                param_hint_layouts[param_hint_key]['value'].addWidget(label)

        for i, peak_model in enumerate(sel_spectrum.peak_models):
            # Button to delete peak_model
            delete = QPushButton(peak_model.prefix)
            delete.setFixedWidth(50)
            delete.clicked.connect(
                lambda idx=i, spectrum=sel_spectrum: self.delete_peak_model(
                    spectrum, idx))
            delete_layout.addWidget(delete)

            # Peak_label
            label = QLineEdit(sel_spectrum.peak_labels[i])
            label.setFixedWidth(80)
            label.textChanged.connect(
                lambda text, idx=i,
                       spectrum=sel_spectrum: self.update_peak_label(spectrum,
                                                                     idx, text))
            label_layout.addWidget(label)

            # Peak model : Lorentizan, Gaussien, etc...
            model = QComboBox()
            model.addItems(PEAK_MODELS)
            current_model_index = PEAK_MODELS.index(
                peak_model.name2) if peak_model.name2 in PEAK_MODELS else 0
            model.setCurrentIndex(current_model_index)
            model.setFixedWidth(120)
            model.currentIndexChanged.connect(
                lambda index, spectrum=sel_spectrum, idx=i,
                       combo=model: self.update_model_name(spectrum, index, idx,
                                                           combo.currentText()))
            model_layout.addWidget(model)

            # variables of peak_model
            param_hints = peak_model.param_hints
            for param_hint_key in param_hint_order:
                if param_hint_key in param_hints:
                    param_hint_value = param_hints[param_hint_key]

                    # 4.1 VALUE
                    value_val = round(param_hint_value.get('value', 0.0), 2)
                    value = QLineEdit(str(value_val))
                    value.setFixedWidth(70)
                    value.setFixedHeight(24)
                    value.setAlignment(Qt.AlignRight)
                    value.textChanged.connect(
                        lambda text, pm=peak_model,
                               key=param_hint_key: self.update_param_hint_value(
                            pm, key, text))
                    param_hint_layouts[param_hint_key]['value'].addWidget(value)

                    # 4.2 FIXED or NOT
                    vary = QCheckBox()
                    vary.setChecked(not param_hint_value.get('vary', False))
                    vary.setFixedHeight(24)
                    vary.stateChanged.connect(
                        lambda state, pm=peak_model,
                               key=param_hint_key: self.update_param_hint_vary(
                            pm, key,
                            not state))
                    param_hint_layouts[param_hint_key]['vary'].addWidget(vary)

                    # 4.3 MIN MAX
                    if cb_limits.isChecked():
                        min_val = round(param_hint_value.get('min', 0.0), 2)
                        min_lineedit = QLineEdit(str(min_val))
                        min_lineedit.setFixedWidth(70)
                        min_lineedit.setFixedHeight(24)
                        min_lineedit.setAlignment(Qt.AlignRight)
                        min_lineedit.textChanged.connect(
                            lambda text, pm=peak_model,
                                   key=param_hint_key:
                            self.update_param_hint_min(
                                pm, key, text))
                        param_hint_layouts[param_hint_key]['min'].addWidget(
                            min_lineedit)

                        max_val = round(param_hint_value.get('max', 0.0), 2)
                        max_lineedit = QLineEdit(str(max_val))
                        max_lineedit.setFixedWidth(70)
                        max_lineedit.setFixedHeight(24)
                        max_lineedit.setAlignment(Qt.AlignRight)
                        max_lineedit.textChanged.connect(
                            lambda text, pm=peak_model,
                                   key=param_hint_key:
                            self.update_param_hint_max(
                                pm, key, text))
                        param_hint_layouts[param_hint_key]['max'].addWidget(
                            max_lineedit)

                    # 4.4 EXPRESSION
                    if cb_expr.isChecked():
                        expr_val = str(param_hint_value.get('expr', ''))
                        expr = QLineEdit(expr_val)
                        expr.setFixedWidth(150)
                        expr.setFixedHeight(
                            24)  # Set a fixed height for QLineEdit
                        expr.setAlignment(Qt.AlignRight)
                        expr.textChanged.connect(
                            lambda text, pm=peak_model,
                                   key=param_hint_key:
                            self.update_param_hint_expr(
                                pm, key, text))
                        param_hint_layouts[param_hint_key]['expr'].addWidget(
                            expr)
                else:
                    # Add empty labels for alignment
                    empty_label = QLabel()
                    empty_label.setFixedHeight(24)
                    param_hint_layouts[param_hint_key]['value'].addWidget(
                        empty_label)
                    param_hint_layouts[param_hint_key]['vary'].addWidget(
                        empty_label)
                    if cb_limits.isChecked():
                        param_hint_layouts[param_hint_key]['min'].addWidget(
                            empty_label)
                        param_hint_layouts[param_hint_key]['max'].addWidget(
                            empty_label)
                    if cb_expr.isChecked():
                        param_hint_layouts[param_hint_key]['expr'].addWidget(
                            empty_label)

        # Add vertical layouts to main layout
        main_layout.addLayout(delete_layout)
        main_layout.addLayout(label_layout)
        main_layout.addLayout(model_layout)

        for param_hint_key, param_hint_layout in param_hint_layouts.items():
            for var_layout in param_hint_layout.values():
                main_layout.addLayout(var_layout)

    def delete_peak_model(self, spectrum, idx):
        """"To delete a peak model"""
        del spectrum.peak_models[idx]
        del spectrum.peak_labels[idx]
        self.update()

    def update_peak_label(self, spectrum, idx, text):
        spectrum.peak_labels[idx] = text

    def update_param_hint_value(self, pm, key, text):
        pm.param_hints[key]['value'] = float(text)

    def update_param_hint_min(self, pm, key, text):
        pm.param_hints[key]['min'] = float(text)

    def update_param_hint_max(self, pm, key, text):
        pm.param_hints[key]['max'] = float(text)

    def update_param_hint_vary(self, pm, key, state):
        pm.param_hints[key]['vary'] = state

    def update_param_hint_expr(self, pm, key, text):
        pm.param_hints[key]['expr'] = text
