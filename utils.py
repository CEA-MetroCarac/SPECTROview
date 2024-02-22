"""
Module contains all utilities functions and common functions
"""
import os
import copy
import time

#import win32clipboard
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PySide6.QtWidgets import QMessageBox, QDialog, QTableWidget, \
    QTableWidgetItem, QVBoxLayout, QTextBrowser
from PySide6.QtCore import Qt, QFile, QObject, Signal, QThread
from PySide6.QtGui import QPalette, QColor, QTextCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


def plot_graph(dfr, x, y, z, style, xmin, xmax, ymin, ymax, title, x_text,
               y_text, xlabel_rot):
    """Plot graph """
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if style == "box plot":
        sns.boxplot(data=dfr, x=x, y=y, hue=z, dodge=True, ax=ax)
    elif style == "point plot":
        sns.pointplot(data=dfr, x=x, y=y, hue=z, linestyle='none',
                      dodge=True, capsize=0.00, ax=ax)
    elif style == "scatter plot":
        sns.scatterplot(data=dfr, x=x, y=y, hue=z, s=100, ax=ax)
    elif style == "bar plot":
        sns.barplot(data=dfr, x=x, y=y, hue=z, errorbar=None, ax=ax)

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
    fig.tight_layout()
    canvas = FigureCanvas(fig)

    return canvas


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


def read_raw_semilab():
    """read raw spectra data of semilab equiments"""
    pass


def read_2Dmaps_ls():
    """Read 2d maps labspec6"""
    pass


def show_alert(message):
    """Show alert"""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Alert")
    msg_box.setText(message)
    msg_box.exec_()


def view_df(tabWidget, df):
    """View selected dataframe"""
    # Create a QDialog to contain the table
    df_viewer = QDialog(tabWidget)
    df_viewer.setWindowTitle("DataFrame Viewer")
    # Create a QTableWidget and populate it with data from the DataFrame
    table_widget = QTableWidget(df_viewer)
    table_widget.setColumnCount(df.shape[1])
    table_widget.setRowCount(df.shape[0])
    table_widget.setHorizontalHeaderLabels(df.columns)
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            item = QTableWidgetItem(str(df.iat[row, col]))
            table_widget.setItem(row, col, item)
    # Set the resizing mode for the QTableWidget to make it resizable
    table_widget.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
    # Use a QVBoxLayout to arrange the table within a scroll area
    layout = QVBoxLayout(df_viewer)
    layout.addWidget(table_widget)
    df_viewer.exec_()


def view_text(ui, title, text):
    """ Create a QTextBrowser to display a text content"""
    report_viewer = QDialog(ui)
    report_viewer.setWindowTitle(title)
    report_viewer.setGeometry(100, 100, 800, 600)

    # Create a QTextBrowser to display the report content
    text_browser = QTextBrowser(report_viewer)
    text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    text_browser.setOpenExternalLinks(True)

    # Display the report text in QTextBrowser
    text_browser.setPlainText(text)
    # Scroll to top of document
    text_browser.moveCursor(QTextCursor.Start)
    # Show the Report viewer dialog
    layout = QVBoxLayout(report_viewer)
    layout.addWidget(text_browser)
    report_viewer.exec()


def view_md_doc(ui, fname):
    """ Create a QDialog to display a markdown file"""
    markdown_viewer = QDialog(ui)
    markdown_viewer.setWindowTitle("Markdown Viewer")
    markdown_viewer.setGeometry(100, 100, 800, 600)

    # Create a QTextBrowser to display the Markdown content
    text_browser = QTextBrowser(markdown_viewer)
    text_browser.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    text_browser.setOpenExternalLinks(
        True)  # Allow opening links in a web browser

    # Load and display the Markdown file
    with open(fname, 'r', encoding='utf-8') as markdown_file:
        markdown_content = markdown_file.read()
        text_browser.setMarkdown(markdown_content)
    text_browser.moveCursor(QTextCursor.Start)  # Scroll to top of document

    layout = QVBoxLayout(markdown_viewer)
    layout.addWidget(text_browser)
    markdown_viewer.exec()  # Show the Markdown viewer dialog


def dark_palette():
    # Get the dark color palette of the application
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(75, 75, 75))
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


def light_palette():
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


class FitThread(QThread):
    fit_progress_changed = Signal(int)
    fit_progress = Signal(int, float)  # number and elapsed time
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
            self.spectra_fs.apply_model(self.model_fs, fnames=[fname])
            num += 1
            elapsed_time = time.time() - start_time
            self.fit_progress.emit(num, elapsed_time)
        self.fit_progress_changed.emit(100)
        self.fit_completed.emit()
