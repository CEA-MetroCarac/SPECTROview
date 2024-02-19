"""
Module contains all utilities functions
"""
import os
import copy
#import win32clipboard
from io import BytesIO
import numpy as np
import pandas as pd
from pathlib import Path
from PySide6.QtWidgets import QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout,QTextBrowser
from PySide6.QtCore import Qt, QFile
from PySide6.QtGui import  QPalette, QColor, QTextCursor


def quadrant(row):
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
    """fnc to copy canvas figure to clipboard"""
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
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Alert")
    msg_box.setText(message)
    msg_box.exec_()

def view_df(tabWidget, df):
    """To view selected dataframe"""
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
    # Create a QDialog to display the Markdown content
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