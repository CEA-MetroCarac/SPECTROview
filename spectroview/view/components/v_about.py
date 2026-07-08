# view/components/v_about.py
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextBrowser
from PySide6.QtGui import QPixmap

from spectroview import VERSION, ICON_DIR


class VAboutDialog(QDialog):
    """About dialog displaying application information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About SPECTROview")
        self.setMinimumSize(600, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Application Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap(os.path.join(ICON_DIR, "logo_spectroview.png"))
        scaled_pixmap = logo_pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(scaled_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # About text
        info = f"""
        <div style="text-align: center;">
            <h2>SPECTROview</h2>
            <p><i>(Tool for Spectroscopic Data Processing and Visualization)</i></p>
            <p><b>Version {VERSION}</b></p>
        </div>
        <hr>
        <h3>Features:</h3>
        <ul>
            <li>Supports processing of spectral data (1D) and hyperspectral data (2D maps or wafer maps).</li>
            <li>Ability to fit multiple spectra or 2Dmaps using predefined models or by creating custom fit models.</li>
            <li>Collect all best-fit results with one click.</li>
            <li>Dedicated module for effortless, fast, and easy data visualization.</li>
        </ul>
        <p style="text-align:left;">
            Full documentation is available at <a href="https://CEA-MetroCarac.github.io/SPECTROview">https://CEA-MetroCarac.github.io/SPECTROview</a>
        </p>
        <hr>
        <h3>Citation:</h3>
        <p style="text-align:left;">
            Le, Van Hoan. "SPECTROview: An Open-Source Application for Interactive Spectroscopic Data Processing, Fitting, and Visualization." <i>SoftwareX</i> 35 (September 2026): 102862.
            <br><a href="https://doi.org/10.1016/j.softx.2026.102862">https://doi.org/10.1016/j.softx.2026.102862</a>
        </p>
        """

        info_browser = QTextBrowser()
        info_browser.setHtml(info)
        info_browser.setOpenExternalLinks(True)
        info_browser.setAlignment(Qt.AlignCenter)
        info_browser.setFrameStyle(QTextBrowser.NoFrame)
        layout.addWidget(info_browser)
