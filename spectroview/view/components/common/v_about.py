# view/components/v_about.py
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QTextBrowser, QVBoxLayout

from spectroview import BUILD_NUMBER, ICON_DIR, VERSION
from spectroview.i18n import bind_window_title, get_language, on_language_changed, tr


class VAboutDialog(QDialog):
    """About dialog displaying application information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        bind_window_title(self, "About SPECTROview")
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

        self.info_browser = QTextBrowser()
        self.info_browser.setOpenExternalLinks(True)
        self.info_browser.setAlignment(Qt.AlignCenter)
        self.info_browser.setFrameStyle(QTextBrowser.NoFrame)
        self._refresh_html()
        on_language_changed(self, self._refresh_html, slot_key="about_html")
        layout.addWidget(self.info_browser)

    def _refresh_html(self) -> None:
        docs_url = (
            "https://CEA-MetroCarac.github.io/SPECTROview/fr/"
            if get_language() == "fr"
            else "https://CEA-MetroCarac.github.io/SPECTROview"
        )
        info = f"""
        <div style="text-align: center;">
            <h2>SPECTROview</h2>
            <p><i>({tr("Tool for Spectroscopic Data Processing and Visualization")})</i></p>
            <p style="margin: 0; padding: 0;"><b>{tr("Version {version}", version=VERSION)}</b></p>
            <p style="margin-top: 3px; font-size: 11px; color: #8e8e93;">{tr("Build: {build}", build=BUILD_NUMBER)}</p>
        </div>
        <hr>
        <h3>{tr("Features:")}</h3>
        <ul>
            <li>{tr("Supports processing of spectral data (1D) and hyperspectral data (2D maps or wafer maps).")}</li>
            <li>{tr("Ability to fit multiple spectra or 2Dmaps using predefined models or by creating custom fit models.")}</li>
            <li>{tr("Collect all best-fit results with one click.")}</li>
            <li>{tr("Dedicated module for effortless, fast, and easy data visualization.")}</li>
        </ul>
        <p style="text-align:left;">
            {tr("Full documentation is available at")} <a href="{docs_url}">{docs_url}</a>
        </p>
        <hr>
        <h3>{tr("Citation:")}</h3>
        <p style="text-align:left;">
            Le, Van Hoan. "SPECTROview: An Open-Source Application for Interactive Spectroscopic Data Processing, Fitting, and Visualization." <i>SoftwareX</i> 35 (September 2026): 102862.
            <br><a href="https://doi.org/10.1016/j.softx.2026.102862">https://doi.org/10.1016/j.softx.2026.102862</a>
        </p>
        """
        self.info_browser.setHtml(info)
