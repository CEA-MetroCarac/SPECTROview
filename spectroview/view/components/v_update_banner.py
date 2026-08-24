"""The in-app notification banner for a newer SPECTROview release."""

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


_DARK_STYLE = """
    QWidget#UpdateBanner {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #1a3a5c, stop:0.5 #1e4976, stop:1 #1a3a5c);
        border-bottom: 1px solid #2d6aad;
    }
    QLabel#banner_icon { color: #64b5f6; font-size: 15px; }
    QLabel#banner_text { color: #e3f2fd; font-size: 12px; }
    QLabel#banner_version { color: #64b5f6; font-weight: bold; font-size: 12px; }
    QPushButton#btn_update {
        background: #2e7d32; color: white; border: 1px solid #66bb6a;
        border-radius: 4px; padding: 3px 10px; font-size: 11px; font-weight: bold;
    }
    QPushButton#btn_update:hover { background: #388e3c; }
    QPushButton#btn_download {
        background: #1565c0; color: #e3f2fd; border: 1px solid #42a5f5;
        border-radius: 4px; padding: 3px 10px; font-size: 11px; font-weight: bold;
    }
    QPushButton#btn_download:hover { background: #1976d2; }
    QPushButton#btn_skip, QPushButton#btn_dismiss {
        background: transparent; color: #90a4ae; border: 1px solid #546e7a;
        border-radius: 4px; padding: 3px 8px; font-size: 11px;
    }
    QPushButton#btn_skip:hover, QPushButton#btn_dismiss:hover {
        color: #cfd8dc; border-color: #78909c;
    }
"""

_LIGHT_STYLE = """
    QWidget#UpdateBanner {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #dbeeff, stop:0.5 #c8e3fb, stop:1 #dbeeff);
        border-bottom: 1px solid #90caf9;
    }
    QLabel#banner_icon { color: #1565c0; font-size: 15px; }
    QLabel#banner_text { color: #1a2e46; font-size: 12px; }
    QLabel#banner_version { color: #1565c0; font-weight: bold; font-size: 12px; }
    QPushButton#btn_update {
        background: #2e7d32; color: white; border: none; border-radius: 4px;
        padding: 3px 10px; font-size: 11px; font-weight: bold;
    }
    QPushButton#btn_update:hover { background: #388e3c; }
    QPushButton#btn_download {
        background: #1976d2; color: white; border: none; border-radius: 4px;
        padding: 3px 10px; font-size: 11px; font-weight: bold;
    }
    QPushButton#btn_download:hover { background: #1565c0; }
    QPushButton#btn_skip, QPushButton#btn_dismiss {
        background: transparent; color: #546e7a; border: 1px solid #b0bec5;
        border-radius: 4px; padding: 3px 8px; font-size: 11px;
    }
    QPushButton#btn_skip:hover, QPushButton#btn_dismiss:hover {
        color: #37474f; border-color: #78909c;
    }
"""


class VUpdateBanner(QWidget):
    """Display release actions and start the supplied automatic-update callback."""

    def __init__(
        self,
        tag: str,
        html_url: str,
        on_skip,
        on_dismiss,
        on_update=None,
        wheel_url: str = "",
        wheel_sha256: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._tag = tag
        self._html_url = html_url
        self._on_skip = on_skip
        self._on_dismiss = on_dismiss
        self._on_update = on_update
        self._wheel_url = wheel_url
        self._wheel_sha256 = wheel_sha256

        self.setObjectName("UpdateBanner")
        self.setFixedHeight(36)
        self._build_ui()
        self.apply_theme("dark")

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 6, 0)
        layout.setSpacing(8)

        icon_label = QLabel("🔔")
        icon_label.setObjectName("banner_icon")
        layout.addWidget(icon_label)

        text_label = QLabel("A new version of SPECTROview is available:")
        text_label.setObjectName("banner_text")
        layout.addWidget(text_label)

        version_label = QLabel(self._tag.lstrip("v"))
        version_label.setObjectName("banner_version")
        layout.addWidget(version_label)
        layout.addStretch(1)

        self.btn_update = QPushButton("⬇ Update")
        self.btn_update.setObjectName("btn_update")
        self.btn_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update.setToolTip("Download, install, and restart with this release.")
        self.btn_update.clicked.connect(self._request_update)
        self.btn_update.setEnabled(bool(self._wheel_url and self._on_update))
        if not self.btn_update.isEnabled():
            self.btn_update.setToolTip("This release does not include an installable wheel.")
        layout.addWidget(self.btn_update)

        self.btn_download = QPushButton("🔍 Show changelog")
        self.btn_download.setObjectName("btn_download")
        self.btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_download.clicked.connect(self._open_release_page)
        layout.addWidget(self.btn_download)

        self.btn_dismiss = QPushButton("🕒 Update later")
        self.btn_dismiss.setObjectName("btn_dismiss")
        self.btn_dismiss.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_dismiss.clicked.connect(self._dismiss)
        layout.addWidget(self.btn_dismiss)

        self.btn_skip = QPushButton("🚫 Skip this version")
        self.btn_skip.setObjectName("btn_skip")
        self.btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_skip.clicked.connect(self._skip)
        layout.addWidget(self.btn_skip)

    def apply_theme(self, theme_key: str) -> None:
        """Switch banner colours to match the application theme."""
        style = _DARK_STYLE if theme_key in ("dark", "soft_dark", "classic_dark") else _LIGHT_STYLE
        self.setStyleSheet(style)

    def set_download_progress(self, percent: int) -> None:
        """Show automatic-update download progress and prevent duplicate requests."""
        self.btn_update.setEnabled(False)
        self.btn_update.setText("⬇ Downloading…" if percent < 0 else f"⬇ Downloading… {percent}%")

    def set_update_error(self) -> None:
        """Restore the update control after a recoverable download failure."""
        self.btn_update.setText("↻ Retry update")
        self.btn_update.setEnabled(bool(self._wheel_url and self._on_update))

    def _request_update(self) -> None:
        if self._on_update is not None and self._wheel_url:
            self._on_update(self._tag, self._wheel_url, self._wheel_sha256)

    def _open_release_page(self) -> None:
        if self._html_url:
            QDesktopServices.openUrl(QUrl(self._html_url))

    def _skip(self) -> None:
        self._on_skip(self._tag)
        self._dismiss()

    def _dismiss(self) -> None:
        self._on_dismiss()
        self.hide()
