# spectroview/settings/settings_dialog.py
"""Modern two-panel Settings dialog with left sidebar navigation, real-time search, and modular tabs."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QEvent, QModelIndex, QObject, QRect, QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QKeyEvent,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from spectroview import VERSION
from spectroview.settings.ui.tab_about import AboutTabPage
from spectroview.settings.ui.tab_ai import AISettingsTabPage
from spectroview.settings.ui.tab_fit import FitSettingsTabPage
from spectroview.settings.ui.tab_general import GeneralTabPage
from spectroview.settings.vm_settings import VMSettings
from spectroview.view.theme import Icons, ThemeManager


# ---------------------------------------------------------------------------
# Navigation Item Delegate (Sidebar styling)
# ---------------------------------------------------------------------------

class NavItemDelegate(QStyledItemDelegate):
    """Renders sidebar items with rounded hover/selection styling and accent pill indicator."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        painter.save()
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            rect = option.rect
            is_selected = bool(option.state & QStyle.State_Selected)
            is_hover = bool(option.state & QStyle.State_MouseOver)

            mgr = ThemeManager.instance()
            is_dark = mgr.is_dark if mgr is not None else True
            t = mgr.tokens if mgr is not None else None

            # Color palette
            accent_col = QColor(t.accent) if t else QColor("#3B82F6")
            text_primary = QColor(t.text_primary) if t else (QColor("#FFFFFF") if is_dark else QColor("#111827"))
            text_secondary = QColor(t.text_secondary) if t else (QColor("#9CA3AF") if is_dark else QColor("#6B7280"))
            border_col = QColor(t.border) if t else (QColor("#374151") if is_dark else QColor("#E5E7EB"))
            elevated_bg = QColor(t.surface_elevated) if t else (QColor("#25282F") if is_dark else QColor("#FFFFFF"))

            # Background pill
            bg_rect = rect.adjusted(6, 2, -6, -2)
            if is_selected:
                painter.setPen(QPen(border_col, 1))
                painter.setBrush(elevated_bg)
                painter.drawRoundedRect(bg_rect, 6, 6)

                # Left accent indicator pill
                pill_rect = QRectF(rect.left() + 8, rect.center().y() - 9, 3.5, 18)
                painter.setPen(Qt.NoPen)
                painter.setBrush(accent_col)
                painter.drawRoundedRect(pill_rect, 1.75, 1.75)
            elif is_hover:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 255, 255, 18) if is_dark else QColor(0, 0, 0, 12))
                painter.drawRoundedRect(bg_rect, 6, 6)

            # Draw icon
            icon: Optional[QIcon] = index.data(Qt.DecorationRole)
            icon_x = rect.left() + 20
            if icon and not icon.isNull():
                icon_rect = QRect(icon_x, int(rect.center().y() - 10), 20, 20)
                icon.paint(painter, icon_rect, Qt.AlignCenter)
                text_x = icon_x + 28
            else:
                text_x = icon_x + 6

            # Draw text
            text = index.data(Qt.DisplayRole)
            text_rect = QRect(text_x, rect.top(), rect.right() - text_x - 8, rect.height())
            font = option.font
            font.setPointSize(10)
            if is_selected:
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(text_primary)
            else:
                painter.setFont(font)
                painter.setPen(text_secondary)

            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)
        finally:
            painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(210, 40)


# ---------------------------------------------------------------------------
# Compatibility Proxy for Legacy QTabWidget Access
# ---------------------------------------------------------------------------

class _TabsCompatibilityProxy:
    """Provides backward-compatibility for callers or tests expecting QTabWidget methods on dialog.tabs."""

    def __init__(self, dialog: SettingsDialog) -> None:
        self._dialog = dialog

    def count(self) -> int:
        return self._dialog._stack.count()

    def widget(self, index: int) -> Optional[QWidget]:
        return self._dialog._stack.widget(index)

    def indexOf(self, widget: QWidget) -> int:
        return self._dialog._stack.indexOf(widget)

    def tabText(self, index: int) -> str:
        if 0 <= index < self._dialog._nav_list.count():
            item = self._dialog._nav_list.item(index)
            return item.text() if item else ""
        return ""

    def setCurrentWidget(self, widget: QWidget) -> None:
        idx = self._dialog._stack.indexOf(widget)
        if idx >= 0:
            self._dialog._nav_list.setCurrentRow(idx)

    def currentWidget(self) -> Optional[QWidget]:
        return self._dialog._stack.currentWidget()

    def currentIndex(self) -> int:
        return self._dialog._stack.currentIndex()

    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < self._dialog._nav_list.count():
            self._dialog._nav_list.setCurrentRow(index)

    def addTab(self, widget: QWidget, label: str) -> int:
        idx = self._dialog._stack.addWidget(widget)
        item = QListWidgetItem(label)
        self._dialog._nav_list.addItem(item)
        return idx


# ---------------------------------------------------------------------------
# Main SettingsDialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    """Modern two-panel Settings dialog with search filtering and modular section pages."""

    def __init__(self, vm: VMSettings, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.vm = vm

        self.setWindowTitle("Settings")
        self.resize(820, 640)
        self.setMinimumSize(720, 520)

        self._init_ui()
        self._connect_vm()

        # Load initial values from VM
        self.vm.load()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Center split: Left Sidebar + Right Content Area
        split_widget = QWidget()
        split_layout = QHBoxLayout(split_widget)
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(0)

        # -- Left Sidebar ---------------------------------------------------
        self._sidebar = QWidget()
        self._sidebar.setFixedWidth(230)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(10, 14, 10, 10)
        sidebar_layout.setSpacing(10)

        # Top Search Bar
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search settings...")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._on_search_text_changed)
        self._search_edit.returnPressed.connect(self._on_search_return_pressed)
        self._search_edit.installEventFilter(self)
        sidebar_layout.addWidget(self._search_edit)

        # Navigation List
        self._nav_list = QListWidget()
        self._nav_list.setFrameShape(QFrame.NoFrame)
        self._nav_list.setItemDelegate(NavItemDelegate(self))
        self._nav_list.currentRowChanged.connect(self._on_nav_row_changed)
        sidebar_layout.addWidget(self._nav_list, 1)

        split_layout.addWidget(self._sidebar)

        # Vertical Divider
        self._v_divider = QFrame()
        self._v_divider.setFrameShape(QFrame.VLine)
        split_layout.addWidget(self._v_divider)

        # -- Right Content Area ----------------------------------------------
        self._content_container = QWidget()
        content_layout = QVBoxLayout(self._content_container)
        content_layout.setContentsMargins(16, 12, 16, 10)
        content_layout.setSpacing(8)

        # Section Title Header
        self._content_title = QLabel()
        self._content_title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 2px 0;")
        content_layout.addWidget(self._content_title)

        self._header_divider = QFrame()
        self._header_divider.setFrameShape(QFrame.HLine)
        content_layout.addWidget(self._header_divider)

        # Stacked Pages
        self._stack = QStackedWidget()
        content_layout.addWidget(self._stack, 1)

        split_layout.addWidget(self._content_container, 1)
        main_layout.addWidget(split_widget, 1)

        # -- Bottom Action Bar -----------------------------------------------
        self._bottom_divider = QFrame()
        self._bottom_divider.setFrameShape(QFrame.HLine)
        main_layout.addWidget(self._bottom_divider)

        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(16, 10, 16, 10)

        self._version_label = QLabel(f"v{VERSION}")
        self._version_label.setStyleSheet("color: gray; font-size: 11px;")
        bottom_bar.addWidget(self._version_label)

        bottom_bar.addStretch(1)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        save_btn = self._buttons.button(QDialogButtonBox.Save)
        cancel_btn = self._buttons.button(QDialogButtonBox.Cancel)
        if save_btn:
            save_btn.setIcon(Icons.check())
            save_btn.setCursor(Qt.PointingHandCursor)
        if cancel_btn:
            cancel_btn.setIcon(Icons.close())
            cancel_btn.setCursor(Qt.PointingHandCursor)

        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        bottom_bar.addWidget(self._buttons)

        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_bar)
        main_layout.addWidget(bottom_widget)

        # -- Initialize Tab Pages --------------------------------------------
        self._tab_general = GeneralTabPage(self)
        self._tab_fit = FitSettingsTabPage(self)
        self._tab_ai = AISettingsTabPage(self)
        self._tab_about = AboutTabPage(self)

        # Sections definition: (widget, title, icon, keywords)
        self._sections: list[tuple[QWidget, str, QIcon, list[str]]] = [
            (self._tab_general, "General", Icons.settings(size=20), self._tab_general.KEYWORDS),
            (self._tab_fit, "Fit Settings", Icons.calc(size=20), self._tab_fit.KEYWORDS),
            (self._tab_ai, "AI Settings", Icons.sparkles(size=20), self._tab_ai.KEYWORDS),
            (self._tab_about, "About", Icons.about(size=20), self._tab_about.KEYWORDS),
        ]

        for page, title, icon, _kw in self._sections:
            self._stack.addWidget(page)
            item = QListWidgetItem(icon, title)
            self._nav_list.addItem(item)

        # Select first tab initially
        self._nav_list.setCurrentRow(0)

        # Compatibility Proxy for code/tests expecting self.tabs
        self.tabs = _TabsCompatibilityProxy(self)

        # Compatibility widget references (forwarding directly to tabs)
        self._bind_compat_properties()

        # Connect internal tab actions
        self.btn_working_folder.clicked.connect(self.vm.pick_working_folder)
        self.btn_export_config.clicked.connect(self._on_export_config)
        self.btn_import_config.clicked.connect(self._on_import_config)
        self.btn_test_connection.clicked.connect(self._on_test_connection)

    def _bind_compat_properties(self) -> None:
        """Bind widget attributes directly on dialog for test suite and caller backward compatibility."""
        # General tab
        self.le_working_folder = self._tab_general.le_working_folder
        self.btn_working_folder = self._tab_general.btn_working_folder
        self.btn_open_working_folder = self._tab_general.btn_open_working_folder
        self.btn_export_config = self._tab_general.btn_export_config
        self.btn_import_config = self._tab_general.btn_import_config
        self.lbl_working_hint = self._tab_general.lbl_working_hint
        self.chk_check_updates_startup = self._tab_general.chk_check_updates_startup

        # Fit tab
        self.chk_fit_negative = self._tab_fit.chk_fit_negative
        self.spin_max_iter = self._tab_fit.spin_max_iter
        self.spin_x_tol = self._tab_fit.spin_x_tol
        self.spin_f_tol = self._tab_fit.spin_f_tol
        self.spin_coef_noise = self._tab_fit.spin_coef_noise
        self.combo_loss = self._tab_fit.combo_loss
        self.spin_f_scale = self._tab_fit.spin_f_scale
        self.spin_maxshift = self._tab_fit.spin_maxshift
        self.spin_minfwhm = self._tab_fit.spin_minfwhm
        self.spin_maxfwhm = self._tab_fit.spin_maxfwhm

        # AI tab
        self.edit_custom = self._tab_ai.edit_custom
        self.edit_custom_url = self._tab_ai.edit_custom_url
        self.edit_custom_models = self._tab_ai.edit_custom_models
        self.edit_custom_label = self._tab_ai.edit_custom_label
        self.btn_toggle_providers = self._tab_ai.btn_toggle_providers
        self.frame_provider_presets = self._tab_ai.frame_provider_presets
        self.edit_openai = self._tab_ai.edit_openai
        self.edit_anthropic = self._tab_ai.edit_anthropic
        self.edit_gemini = self._tab_ai.edit_gemini
        self.edit_deepseek = self._tab_ai.edit_deepseek
        self.edit_mistral = self._tab_ai.edit_mistral
        self.cbb_test_provider = self._tab_ai.cbb_test_provider
        self.btn_test_connection = self._tab_ai.btn_test_connection
        self.lbl_test_status = self._tab_ai.lbl_test_status
        self.chk_mcp_enabled = self._tab_ai.chk_mcp_enabled
        self.spin_mcp_port = self._tab_ai.spin_mcp_port
        self.lbl_mcp_endpoint = self._tab_ai.lbl_mcp_endpoint

        # About tab
        self.btn_check_updates = self._tab_about.btn_check_updates

    # ──────────────────────────────────────────────
    # VM Connections
    # ──────────────────────────────────────────────
    def _connect_vm(self) -> None:
        self.vm.settings_loaded.connect(self._apply_settings)
        self.vm.working_folder_changed.connect(self.le_working_folder.setText)
        self.vm.connection_test_finished.connect(self._on_connection_test_finished)
        self.vm.settings_saved.connect(self.accept)

    def _apply_settings(self, data: dict) -> None:
        self._tab_general.apply_settings(data)
        self._tab_fit.apply_settings(data)
        self._tab_ai.apply_settings(data)

    # ──────────────────────────────────────────────
    # Navigation & Search Filtering
    # ──────────────────────────────────────────────
    def _on_nav_row_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)
            item = self._nav_list.item(row)
            if item:
                self._content_title.setText(item.text())

    def _on_search_text_changed(self, query: str) -> None:
        q = query.strip().lower()
        first_visible = -1
        current_visible = False
        current_row = self._nav_list.currentRow()

        for idx, (_page, title, _icon, keywords) in enumerate(self._sections):
            item = self._nav_list.item(idx)
            if not item:
                continue
            matches = not q or q in title.lower() or any(q in kw.lower() for kw in keywords)
            item.setHidden(not matches)
            if matches:
                if first_visible < 0:
                    first_visible = idx
                if idx == current_row:
                    current_visible = True

        if not current_visible and first_visible >= 0:
            self._nav_list.setCurrentRow(first_visible)

    def _on_search_return_pressed(self) -> None:
        current_row = self._nav_list.currentRow()
        if 0 <= current_row < self._nav_list.count():
            item = self._nav_list.item(current_row)
            if item and not item.isHidden():
                self._nav_list.setFocus()
                return
        for idx in range(self._nav_list.count()):
            item = self._nav_list.item(idx)
            if item and not item.isHidden():
                self._nav_list.setCurrentRow(idx)
                self._nav_list.setFocus()
                break

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._search_edit and event.type() == QEvent.KeyPress:
            key_event = event  # type: ignore[assignment]
            if key_event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._on_search_return_pressed()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._search_edit.hasFocus():
                self._on_search_return_pressed()
                event.accept()
                return
            focused = self.focusWidget()
            if not isinstance(focused, QPushButton):
                event.accept()
                return
        super().keyPressEvent(event)

    # ──────────────────────────────────────────────
    # Section Navigation Shortcuts
    # ──────────────────────────────────────────────
    def select_section(self, name: str) -> None:
        """Select a section by name: 'general', 'fit', 'ai', 'about'."""
        name_lower = name.lower()
        if name_lower in ("general", "gen"):
            idx = self._stack.indexOf(self._tab_general)
        elif name_lower in ("fit", "fitting", "fit_settings"):
            idx = self._stack.indexOf(self._tab_fit)
        elif name_lower in ("ai", "ai_agent", "chat"):
            idx = self._stack.indexOf(self._tab_ai)
        elif name_lower in ("about", "info"):
            idx = self._stack.indexOf(self._tab_about)
        else:
            idx = 0

        if idx >= 0:
            self._nav_list.setCurrentRow(idx)

    def open_about_tab(self) -> None:
        """Focus directly on the About tab."""
        self.select_section("about")

    # ──────────────────────────────────────────────
    # Action Handlers
    # ──────────────────────────────────────────────
    def _on_open_working_folder(self) -> None:
        folder = self.le_working_folder.text().strip()
        if folder:
            if not os.path.exists(folder):
                try:
                    os.makedirs(folder, exist_ok=True)
                except OSError:
                    pass
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_export_config(self) -> None:
        filepath = self.vm.export_config_dialog(self)
        if filepath:
            QMessageBox.information(
                self,
                "Config Exported",
                f"Configuration successfully exported to:\n{filepath}",
            )

    def _on_import_config(self) -> None:
        success = self.vm.import_config_dialog(self)
        if success:
            QMessageBox.information(
                self,
                "Config Restored",
                "Configuration restored successfully from JSON file.",
            )
        else:
            QMessageBox.warning(
                self,
                "Restore Failed",
                "Could not restore configuration from the selected file.",
            )

    def _on_test_connection(self) -> None:
        provider = self.cbb_test_provider.currentText()
        if provider == "Custom":
            api_key = self.edit_custom.text().strip()
            base_url = self.edit_custom_url.text().strip()
            models_text = self.edit_custom_models.text().strip()
            model = models_text.split(",")[0].strip() if models_text else ""
        elif provider == "OpenAI":
            api_key = self.edit_openai.text().strip()
            base_url, model = "", ""
        elif provider == "Anthropic":
            api_key = self.edit_anthropic.text().strip()
            base_url, model = "", ""
        elif provider == "Gemini":
            api_key = self.edit_gemini.text().strip()
            base_url, model = "", ""
        elif provider == "DeepSeek":
            api_key = self.edit_deepseek.text().strip()
            base_url, model = "", ""
        elif provider == "Mistral":
            api_key = self.edit_mistral.text().strip()
            base_url, model = "", ""
        elif provider == "Ollama":
            api_key, base_url, model = "", "", ""
        else:
            api_key, base_url, model = "", "", ""

        self.btn_test_connection.setEnabled(False)
        self.lbl_test_status.setText("Testing connection…")
        self.lbl_test_status.setStyleSheet("color: #FFA726; font-size: 11px;")
        self.lbl_test_status.setToolTip("")
        self.vm.test_connection(provider, api_key, base_url, model)

    def _on_connection_test_finished(self, success: bool, message: str) -> None:
        self.btn_test_connection.setEnabled(True)
        if success:
            self.lbl_test_status.setText(f"✓ {message}")
            self.lbl_test_status.setStyleSheet("color: #1E9E5A; font-size: 11px; font-weight: bold;")
            self.lbl_test_status.setToolTip(message)
        else:
            first_line = message.splitlines()[0] if message else "Failed"
            self.lbl_test_status.setText(f"✕ {first_line}")
            self.lbl_test_status.setStyleSheet("color: #D14343; font-size: 11px;")
            self.lbl_test_status.setToolTip(message)

    def _on_accept(self) -> None:
        data = {}
        data.update(self._tab_general.get_settings())
        data.update(self._tab_fit.get_settings())
        data.update(self._tab_ai.get_settings())
        self.vm.save(data)

    def closeEvent(self, event) -> None:
        if hasattr(self.vm, "cancel_test_connection"):
            self.vm.cancel_test_connection()
        super().closeEvent(event)


# Alias for backward compatibility
VSettingsDialog = SettingsDialog

