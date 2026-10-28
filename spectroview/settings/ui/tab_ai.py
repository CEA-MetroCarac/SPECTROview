# spectroview/settings/ui/tab_ai.py
"""AI settings tab: LLM provider credentials, connection tests, and MCP server options."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def _highlight_group_style() -> str:
    return (
        "QGroupBox { border: 1.5px solid #3B82F6; border-radius: 6px; margin-top: 10px; font-weight: bold; } "
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px; color: #3B82F6; }"
    )


def _highlight_label_style() -> str:
    return "background-color: rgba(59, 130, 246, 0.2); border: 1px solid #3B82F6; border-radius: 4px; padding: 2px 4px; color: #3B82F6; font-weight: bold;"


class AISettingsTabPage(QWidget):
    """The AI Settings tab: provider credentials, model selection, and local MCP server."""

    KEYWORDS = [
        "ai", "chat", "llm", "agent", "assistant", "openai", "anthropic",
        "gemini", "deepseek", "mistral", "ollama", "custom", "api", "key",
        "url", "base url", "model", "models", "test", "test connection",
        "mcp", "server", "port", "endpoint"
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._searchable_rows: list[tuple[list[str], QLabel, bool]] = []
        self._init_ui()

    def _init_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ───── API Keys ─────
        self.grp_api = QGroupBox("LLM Provider Configuration")
        api_layout = QVBoxLayout(self.grp_api)
        api_layout.setContentsMargins(8, 8, 8, 8)
        api_layout.setSpacing(6)

        lbl_vault_note = QLabel("🔒 API keys are securely stored in the system credential vault and excluded from config backups.")
        lbl_vault_note.setStyleSheet("color: gray; font-size: 11px; font-style: italic; margin-bottom: 4px;")
        api_layout.addWidget(lbl_vault_note)

        def add_api_row(
            box_layout: QVBoxLayout,
            label: str,
            widget: QWidget,
            tooltip: str = "",
            is_preset: bool = False,
            keywords: list[str] | None = None,
        ) -> QLabel:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(130)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if tooltip:
                lbl.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            box_layout.addLayout(row)
            terms = [label.lower(), tooltip.lower()]
            if keywords:
                terms.extend(keywords)
            self._searchable_rows.append((terms, lbl, is_preset))
            return lbl

        self.edit_custom = QLineEdit()
        self.edit_custom.setEchoMode(QLineEdit.Password)
        self.edit_custom.setPlaceholderText("Enter custom API key or leave blank if self-hosted / not needed")
        add_api_row(api_layout, "Custom API Key:", self.edit_custom, keywords=["custom", "key", "api"])

        self.edit_custom_url = QLineEdit()
        self.edit_custom_url.setPlaceholderText("https://api.example.com/v1")
        add_api_row(api_layout, "Base URL:", self.edit_custom_url, "OpenAI-compatible base URL for custom or local endpoints", keywords=["url", "endpoint"])

        self.edit_custom_models = QLineEdit()
        self.edit_custom_models.setPlaceholderText("model-a, model-b, model-c")
        add_api_row(
            api_layout,
            "Model Name(s):",
            self.edit_custom_models,
            "Comma-separated list of model names for the custom provider dropdown",
            keywords=["model", "models"],
        )

        self.edit_custom_label = QLineEdit()
        self.edit_custom_label.setPlaceholderText("e.g. Lab LLM, Internal Mistral (default: Custom)")
        add_api_row(
            api_layout,
            "Display Label:",
            self.edit_custom_label,
            "Display label shown for this provider in the AI chat panel",
            keywords=["label", "display"],
        )

        # ───── Collapsible "Other cloud providers" ─────
        self.btn_toggle_providers = QToolButton()
        self.btn_toggle_providers.setCheckable(True)
        self.btn_toggle_providers.setChecked(False)
        self.btn_toggle_providers.setText("▸ Other cloud providers:")
        self.btn_toggle_providers.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_providers.toggled.connect(self._on_toggle_providers)
        api_layout.addWidget(self.btn_toggle_providers, alignment=Qt.AlignLeft)

        self.frame_provider_presets = QFrame()
        self.frame_provider_presets.setFrameShape(QFrame.NoFrame)
        provider_layout = QVBoxLayout(self.frame_provider_presets)
        provider_layout.setContentsMargins(10, 4, 4, 4)
        provider_layout.setSpacing(6)

        self.edit_openai = QLineEdit()
        self.edit_openai.setEchoMode(QLineEdit.Password)
        self.edit_openai.setPlaceholderText("sk-...")
        add_api_row(provider_layout, "OpenAI API Key:", self.edit_openai, is_preset=True, keywords=["openai", "gpt"])

        self.edit_anthropic = QLineEdit()
        self.edit_anthropic.setEchoMode(QLineEdit.Password)
        self.edit_anthropic.setPlaceholderText("sk-ant-...")
        add_api_row(provider_layout, "Anthropic API Key:", self.edit_anthropic, is_preset=True, keywords=["anthropic", "claude"])

        self.edit_gemini = QLineEdit()
        self.edit_gemini.setEchoMode(QLineEdit.Password)
        self.edit_gemini.setPlaceholderText("AIzaSy...")
        add_api_row(provider_layout, "Gemini API Key:", self.edit_gemini, is_preset=True, keywords=["gemini", "google"])

        self.edit_deepseek = QLineEdit()
        self.edit_deepseek.setEchoMode(QLineEdit.Password)
        self.edit_deepseek.setPlaceholderText("sk-...")
        add_api_row(provider_layout, "DeepSeek API Key:", self.edit_deepseek, is_preset=True, keywords=["deepseek"])

        self.edit_mistral = QLineEdit()
        self.edit_mistral.setEchoMode(QLineEdit.Password)
        self.edit_mistral.setPlaceholderText("...")
        add_api_row(provider_layout, "Mistral API Key:", self.edit_mistral, is_preset=True, keywords=["mistral", "codestral"])

        self.frame_provider_presets.setVisible(False)
        api_layout.addWidget(self.frame_provider_presets)

        # ───── Test Connection ─────
        test_row = QHBoxLayout()
        test_row.setContentsMargins(0, 4, 0, 0)
        test_row.setSpacing(8)

        lbl_test = QLabel("Test:")
        lbl_test.setFixedWidth(130)
        lbl_test.setTextInteractionFlags(Qt.TextSelectableByMouse)
        test_row.addWidget(lbl_test)

        self.cbb_test_provider = QComboBox()
        self.cbb_test_provider.setObjectName("cbb_test_provider")
        self.cbb_test_provider.addItems(["Custom", "OpenAI", "Anthropic", "Gemini", "DeepSeek", "Mistral", "Ollama"])
        self.cbb_test_provider.setToolTip("Select provider to test")
        self.cbb_test_provider.setFixedWidth(110)
        test_row.addWidget(self.cbb_test_provider)

        self.btn_test_connection = QPushButton("Test Connection")
        self.btn_test_connection.setObjectName("btn_test_connection")
        self.btn_test_connection.setCursor(Qt.PointingHandCursor)
        test_row.addWidget(self.btn_test_connection)

        self.lbl_test_status = QLabel("")
        self.lbl_test_status.setObjectName("lbl_test_status")
        self.lbl_test_status.setStyleSheet("font-size: 11px;")
        self.lbl_test_status.setWordWrap(True)
        self.lbl_test_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        test_row.addWidget(self.lbl_test_status, 1)

        api_layout.addLayout(test_row)
        layout.addWidget(self.grp_api)

        # ───── Local MCP Server ─────
        self.grp_mcp = QGroupBox("Local MCP Server")
        mcp_layout = QVBoxLayout(self.grp_mcp)
        mcp_layout.setContentsMargins(8, 8, 8, 8)
        mcp_layout.setSpacing(6)

        self.chk_mcp_enabled = QCheckBox("Enable local Model Context Protocol (MCP) endpoint")
        self.chk_mcp_enabled.setToolTip(
            "Expose the running SPECTROview session to MCP clients on this computer only."
        )
        mcp_layout.addWidget(self.chk_mcp_enabled)

        mcp_row = QHBoxLayout()
        mcp_row.addWidget(QLabel("Port:"))
        self.spin_mcp_port = QSpinBox()
        self.spin_mcp_port.setRange(1024, 65535)
        self.spin_mcp_port.setValue(8765)
        mcp_row.addWidget(self.spin_mcp_port)
        mcp_row.addStretch()
        mcp_layout.addLayout(mcp_row)

        self.lbl_mcp_endpoint = QLabel("Endpoint: http://127.0.0.1:8765/mcp")
        self.lbl_mcp_endpoint.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_mcp_endpoint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.spin_mcp_port.valueChanged.connect(
            lambda port: self.lbl_mcp_endpoint.setText(
                f"Endpoint: http://127.0.0.1:{port}/mcp"
            )
        )
        mcp_layout.addWidget(self.lbl_mcp_endpoint)
        layout.addWidget(self.grp_mcp)

        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._scroll.setWidget(container)
        outer_layout.addWidget(self._scroll)

    def _on_toggle_providers(self, checked: bool) -> None:
        self.frame_provider_presets.setVisible(checked)
        arrow = "▾" if checked else "▸"
        self.btn_toggle_providers.setText(f"{arrow} Other cloud providers:")

    def apply_settings(self, data: dict) -> None:
        self.edit_openai.setText(data.get("api_key_OpenAI", ""))
        self.edit_anthropic.setText(data.get("api_key_Anthropic", ""))
        self.edit_gemini.setText(data.get("api_key_Gemini", ""))
        self.edit_deepseek.setText(data.get("api_key_DeepSeek", ""))
        self.edit_mistral.setText(data.get("api_key_Mistral", ""))
        self.edit_custom.setText(data.get("api_key_Custom", ""))
        self.edit_custom_url.setText(data.get("custom_base_url", ""))
        self.edit_custom_models.setText(data.get("custom_models", ""))
        self.edit_custom_label.setText(data.get("custom_label", ""))
        self.chk_mcp_enabled.setChecked(data.get("mcp_enabled", False))
        self.spin_mcp_port.setValue(data.get("mcp_port", 8765))

    def get_settings(self) -> dict:
        return {
            "api_key_OpenAI": self.edit_openai.text().strip(),
            "api_key_Anthropic": self.edit_anthropic.text().strip(),
            "api_key_Gemini": self.edit_gemini.text().strip(),
            "api_key_DeepSeek": self.edit_deepseek.text().strip(),
            "api_key_Mistral": self.edit_mistral.text().strip(),
            "api_key_Custom": self.edit_custom.text().strip(),
            "custom_base_url": self.edit_custom_url.text().strip(),
            "custom_models": self.edit_custom_models.text().strip(),
            "custom_label": self.edit_custom_label.text().strip(),
            "mcp_enabled": self.chk_mcp_enabled.isChecked(),
            "mcp_port": self.spin_mcp_port.value(),
        }

    # ───── Search Highlighting & Scrolling ─────
    def highlight_search(self, query: str) -> None:
        """Highlight matching options/groupboxes and scroll first match into view."""
        q = query.strip().lower()

        # Clear existing styles
        self.grp_api.setStyleSheet("")
        self.grp_mcp.setStyleSheet("")
        for _terms, lbl, _is_p in self._searchable_rows:
            lbl.setStyleSheet("")

        if not q:
            return

        first_match: Optional[QWidget] = None

        # Check MCP server
        mcp_terms = ["mcp", "server", "port", "endpoint", "protocol"]
        if any(term in q or q in term for term in mcp_terms):
            self.grp_mcp.setStyleSheet(_highlight_group_style())
            first_match = self.grp_mcp

        # Check rows
        for terms, lbl, is_preset in self._searchable_rows:
            if any(q in t or t in q for t in terms):
                if is_preset and not self.btn_toggle_providers.isChecked():
                    self.btn_toggle_providers.setChecked(True)
                lbl.setStyleSheet(_highlight_label_style())
                if first_match is None:
                    first_match = lbl

        if first_match:
            self._scroll.ensureWidgetVisible(first_match, 0, 30)
