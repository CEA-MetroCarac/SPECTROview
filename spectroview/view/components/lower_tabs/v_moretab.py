# spectroview/view/components/lower_tabs/v_moretab.py
"""Widget for displaying spectrum acquisition metadata and custom properties."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spectroview.i18n import (
    bind_form_row,
    bind_header_labels,
    bind_text,
    bind_title,
    bind_tooltip,
)
from spectroview.view.theme import Icons, current_tokens


class VMoreTab(QWidget):
    """View for Metadata tab - displays acquisition metadata, custom properties, and normalization."""

    normalize_requested = Signal(float)
    undo_normalization_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the UI layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # --- LEFT: Metadata Table ---
        self.group_metadata = QGroupBox()
        bind_title(self.group_metadata, "Metadata")
        layout_metadata = QVBoxLayout(self.group_metadata)
        layout_metadata.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.setFrameShape(QTableWidget.NoFrame)
        self.table.setColumnCount(2)
        bind_header_labels(self.table, ["Parameter", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setStyleSheet("QTableWidget::item { padding: 2px 4px; }")
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout_metadata.addWidget(self.table)

        # Placeholder label for no data
        tok = current_tokens()
        self.lbl_placeholder = QLabel()
        bind_text(
            self.lbl_placeholder,
            "No metadata available\n\nSelect a spectrum to view its metadata",
        )
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet(f"color: {tok.text_muted}; font-size: 14px; padding: 40px;")
        layout_metadata.addWidget(self.lbl_placeholder)

        main_layout.addWidget(self.group_metadata, stretch=2)

        # --- MIDDLE: Custom Attributes ---
        self.group_custom = QGroupBox()
        bind_title(self.group_custom, "Other informations")
        layout_custom = QFormLayout(self.group_custom)

        self.lbl_label = QLineEdit()
        self.lbl_label.setReadOnly(True)

        self.lbl_color = QLineEdit()
        self.lbl_color.setReadOnly(True)

        self.lbl_xcorr = QLineEdit()
        self.lbl_xcorr.setReadOnly(True)

        self.lbl_source = QLineEdit()
        self.lbl_source.setReadOnly(True)

        self.lbl_baseline_subtracted = QLineEdit()
        self.lbl_baseline_subtracted.setReadOnly(True)

        self.lbl_norm_factor = QLineEdit()
        self.lbl_norm_factor.setReadOnly(True)

        self.lbl_baseline_mode = QLineEdit()
        self.lbl_baseline_mode.setReadOnly(True)

        self.lbl_baseline_coef = QLineEdit()
        self.lbl_baseline_coef.setReadOnly(True)

        self.lbl_baseline_points = QLineEdit()
        self.lbl_baseline_points.setReadOnly(True)

        bind_form_row(layout_custom, "Label:", self.lbl_label)
        bind_form_row(layout_custom, "Color:", self.lbl_color)
        bind_form_row(layout_custom, "X-Correction:", self.lbl_xcorr)
        bind_form_row(layout_custom, "Source Path:", self.lbl_source)
        bind_form_row(layout_custom, "Intensity Normalization Factor:", self.lbl_norm_factor)
        bind_form_row(layout_custom, "Baseline Mode:", self.lbl_baseline_mode)
        bind_form_row(layout_custom, "Baseline Coef:", self.lbl_baseline_coef)
        bind_form_row(layout_custom, "Baseline Points:", self.lbl_baseline_points)
        bind_form_row(layout_custom, "Baseline Subtracted:", self.lbl_baseline_subtracted)

        main_layout.addWidget(self.group_custom, stretch=2)

        # --- RIGHT: Intensity Normalization ---
        self.group_norm = QGroupBox()
        bind_title(self.group_norm, "More options")
        layout_norm = QVBoxLayout(self.group_norm)

        lbl_norm_title = QLabel()
        bind_text(lbl_norm_title, "Normalization Factor:")
        layout_norm.addWidget(lbl_norm_title)
        self.spin_norm_factor = QDoubleSpinBox()
        self.spin_norm_factor.setDecimals(4)
        self.spin_norm_factor.setRange(1e-6, 1e6)
        self.spin_norm_factor.setValue(1.0)

        self.btn_normalize = QPushButton()
        bind_text(self.btn_normalize, "Normalize")
        self.btn_normalize.setIcon(Icons.norm())
        bind_tooltip(self.btn_normalize, "Normalize the current spectrum")
        self.btn_undo_norm = QPushButton()
        bind_text(self.btn_undo_norm, "Undo")
        self.btn_undo_norm.setIcon(Icons.undo())
        bind_tooltip(self.btn_undo_norm, "Undo the last normalization")

        # Horizontal layout for normalize buttons
        norm_btns_layout = QHBoxLayout()
        norm_btns_layout.addWidget(self.btn_undo_norm)
        norm_btns_layout.addWidget(self.btn_normalize)

        layout_norm.addWidget(self.spin_norm_factor)
        layout_norm.addLayout(norm_btns_layout)
        layout_norm.addStretch()

        main_layout.addWidget(self.group_norm, stretch=1)

        # Connect signals
        self.btn_normalize.clicked.connect(self._on_normalize)
        self.btn_undo_norm.clicked.connect(self.undo_normalization_requested.emit)

        # Initially clear all blocks
        self.clear_metadata()

    def _on_normalize(self):
        factor = self.spin_norm_factor.value()
        self.normalize_requested.emit(factor)

    def show_metadata(self, item):
        """Display metadata and custom properties from a spectrum object or a dict."""

        # 1. Determine metadata and spectrum
        if isinstance(item, dict):
            metadata = item
            spectrum = None
        elif item is not None:
            metadata = getattr(item, 'metadata', {})
            spectrum = item
        else:
            metadata = {}
            spectrum = None

        # 2. Populate Metadata Table
        if not metadata or len(metadata) == 0:
            self.table.setVisible(False)
            self.lbl_placeholder.setVisible(True)
            self.table.setRowCount(0)
        else:
            self.lbl_placeholder.setVisible(False)
            self.table.setVisible(True)
            self.table.setRowCount(0)
            for row_idx, (key, value) in enumerate(metadata.items()):
                self.table.insertRow(row_idx)
                key_item = QTableWidgetItem(str(key))
                key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_idx, 0, key_item)
                value_item = QTableWidgetItem(str(value))
                value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_idx, 1, value_item)

        # 3. Populate Custom Attributes (ONLY if item is a spectrum)
        if spectrum:
            self.lbl_label.setText(str(spectrum.label) if spectrum.label else "")
            self.lbl_color.setText(str(spectrum.color) if spectrum.color else "")
            self.lbl_xcorr.setText(str(spectrum.xcorrection_value))
            self.lbl_source.setText(str(spectrum.source_path) if spectrum.source_path else "")

            # Set normalizer factor string
            self.lbl_norm_factor.setText(str(getattr(spectrum, 'intensity_norm_factor', 1.0)))

            # Extract baseline attributes separately
            mode_str = ""
            coef_str = ""
            points_str = ""
            subtracted_str = ""
            try:
                if hasattr(spectrum, 'baseline') and spectrum.baseline is not None:
                    mode_str = str(getattr(spectrum.baseline, 'mode', ''))
                    coef_str = str(getattr(spectrum.baseline, 'coef', ''))
                    points_str = str(getattr(spectrum.baseline, 'points', ''))
                    subtracted_str = str(getattr(spectrum.baseline, 'is_subtracted', ''))
            except Exception:
                pass

            self.lbl_baseline_mode.setText(mode_str)
            self.lbl_baseline_coef.setText(coef_str)
            self.lbl_baseline_points.setText(points_str)
            self.lbl_baseline_subtracted.setText(subtracted_str)

            # Set normalizer factor spinbox value
            self.spin_norm_factor.setValue(getattr(spectrum, 'intensity_norm_factor', 1.0))

            # Enable controls
            self.group_custom.setEnabled(True)
            self.group_norm.setEnabled(True)
        else:
            # It's just a dict (e.g. Map selected) or None, clear custom attributes
            self.lbl_label.clear()
            self.lbl_color.clear()
            self.lbl_xcorr.clear()
            self.lbl_source.clear()
            self.lbl_norm_factor.clear()
            self.lbl_baseline_mode.clear()
            self.lbl_baseline_coef.clear()
            self.lbl_baseline_points.clear()
            self.lbl_baseline_subtracted.clear()

            self.spin_norm_factor.setValue(1.0)

    def clear_metadata(self):
        """Clear all displays."""
        self.table.setVisible(False)
        self.lbl_placeholder.setVisible(True)
        self.table.setRowCount(0)

        self.lbl_label.clear()
        self.lbl_color.clear()
        self.lbl_xcorr.clear()
        self.lbl_source.clear()
        self.lbl_norm_factor.clear()
        self.lbl_baseline_mode.clear()
        self.lbl_baseline_coef.clear()
        self.lbl_baseline_points.clear()
        self.lbl_baseline_subtracted.clear()

        self.spin_norm_factor.setValue(1.0)

        self.group_custom.setEnabled(False)
        self.group_norm.setEnabled(False)

    def apply_theme(self, theme: str = ""):
        """Update styling on theme change."""
        tok = current_tokens()
        self.lbl_placeholder.setStyleSheet(f"color: {tok.text_muted}; font-size: 14px; padding: 40px;")
