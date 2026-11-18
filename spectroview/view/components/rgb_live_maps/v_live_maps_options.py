"""Options menu builder and control synchronization mixin for VLiveMaps."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PySide6.QtGui import QAction, QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSpinBox,
    QWidget,
    QWidgetAction,
)

from spectroview.view.components.rgb_live_maps.v_live_maps_legend_editor import (
    LEGEND_LOCATIONS,
)


class LiveMapsOptionsMenuMixin:
    """Two-column 'More view options' menu construction and widget synchronization."""

    def _wrap_menu_row(self, parent_widget: QWidget, label: str, widget: QWidget) -> QWidget:
        """Build a compact label + control row with compact vertical margins."""
        row_widget = QWidget(parent_widget)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 1, 2, 1)
        row_layout.setSpacing(6)
        row_layout.addWidget(QLabel(label, row_widget))
        row_layout.addWidget(widget, stretch=1)
        return row_widget

    @staticmethod
    def _menu_h_separator(parent_widget: QWidget) -> QFrame:
        """Return a hairline horizontal separator for the 2-column options menu."""
        line = QFrame(parent_widget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setFixedHeight(1)
        return line

    def _bind_menu_checkbox(
        self,
        parent_widget: QWidget,
        menu: QMenu,
        text: str,
        checked: bool,
        on_toggled,
    ) -> Tuple[QAction, QCheckBox]:
        """Create a compact QCheckBox synchronized with a programmatic QAction."""
        act = QAction(text, menu)
        act.setCheckable(True)
        act.setChecked(bool(checked))

        chk = QCheckBox(text, parent_widget)
        chk.setChecked(bool(checked))

        def _on_act_toggled(val: bool):
            val = bool(val)
            if chk.isChecked() != val:
                chk.blockSignals(True)
                chk.setChecked(val)
                chk.blockSignals(False)
            on_toggled(val)

        def _on_chk_toggled(val: bool):
            val = bool(val)
            if act.isChecked() != val:
                act.setChecked(val)

        act.toggled.connect(_on_act_toggled)
        chk.toggled.connect(_on_chk_toggled)
        return act, chk

    @staticmethod
    def _sanitize_dim_text(line_edit: QLineEdit, fallback: str) -> str:
        """Validate positive copied-figure inch dimension text."""
        raw = str(line_edit.text() or "").strip().replace(",", ".")
        try:
            val = float(raw)
            if 0.5 <= val <= 50.0 and np.isfinite(val):
                formatted = f"{val:g}" if "." not in raw else raw
                if line_edit.text() != formatted:
                    line_edit.blockSignals(True)
                    line_edit.setText(formatted)
                    line_edit.blockSignals(False)
                return formatted
        except (TypeError, ValueError):
            pass
        line_edit.blockSignals(True)
        line_edit.setText(fallback)
        line_edit.blockSignals(False)
        return fallback

    def _commit_copy_size_entries(self):
        """Commit Copied figure size inputs when editing finishes or menu closes."""
        if hasattr(self, "width_entry"):
            w = self._sanitize_dim_text(self.width_entry, "5.5")
            self._update_option("copy_fig_width", w)
        if hasattr(self, "height_entry"):
            h = self._sanitize_dim_text(self.height_entry, "4.0")
            self._update_option("copy_fig_height", h)

    def _create_options_menu(self) -> QMenu:
        """Build a compact 2-column 'More view options' menu matching VSpectraViewer spacing."""
        from PySide6.QtWidgets import QVBoxLayout

        menu = QMenu(self)
        panel = QWidget(menu)
        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(6, 4, 6, 4)
        outer_layout.setSpacing(2)

        cols_layout = QHBoxLayout()
        cols_layout.setContentsMargins(0, 0, 0, 0)
        cols_layout.setSpacing(8)

        left_col = QWidget(panel)
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        right_col = QWidget(panel)
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # ── Left Column: Plot theme, Axis scales, Style/Sizes & Copy size/theme ──
        self.cbb_plot_theme = QComboBox(left_col)
        self.cbb_plot_theme.addItems(["Light Mode", "Dark Mode"])
        self.cbb_plot_theme.setCurrentText(self._view_options["plot_theme"])
        self.cbb_plot_theme.currentTextChanged.connect(
            lambda txt: self._update_option("plot_theme", txt)
        )
        left_layout.addWidget(self._wrap_menu_row(left_col, "Plot theme:", self.cbb_plot_theme))

        self.cbb_xscale = QComboBox(left_col)
        self.cbb_xscale.addItems(["Linear", "Log"])
        self.cbb_xscale.currentTextChanged.connect(
            lambda txt: self._update_option("xscale", txt.lower())
        )
        left_layout.addWidget(self._wrap_menu_row(left_col, "X-scale:", self.cbb_xscale))

        self.cbb_yscale = QComboBox(left_col)
        self.cbb_yscale.addItems(["Linear", "Log"])
        self.cbb_yscale.currentTextChanged.connect(
            lambda txt: self._update_option("yscale", txt.lower())
        )
        left_layout.addWidget(self._wrap_menu_row(left_col, "Y-scale:", self.cbb_yscale))

        self.cbb_zscale = QComboBox(left_col)
        self.cbb_zscale.addItems(["Linear", "Log"])
        self.cbb_zscale.currentTextChanged.connect(
            lambda txt: self._update_option("zscale", txt.lower())
        )
        left_layout.addWidget(self._wrap_menu_row(left_col, "Color (Z) scale:", self.cbb_zscale))

        left_layout.addWidget(self._menu_h_separator(left_col))

        self._style_labels_to_keys = {
            "Line": "line",
            "Line + Dots": "line+dots",
            "Dot": "dot",
        }
        self._style_keys_to_labels = {v: k for k, v in self._style_labels_to_keys.items()}
        self.cbb_plot_style = QComboBox(left_col)
        self.cbb_plot_style.addItems(list(self._style_labels_to_keys.keys()))
        self.cbb_plot_style.setCurrentText(
            self._style_keys_to_labels.get(self._view_options["plot_style"], "Line")
        )
        self.cbb_plot_style.currentTextChanged.connect(
            lambda txt: self._update_option(
                "plot_style", self._style_labels_to_keys.get(txt, "line")
            )
        )
        left_layout.addWidget(self._wrap_menu_row(left_col, "Plot style:", self.cbb_plot_style))

        self.spb_linewidth = QDoubleSpinBox(left_col)
        self.spb_linewidth.setRange(0.5, 6.0)
        self.spb_linewidth.setSingleStep(0.2)
        self.spb_linewidth.setDecimals(1)
        self.spb_linewidth.setKeyboardTracking(False)
        self.spb_linewidth.setValue(self._view_options["line_width"])
        self.spb_linewidth.valueChanged.connect(
            lambda val: self._update_option("line_width", float(val))
        )
        left_layout.addWidget(self._wrap_menu_row(left_col, "Line width:", self.spb_linewidth))

        self.spb_dotsize = QDoubleSpinBox(left_col)
        self.spb_dotsize.setRange(1.0, 14.0)
        self.spb_dotsize.setSingleStep(0.5)
        self.spb_dotsize.setDecimals(1)
        self.spb_dotsize.setKeyboardTracking(False)
        self.spb_dotsize.setValue(self._view_options["dot_size"])
        self.spb_dotsize.valueChanged.connect(
            lambda val: self._update_option("dot_size", float(val))
        )
        left_layout.addWidget(self._wrap_menu_row(left_col, "Dot size:", self.spb_dotsize))

        self.spb_text_fontsize = QSpinBox(left_col)
        self.spb_text_fontsize.setRange(6, 24)
        self.spb_text_fontsize.setSuffix(" pt")
        self.spb_text_fontsize.setKeyboardTracking(False)
        self.spb_text_fontsize.setValue(self._view_options["text_fontsize"])
        self.spb_text_fontsize.valueChanged.connect(
            lambda val: self._update_option("text_fontsize", int(val))
        )
        left_layout.addWidget(self._wrap_menu_row(left_col, "Text size:", self.spb_text_fontsize))

        left_layout.addWidget(self._menu_h_separator(left_col))

        ratio_widget = QWidget(left_col)
        ratio_layout = QHBoxLayout(ratio_widget)
        ratio_layout.setContentsMargins(2, 1, 2, 1)
        ratio_layout.setSpacing(4)
        ratio_layout.addWidget(QLabel("Copied figure size:", ratio_widget))
        ratio_layout.addStretch(1)

        dim_validator = QDoubleValidator(0.5, 50.0, 2, ratio_widget)
        self.width_entry = QLineEdit(ratio_widget)
        self.width_entry.setFixedWidth(40)
        self.width_entry.setValidator(dim_validator)
        self.width_entry.setText(self._view_options["copy_fig_width"])
        self.width_entry.setToolTip("Copied figure width (inches)")
        self.width_entry.editingFinished.connect(self._commit_copy_size_entries)
        ratio_layout.addWidget(self.width_entry)

        self.height_entry = QLineEdit(ratio_widget)
        self.height_entry.setFixedWidth(40)
        self.height_entry.setValidator(dim_validator)
        self.height_entry.setText(self._view_options["copy_fig_height"])
        self.height_entry.setToolTip("Copied figure height (inches)")
        self.height_entry.editingFinished.connect(self._commit_copy_size_entries)
        ratio_layout.addWidget(self.height_entry)
        left_layout.addWidget(ratio_widget)

        self.cbb_copy_theme = QComboBox(left_col)
        self.cbb_copy_theme.addItems(["Light Mode", "Dark Mode"])
        self.cbb_copy_theme.setCurrentText(self._view_options["copy_fig_theme"])
        self.cbb_copy_theme.currentTextChanged.connect(
            lambda txt: self._update_option("copy_fig_theme", txt)
        )
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Copied figure theme:", self.cbb_copy_theme)
        )
        left_layout.addStretch(1)

        # ── Right Column: Legend, Display Aids & Reset ──
        self.act_show_legend, self.chk_show_legend = self._bind_menu_checkbox(
            right_col, menu, "Show legend", self._view_options["show_legend"],
            self._on_legend_toggled,
        )
        right_layout.addWidget(self.chk_show_legend)

        self.cbb_legend_loc = QComboBox(right_col)
        self.cbb_legend_loc.addItems(LEGEND_LOCATIONS)
        self.cbb_legend_loc.setCurrentText(self._view_options["legend_loc"])
        self.cbb_legend_loc.currentTextChanged.connect(
            lambda txt: self._update_option("legend_loc", txt)
        )
        right_layout.addWidget(self._wrap_menu_row(right_col, "Legend location:", self.cbb_legend_loc))

        self.spb_legend_ncol = QSpinBox(right_col)
        self.spb_legend_ncol.setRange(1, 4)
        self.spb_legend_ncol.setKeyboardTracking(False)
        self.spb_legend_ncol.setValue(self._view_options["legend_ncol"])
        self.spb_legend_ncol.valueChanged.connect(
            lambda val: self._update_option("legend_ncol", int(val))
        )
        right_layout.addWidget(self._wrap_menu_row(right_col, "Legend columns:", self.spb_legend_ncol))

        self.spb_legend_fontsize = QSpinBox(right_col)
        self.spb_legend_fontsize.setRange(6, 16)
        self.spb_legend_fontsize.setKeyboardTracking(False)
        self.spb_legend_fontsize.setValue(self._view_options["legend_fontsize"])
        self.spb_legend_fontsize.valueChanged.connect(
            lambda val: self._update_option("legend_fontsize", int(val))
        )
        right_layout.addWidget(
            self._wrap_menu_row(right_col, "Legend font size:", self.spb_legend_fontsize)
        )

        self.act_edit_legend = QAction("Edit legend labels...", menu)
        self.act_edit_legend.triggered.connect(self.open_legend_editor)
        self.btn_edit_legend = QPushButton("Edit legend labels...", right_col)
        self.btn_edit_legend.setProperty("variant", "ghost")
        self.btn_edit_legend.setStyleSheet("text-align: left; padding: 3px 6px;")
        self.btn_edit_legend.clicked.connect(
            lambda: (menu.close(), self.act_edit_legend.trigger())
        )
        right_layout.addWidget(self.btn_edit_legend)

        right_layout.addWidget(self._menu_h_separator(right_col))

        self.act_show_figure_label, self.chk_show_figure_label = self._bind_menu_checkbox(
            right_col, menu, "Show figure label", self._view_options["show_figure_label"],
            lambda chk: self._update_option("show_figure_label", bool(chk)),
        )
        self.chk_show_figure_label.setToolTip(
            "Show or hide the title label above each RGB plot canvas"
        )
        right_layout.addWidget(self.chk_show_figure_label)

        self.act_show_grid, self.chk_show_grid = self._bind_menu_checkbox(
            right_col, menu, "Show grid", self._view_options["show_grid"],
            lambda chk: self._update_option("show_grid", bool(chk)),
        )
        right_layout.addWidget(self.chk_show_grid)

        self.act_share_y, self.chk_share_y = self._bind_menu_checkbox(
            right_col, menu, "Share Y-axis (1D)", self._view_options["share_y"],
            lambda chk: self._update_option("share_y", bool(chk)),
        )
        right_layout.addWidget(self.chk_share_y)

        self.act_smoothing, self.chk_smoothing = self._bind_menu_checkbox(
            right_col, menu, "Heatmap smoothing", self._view_options["heatmap_smoothing"],
            lambda chk: self._update_option("heatmap_smoothing", bool(chk)),
        )
        right_layout.addWidget(self.chk_smoothing)

        self.act_show_colorbar, self.chk_show_colorbar = self._bind_menu_checkbox(
            right_col, menu, "Show colorbars", self._view_options["show_colorbar"],
            lambda chk: self._update_option("show_colorbar", bool(chk)),
        )
        right_layout.addWidget(self.chk_show_colorbar)

        right_layout.addStretch(1)
        right_layout.addWidget(self._menu_h_separator(right_col))

        self.act_reset_options = QAction("Reset view options", menu)
        self.act_reset_options.triggered.connect(self.reset_view_options)
        self.btn_reset_options = QPushButton("Reset view options", right_col)
        self.btn_reset_options.setProperty("variant", "ghost")
        self.btn_reset_options.setStyleSheet("text-align: left; padding: 3px 6px;")
        self.btn_reset_options.clicked.connect(
            lambda: (menu.close(), self.act_reset_options.trigger())
        )
        right_layout.addWidget(self.btn_reset_options)

        v_sep = QFrame(panel)
        v_sep.setFrameShape(QFrame.VLine)
        v_sep.setFrameShadow(QFrame.Sunken)

        cols_layout.addWidget(left_col, stretch=1)
        cols_layout.addWidget(v_sep)
        cols_layout.addWidget(right_col, stretch=1)
        outer_layout.addLayout(cols_layout)

        container_action = QWidgetAction(menu)
        container_action.setDefaultWidget(panel)
        menu.addAction(container_action)
        menu.aboutToHide.connect(self._commit_copy_size_entries)

        return menu

    def _sync_options_controls(self):
        """Synchronize menu widgets and toolbar toggles to self._view_options."""
        opts = self._view_options
        widgets = (
            self.cbb_plot_theme,
            self.cbb_xscale, self.cbb_yscale, self.cbb_zscale,
            self.cbb_plot_style, self.spb_linewidth, self.spb_dotsize,
            self.spb_text_fontsize,
            self.width_entry, self.height_entry, self.cbb_copy_theme,
            self.btn_legend, self.act_show_legend, self.chk_show_legend,
            self.cbb_legend_loc, self.spb_legend_ncol, self.spb_legend_fontsize,
            self.act_show_figure_label, self.chk_show_figure_label,
            self.act_show_grid, self.chk_show_grid,
            self.act_share_y, self.chk_share_y,
            self.act_smoothing, self.chk_smoothing,
            self.act_show_colorbar, self.chk_show_colorbar,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            self.cbb_plot_theme.setCurrentText(opts["plot_theme"])
            self.cbb_xscale.setCurrentText("Log" if opts["xscale"] == "log" else "Linear")
            self.cbb_yscale.setCurrentText("Log" if opts["yscale"] == "log" else "Linear")
            self.cbb_zscale.setCurrentText("Log" if opts["zscale"] == "log" else "Linear")
            self.cbb_plot_style.setCurrentText(
                self._style_keys_to_labels.get(opts["plot_style"], "Line")
            )
            self.spb_linewidth.setValue(float(opts["line_width"]))
            self.spb_dotsize.setValue(float(opts["dot_size"]))
            self.spb_text_fontsize.setValue(int(opts["text_fontsize"]))
            self.width_entry.setText(str(opts["copy_fig_width"]))
            self.height_entry.setText(str(opts["copy_fig_height"]))
            self.cbb_copy_theme.setCurrentText(opts["copy_fig_theme"])
            self.btn_legend.setChecked(bool(opts["show_legend"]))
            self.act_show_legend.setChecked(bool(opts["show_legend"]))
            self.chk_show_legend.setChecked(bool(opts["show_legend"]))
            loc = opts["legend_loc"]
            if self.cbb_legend_loc.findText(loc) >= 0:
                self.cbb_legend_loc.setCurrentText(loc)
            self.spb_legend_ncol.setValue(int(opts["legend_ncol"]))
            self.spb_legend_fontsize.setValue(int(opts["legend_fontsize"]))
            self.act_show_figure_label.setChecked(bool(opts["show_figure_label"]))
            self.chk_show_figure_label.setChecked(bool(opts["show_figure_label"]))
            self.act_show_grid.setChecked(bool(opts["show_grid"]))
            self.chk_show_grid.setChecked(bool(opts["show_grid"]))
            self.act_share_y.setChecked(bool(opts["share_y"]))
            self.chk_share_y.setChecked(bool(opts["share_y"]))
            self.act_smoothing.setChecked(bool(opts["heatmap_smoothing"]))
            self.chk_smoothing.setChecked(bool(opts["heatmap_smoothing"]))
            self.act_show_colorbar.setChecked(bool(opts["show_colorbar"]))
            self.chk_show_colorbar.setChecked(bool(opts["show_colorbar"]))
        finally:
            for w in widgets:
                w.blockSignals(False)
