"""Options menu builder, view-option synchronization, and plot theme helpers for VSpectraViewer."""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from spectroview import (
    PLOT_POLICY_DARK,
    PLOT_POLICY_LIGHT,
)
from spectroview.i18n import (
    bind_combo_items,
    bind_placeholder,
    bind_text,
    bind_tooltip,
    combo_data_or_text,
    set_combo_by_data_or_text,
    tr,
)
from spectroview.view.components.common.customized_widgets import CustomizedPalette
from spectroview.view.components.spectra_viewer.common import (
    SPECTRA_COLOR_PALETTES,
    SPECTRA_CUSTOM_PALETTES,
    get_cached_style_dict,
)
from spectroview.view.theme import current_tokens


class SpectraOptionsMenuMixin:
    """Mixin encapsulating the two-column options menu and view option state sync."""

    def _wrap_menu_row(
        self, parent_widget: QWidget, label: str, widget: QWidget
    ) -> QWidget:
        """Build a compact label + control row for the 2-column options menu."""
        row_widget = QWidget(parent_widget)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 1, 2, 1)
        row_layout.setSpacing(6)
        lbl = QLabel(row_widget)
        bind_text(lbl, label)
        row_layout.addWidget(lbl)
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
    ):
        """Create a compact QCheckBox synchronized with a programmatic QAction."""
        act = QAction(menu)
        bind_text(act, text)
        act.setCheckable(True)
        act.setChecked(bool(checked))

        chk = QCheckBox(parent_widget)
        bind_text(chk, text)
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

    def _create_options_menu(self):
        """Build a compact 2-column 'More view options' menu matching VLiveMaps spacing."""
        menu = QMenu(self)
        panel = QWidget(menu)
        self._options_panel = panel

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

        # ── Left Column: Theme, Axes, Plot Style, Palette, Line/Dot Sizes & Copy Options ──
        self.cbb_theme = QComboBox(left_col)
        bind_combo_items(
            self.cbb_theme,
            [("Light Mode", "Light Mode"), ("Dark Mode", "Dark Mode")],
        )
        self.cbb_theme.currentIndexChanged.connect(self._apply_plot_style)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Plot Theme:", self.cbb_theme)
        )

        self.cbb_xaxis = QComboBox(left_col)
        bind_combo_items(
            self.cbb_xaxis,
            [
                ("Wavenumber (cm$^{-1}$)", "Wavenumber (cm$^{-1}$)"),
                ("Wavelength (nm)", "Wavelength (nm)"),
                ("Emission energy (eV)", "Emission energy (eV)"),
                ("Binding energy (eV)", "Binding energy (eV)"),
                ("Energy (keV)", "Energy (keV)"),
                ("Frequency (GHz)", "Frequency (GHz)"),
                (r"$2\theta$ (°)", r"$2\theta$ (°)"),
                ("Time (ns)", "Time (ns)"),
            ],
        )
        self.cbb_xaxis.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "X-axis:", self.cbb_xaxis)
        )

        self.cbb_yaxis = QComboBox(left_col)
        bind_combo_items(
            self.cbb_yaxis,
            [
                ("Intensity (a.u.)", "Intensity (a.u.)"),
                ("Intensity (a.u./s)", "Intensity (a.u./s)"),
            ],
        )
        self.cbb_yaxis.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Y-axis:", self.cbb_yaxis)
        )

        self.cbb_yscale = QComboBox(left_col)
        bind_combo_items(
            self.cbb_yscale,
            [("Linear", "Linear"), ("Log", "Log")],
        )
        self.cbb_yscale.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Y-scale:", self.cbb_yscale)
        )

        left_layout.addWidget(self._menu_h_separator(left_col))

        self.cbb_plotstyle = QComboBox(left_col)
        bind_combo_items(
            self.cbb_plotstyle,
            [("line", "line"), ("dot", "dot")],
        )
        self.cbb_plotstyle.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Spectrum plot style:", self.cbb_plotstyle)
        )

        self.cbb_color_palette = CustomizedPalette(
            palette_list=SPECTRA_COLOR_PALETTES,
            custom_palettes=SPECTRA_CUSTOM_PALETTES,
            parent=left_col,
        )
        self.cbb_color_palette.setCurrentText("DEFAULT_COLORS")
        bind_tooltip(
            self.cbb_color_palette,
            "Choose colors for the selected spectrum series. Custom colors "
            "set from the legend continue to take priority.",
        )
        self.cbb_color_palette.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Color palette:", self.cbb_color_palette)
        )

        self.spin_lw = QDoubleSpinBox(left_col)
        self.spin_lw.setRange(0.1, 5)
        self.spin_lw.setSingleStep(0.25)
        self.spin_lw.setValue(0.75)
        self.spin_lw.valueChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Line width:", self.spin_lw)
        )

        self.spin_dotsize = QDoubleSpinBox(left_col)
        self.spin_dotsize.setRange(0.5, 10)
        self.spin_dotsize.setSingleStep(0.5)
        self.spin_dotsize.setValue(3)
        self.spin_dotsize.valueChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Dot size:", self.spin_dotsize)
        )

        left_layout.addWidget(self._menu_h_separator(left_col))

        ratio_widget = QWidget(left_col)
        ratio_layout = QHBoxLayout(ratio_widget)
        ratio_layout.setContentsMargins(2, 1, 2, 1)
        ratio_layout.setSpacing(4)
        lbl_copy_size = QLabel(ratio_widget)
        bind_text(lbl_copy_size, "Copied figure size:")
        ratio_layout.addWidget(lbl_copy_size)
        ratio_layout.addStretch(1)

        self.width_entry = QLineEdit(ratio_widget)
        self.width_entry.setFixedWidth(40)
        self.width_entry.setText("5.5")
        self.width_entry.editingFinished.connect(self._emit_view_options)
        ratio_layout.addWidget(self.width_entry)

        self.height_entry = QLineEdit(ratio_widget)
        self.height_entry.setFixedWidth(40)
        self.height_entry.setText("4.0")
        self.height_entry.editingFinished.connect(self._emit_view_options)
        ratio_layout.addWidget(self.height_entry)
        left_layout.addWidget(ratio_widget)

        self.cbb_copy_theme = QComboBox(left_col)
        bind_combo_items(
            self.cbb_copy_theme,
            [("Light Mode", "Light Mode"), ("Dark Mode", "Dark Mode")],
        )
        self.cbb_copy_theme.currentIndexChanged.connect(self._emit_view_options)
        left_layout.addWidget(
            self._wrap_menu_row(left_col, "Copied figure theme:", self.cbb_copy_theme)
        )
        left_layout.addStretch(1)

        # ── Right Column: Display Toggles & Legend / Colorbar Settings ──
        self.act_raw, self.chk_raw = self._bind_menu_checkbox(
            right_col, menu, "Raw", False, lambda _v: self._emit_view_options()
        )
        right_layout.addWidget(self.chk_raw)

        self.act_bestfit_colorful, self.chk_bestfit_colorful = self._bind_menu_checkbox(
            right_col,
            menu,
            "Bestfit curve colorful",
            True,
            lambda _v: self._emit_view_options(),
        )
        right_layout.addWidget(self.chk_bestfit_colorful)

        self.act_show_peak_label, self.chk_show_peak_label = self._bind_menu_checkbox(
            right_col,
            menu,
            "Show peak label",
            False,
            lambda _v: self._emit_view_options(),
        )
        right_layout.addWidget(self.chk_show_peak_label)

        self.act_residual, self.chk_residual = self._bind_menu_checkbox(
            right_col, menu, "Residual", False, lambda _v: self._emit_view_options()
        )
        right_layout.addWidget(self.chk_residual)

        self.act_grid, self.chk_grid = self._bind_menu_checkbox(
            right_col, menu, "Grid", False, lambda _v: self._emit_view_options()
        )
        right_layout.addWidget(self.chk_grid)

        self.act_noise_level, self.chk_noise_level = self._bind_menu_checkbox(
            right_col,
            menu,
            "Show noise level",
            False,
            lambda _v: self._emit_view_options(),
        )
        right_layout.addWidget(self.chk_noise_level)

        right_layout.addWidget(self._menu_h_separator(right_col))

        self.spin_max_legend_items = QSpinBox(right_col)
        self.spin_max_legend_items.setRange(1, 1000)
        self.spin_max_legend_items.setValue(15)
        self.spin_max_legend_items.valueChanged.connect(self._emit_view_options)
        right_layout.addWidget(
            self._wrap_menu_row(
                right_col, "Max legend items:", self.spin_max_legend_items
            )
        )
        self.spin_max_overlays = self.spin_max_legend_items

        self.act_legend_outside, self.chk_legend_outside = self._bind_menu_checkbox(
            right_col,
            menu,
            "Move legend box outside",
            False,
            lambda _v: self._emit_view_options(),
        )
        right_layout.addWidget(self.chk_legend_outside)

        self.act_legend_colorbar, self.chk_legend_colorbar = self._bind_menu_checkbox(
            right_col,
            menu,
            "Show colorbar instead of legend",
            False,
            self._on_legend_colorbar_toggled,
        )
        right_layout.addWidget(self.chk_legend_colorbar)

        # Colorbar sub-options (shown only when "Show colorbar instead of legend" is checked)
        self.cbb_colorbar_param = QComboBox(right_col)
        self.cbb_colorbar_param.addItem(tr("Index"), None)
        bind_tooltip(
            self.cbb_colorbar_param,
            "Colorbar parameter: choose parameter for the colorbar.\n"
            "Index = spectrum order, or select a token parsed from filenames.",
        )
        self.cbb_colorbar_param.currentIndexChanged.connect(
            self._on_colorbar_param_changed
        )
        self.row_colorbar_param = self._wrap_menu_row(
            right_col, "Colorbar parameter:", self.cbb_colorbar_param
        )
        right_layout.addWidget(self.row_colorbar_param)

        self.le_colorbar_label = QLineEdit(right_col)
        bind_placeholder(self.le_colorbar_label, "Colorbar label")
        bind_tooltip(
            self.le_colorbar_label,
            "Custom label for the spectrum colorbar.\n"
            "Leave empty for the automatic token name.\n"
            "Example: Temperature (°C)",
        )
        self.le_colorbar_label.textChanged.connect(self._on_heatmap_ylabel_changed)
        self.row_colorbar_label = self._wrap_menu_row(
            right_col, "Colorbar label:", self.le_colorbar_label
        )
        right_layout.addWidget(self.row_colorbar_label)
        right_layout.addStretch(1)

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

        # Hide colorbar parameter & label rows until colorbar mode is checked
        self._update_colorbar_options_visibility(False)

        return menu

    # ─────────────────────────────────────────
    # Signal emitters & Synchronization between spectra and maps workspaces
    # ─────────────────────────────────────────
    def get_options_state(self):
        """Get the complete state of all view options."""
        cb_param = "Index"
        if hasattr(self, "cbb_colorbar_param"):
            cb_param = (
                "Index"
                if self.cbb_colorbar_param.currentData() is None
                and self.cbb_colorbar_param.currentIndex() == 0
                else self.cbb_colorbar_param.currentText()
            )
        hm_yaxis = "Index"
        if hasattr(self, "cbb_heatmap_yaxis"):
            hm_yaxis = (
                "Index"
                if self.cbb_heatmap_yaxis.currentData() is None
                and self.cbb_heatmap_yaxis.currentIndex() == 0
                else self.cbb_heatmap_yaxis.currentText()
            )
        return {
            "theme": combo_data_or_text(self.cbb_theme) if hasattr(self, "cbb_theme") else "Light Mode",
            "xaxis": combo_data_or_text(self.cbb_xaxis) if hasattr(self, "cbb_xaxis") else "",
            "yaxis": combo_data_or_text(self.cbb_yaxis) if hasattr(self, "cbb_yaxis") else "",
            "yscale": combo_data_or_text(self.cbb_yscale) if hasattr(self, "cbb_yscale") else "Linear",
            "plotstyle": combo_data_or_text(self.cbb_plotstyle) if hasattr(self, "cbb_plotstyle") else "line",
            "color_palette": self.cbb_color_palette.currentText() if hasattr(self, "cbb_color_palette") else "DEFAULT_COLORS",
            "lw": self.spin_lw.value() if hasattr(self, "spin_lw") else 0.75,
            "dotsize": self.spin_dotsize.value() if hasattr(self, "spin_dotsize") else 3.0,
            "raw": self.act_raw.isChecked() if hasattr(self, "act_raw") else False,
            "bestfit_colorful": self.act_bestfit_colorful.isChecked() if hasattr(self, "act_bestfit_colorful") else True,
            "show_peak_label": self.act_show_peak_label.isChecked() if hasattr(self, "act_show_peak_label") else False,
            "residual": self.act_residual.isChecked() if hasattr(self, "act_residual") else False,
            "grid": self.act_grid.isChecked() if hasattr(self, "act_grid") else False,
            "noise_level": self.act_noise_level.isChecked() if hasattr(self, "act_noise_level") else False,
            "width": self.width_entry.text() if hasattr(self, "width_entry") else "5.5",
            "height": self.height_entry.text() if hasattr(self, "height_entry") else "4.0",
            "legend": self.btn_legend.isChecked() if hasattr(self, "btn_legend") else False,
            "legend_outside": self.act_legend_outside.isChecked() if hasattr(self, "act_legend_outside") else False,
            "legend_colorbar": self.act_legend_colorbar.isChecked() if hasattr(self, "act_legend_colorbar") else False,
            "colorbar_param": cb_param,
            "colorbar_label": self.le_colorbar_label.text() if hasattr(self, "le_colorbar_label") else "",
            "bestfit": self.btn_bestfit.isChecked() if hasattr(self, "btn_bestfit") else True,
            "max_legend_items": self.spin_max_legend_items.value() if hasattr(self, "spin_max_legend_items") else 15,
            "copy_fig_theme": combo_data_or_text(self.cbb_copy_theme) if hasattr(self, "cbb_copy_theme") else "Light Mode",
            "heatmap": self.btn_heatmap.isChecked() if hasattr(self, "btn_heatmap") else False,
            "view_3d": self.btn_3d.isChecked() if hasattr(self, "btn_3d") else False,
            "heatmap_yaxis": hm_yaxis,
            "heatmap_ylabel": self.le_heatmap_ylabel.text() if hasattr(self, "le_heatmap_ylabel") else "",
            "heatmap_interp": self.spin_heatmap_interp.value() if hasattr(self, "spin_heatmap_interp") else 300,
        }

    def set_options_state(self, state):
        """Set the complete state of all view options without causing infinite loops."""
        if not hasattr(self, "cbb_theme"):
            return

        current_state = self.get_options_state()
        if current_state == state and getattr(self, "_has_applied_plot_style", False):
            return

        self._is_setting_options = True

        def _update(widget, setter, value):
            if value is None:
                return
            widget.blockSignals(True)
            setter(value)
            widget.blockSignals(False)

        theme_val = state.get("theme")
        if theme_val == "Soft Dark Mode":
            theme_val = "Dark Mode"
        _update(self.cbb_theme, lambda v: set_combo_by_data_or_text(self.cbb_theme, v), theme_val)
        _update(self.cbb_xaxis, lambda v: set_combo_by_data_or_text(self.cbb_xaxis, v), state.get("xaxis"))
        _update(self.cbb_yaxis, lambda v: set_combo_by_data_or_text(self.cbb_yaxis, v), state.get("yaxis"))
        _update(self.cbb_yscale, lambda v: set_combo_by_data_or_text(self.cbb_yscale, v), state.get("yscale"))
        _update(self.cbb_plotstyle, lambda v: set_combo_by_data_or_text(self.cbb_plotstyle, v), state.get("plotstyle"))
        if hasattr(self, "cbb_color_palette"):
            _update(
                self.cbb_color_palette,
                self.cbb_color_palette.setCurrentText,
                state.get("color_palette", "DEFAULT_COLORS"),
            )
        _update(self.spin_lw, self.spin_lw.setValue, state.get("lw"))
        _update(self.spin_dotsize, self.spin_dotsize.setValue, state.get("dotsize"))
        _update(self.act_raw, self.act_raw.setChecked, state.get("raw"))
        _update(
            self.act_bestfit_colorful,
            self.act_bestfit_colorful.setChecked,
            state.get("bestfit_colorful"),
        )
        _update(
            self.act_show_peak_label,
            self.act_show_peak_label.setChecked,
            state.get("show_peak_label"),
        )
        _update(self.act_residual, self.act_residual.setChecked, state.get("residual"))
        _update(self.act_grid, self.act_grid.setChecked, state.get("grid"))
        if hasattr(self, "act_noise_level"):
            _update(
                self.act_noise_level,
                self.act_noise_level.setChecked,
                state.get("noise_level"),
            )
        _update(self.width_entry, self.width_entry.setText, state.get("width"))
        _update(self.height_entry, self.height_entry.setText, state.get("height"))
        _update(self.btn_legend, self.btn_legend.setChecked, state.get("legend"))
        if hasattr(self, "act_legend_outside"):
            _update(
                self.act_legend_outside,
                self.act_legend_outside.setChecked,
                state.get("legend_outside", False),
            )
        if hasattr(self, "act_legend_colorbar"):
            _update(
                self.act_legend_colorbar,
                self.act_legend_colorbar.setChecked,
                state.get("legend_colorbar", False),
            )
        if hasattr(self, "cbb_colorbar_param"):
            cb_param = state.get("colorbar_param", "Index")
            if cb_param:
                if cb_param in ("Index", tr("Index")) and self.cbb_colorbar_param.count() > 0:
                    _update(
                        self.cbb_colorbar_param,
                        self.cbb_colorbar_param.setCurrentIndex,
                        0,
                    )
                else:
                    if self.cbb_colorbar_param.findText(cb_param) < 0:
                        self.cbb_colorbar_param.blockSignals(True)
                        self.cbb_colorbar_param.addItem(cb_param, None)
                        self.cbb_colorbar_param.blockSignals(False)
                    _update(
                        self.cbb_colorbar_param,
                        self.cbb_colorbar_param.setCurrentText,
                        cb_param,
                    )
            self._record_token_selection(
                self.cbb_colorbar_param, self._colorbar_param_selection
            )
            if self._tensor_data:
                self._refresh_colorbar_param_choices()
        if hasattr(self, "le_colorbar_label"):
            _update(
                self.le_colorbar_label,
                self.le_colorbar_label.setText,
                state.get("colorbar_label", ""),
            )
        _update(self.btn_bestfit, self.btn_bestfit.setChecked, state.get("bestfit"))
        if hasattr(self, "spin_max_legend_items"):
            _update(
                self.spin_max_legend_items,
                self.spin_max_legend_items.setValue,
                state.get("max_legend_items", 15),
            )
        if hasattr(self, "cbb_copy_theme"):
            _update(
                self.cbb_copy_theme,
                lambda v: set_combo_by_data_or_text(self.cbb_copy_theme, v),
                state.get("copy_fig_theme", "Light Mode"),
            )
        want_heatmap = bool(state.get("heatmap", False))
        want_3d = bool(state.get("view_3d", False)) and not want_heatmap
        if hasattr(self, "btn_heatmap"):
            _update(self.btn_heatmap, self.btn_heatmap.setChecked, want_heatmap)
        if hasattr(self, "btn_3d"):
            _update(self.btn_3d, self.btn_3d.setChecked, want_3d)
        mode_changed = (
            getattr(self, "_heatmap_mode", False) != want_heatmap
            or getattr(self, "_3d_mode", False) != want_3d
        )
        self._heatmap_mode = want_heatmap
        self._3d_mode = want_3d
        if want_3d and mode_changed:
            self._reset_3d_view = True
        if mode_changed:
            self._sync_mode_toolbar_state()

        if hasattr(self, "cbb_heatmap_yaxis"):
            hm_yaxis = state.get("heatmap_yaxis", "Index")
            if hm_yaxis:
                if hm_yaxis in ("Index", tr("Index")) and self.cbb_heatmap_yaxis.count() > 0:
                    _update(
                        self.cbb_heatmap_yaxis,
                        self.cbb_heatmap_yaxis.setCurrentIndex,
                        0,
                    )
                else:
                    if self.cbb_heatmap_yaxis.findText(hm_yaxis) < 0:
                        self.cbb_heatmap_yaxis.blockSignals(True)
                        self.cbb_heatmap_yaxis.addItem(hm_yaxis, None)
                        self.cbb_heatmap_yaxis.blockSignals(False)
                    _update(
                        self.cbb_heatmap_yaxis,
                        self.cbb_heatmap_yaxis.setCurrentText,
                        hm_yaxis,
                    )
            self._record_token_selection(
                self.cbb_heatmap_yaxis, self._heatmap_yaxis_selection
            )
            if self._tensor_data and (
                getattr(self, "_heatmap_mode", False)
                or getattr(self, "_3d_mode", False)
            ):
                self._refresh_heatmap_yaxis_choices()
        if hasattr(self, "le_heatmap_ylabel"):
            _update(
                self.le_heatmap_ylabel,
                self.le_heatmap_ylabel.setText,
                state.get("heatmap_ylabel", ""),
            )
        if hasattr(self, "spin_heatmap_interp"):
            _update(
                self.spin_heatmap_interp,
                self.spin_heatmap_interp.setValue,
                state.get("heatmap_interp", 300),
            )

        self._sync_menu_checkboxes()
        self._apply_plot_style()
        self._plot()
        self._is_setting_options = False

    def _sync_menu_checkboxes(self):
        """Synchronize 2-column menu QCheckBox widgets and conditional rows with QAction states."""
        pairs = (
            ("act_raw", "chk_raw"),
            ("act_bestfit_colorful", "chk_bestfit_colorful"),
            ("act_show_peak_label", "chk_show_peak_label"),
            ("act_residual", "chk_residual"),
            ("act_grid", "chk_grid"),
            ("act_noise_level", "chk_noise_level"),
            ("act_legend_outside", "chk_legend_outside"),
            ("act_legend_colorbar", "chk_legend_colorbar"),
        )
        for act_attr, chk_attr in pairs:
            act = getattr(self, act_attr, None)
            chk = getattr(self, chk_attr, None)
            if (
                act is not None
                and chk is not None
                and chk.isChecked() != act.isChecked()
            ):
                chk.blockSignals(True)
                chk.setChecked(act.isChecked())
                chk.blockSignals(False)
        if hasattr(self, "act_legend_colorbar"):
            self._update_colorbar_options_visibility(
                self.act_legend_colorbar.isChecked()
            )

    def _apply_plot_style(self):
        self._has_applied_plot_style = True
        style_name = combo_data_or_text(self.cbb_theme)
        if style_name == "Dark Mode":
            style_path = PLOT_POLICY_DARK
        else:
            style_path = PLOT_POLICY_LIGHT

        # Parse the style file from cached dict without modifying global rcParams
        style_dict = get_cached_style_dict(style_path)

        fig_bg = style_dict.get("figure.facecolor", "white")
        self.figure.patch.set_facecolor(fig_bg)
        if hasattr(self, "canvas_card"):
            tok = current_tokens()
            fig_hex = (
                fig_bg
                if str(fig_bg).startswith("#") or str(fig_bg).isalpha()
                else f"#{fig_bg}"
            )
            self.canvas_card.setStyleSheet(
                f"QFrame#canvasCard {{ background-color: {fig_hex}; border: 1px solid {tok.border}; border-radius: 8px; }}"
            )
        self.figure.patch.set_edgecolor(style_dict.get("figure.edgecolor", "white"))
        if getattr(self.ax, "name", None) == "3d":
            self._apply_3d_axes_style()
        else:
            self._apply_2d_axes_style(style_dict)

        if not getattr(self, "_is_copying", False):
            self.plotStyleChanged.emit()
            if not getattr(self, "_is_setting_options", False):
                self.allOptionsSyncChanged.emit(self.get_options_state())
            self._plot()

    def _emit_view_options(self):
        if getattr(self, "_is_setting_options", False):
            return

        # Sync options with other viewers
        self.allOptionsSyncChanged.emit(self.get_options_state())

        # Update plot immediately whenever a view option changes
        self._plot()
