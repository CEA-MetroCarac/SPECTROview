"""Live Maps Tab View component for SPECTROview.

Displays 3 live-updating panels corresponding to Red, Blue, and Green
spectral range selectors:
- In Maps workspace ('map' mode): renders 2D spatial heatmaps with colormap
  selection for 2D maps, or line plots along the profile for 1D maps (profiles).
- In Spectra workspace ('spectra' mode): renders 1D parameter variation graphs across selected spectra.

The Green panel can alternatively display the Red/Blue intensity ratio (R/B).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spectroview import DEFAULT_COLORS
from spectroview.model.spectra.spectral_parameters import (
    SELECTOR_BLUE,
    SELECTOR_GREEN,
    SELECTOR_KEYS,
    SELECTOR_RED,
)
from spectroview.view.components.rgb_live_maps.common import (
    AVAILABLE_PALETTES,
    DEFAULT_PALETTES,
    _copy_view_options,
    _get_cached_style_dict,
    _normalize_1d_values,
    _normalize_plot_theme,
    _parse_norm_bound,
    _resolve_style_path,
)
from spectroview.view.components.rgb_live_maps.v_live_map_subpanel import LiveMapSubpanel
from spectroview.view.components.rgb_live_maps.v_live_maps_legend_editor import (
    LEGEND_LOCATIONS,
    LiveMapsLegendEditorDialog,
)
from spectroview.view.components.rgb_live_maps.v_live_maps_options import (
    LiveMapsOptionsMenuMixin,
)
from spectroview.view.theme import Icons, current_tokens


class VLiveMaps(LiveMapsOptionsMenuMixin, QWidget):
    """Live Maps Tab View coordinating the 3 spectral range subpanels."""

    parameter_change_requested = Signal(str, str)  # (color_id, parameter)
    active_2d_map_requested = Signal(str)          # active 2D map name from switcher
    view_options_changed = Signal(dict)            # emitted on user/reset view option edits
    spectra_selected = Signal(list)                # (X, Y) spectrum points clicked/dragged in Live Maps
    copy_data_requested = Signal(str)              # color_id when user Ctrl+Clicks a subpanel Copy button

    def __init__(self, parent=None, mode: str = "map"):
        super().__init__(parent)
        self.setObjectName("LiveMapsTab")
        self.mode = mode

        self.panels: Dict[str, LiveMapSubpanel] = {}
        self._active_states: Dict[str, bool] = {
            SELECTOR_RED: False,
            SELECTOR_BLUE: False,
            SELECTOR_GREEN: False,
        }
        self._ranges: Dict[str, Tuple[float, float]] = {
            SELECTOR_RED: (200.0, 400.0),
            SELECTOR_BLUE: (500.0, 700.0),
            SELECTOR_GREEN: (800.0, 1000.0),
        }
        self._2d_candidates: List[str] = []
        self._last_raw_entries: List[dict] = []
        self._context_status: str = ""
        self._view_options = _copy_view_options()
        self._is_setting_options = False
        self._spatial_sites: Optional[dict] = None
        self._selected_points: list[tuple[float, float]] = []
        self._selected_map_name: Optional[str] = None

        self._current_theme = self._view_options["plot_theme"]
        self._init_ui()
        self.set_mode(mode)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Top bar: slim context header (left) + spacer + view options buttons (right) ──
        self.top_bar = QWidget(self)
        self.top_bar.setObjectName("liveMapsTopBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # ── Slim shared context header (2D switcher and diagnostic/exclusion status) ──
        self.header_bar = QFrame(self.top_bar)
        self.header_bar.setObjectName("liveMapsContextHeader")
        self.header_bar.setFixedHeight(30)
        header_layout = QHBoxLayout(self.header_bar)
        header_layout.setContentsMargins(6, 2, 6, 2)
        header_layout.setSpacing(8)

        # Multi-2D candidate switcher: ◀ [ComboBox] ▶
        self.switcher_container = QWidget(self.header_bar)
        switcher_layout = QHBoxLayout(self.switcher_container)
        switcher_layout.setContentsMargins(0, 0, 0, 0)
        switcher_layout.setSpacing(4)

        lbl_2d = QLabel("2D Map:", self.switcher_container)
        lbl_2d.setStyleSheet("font-size: 11px; font-weight: bold;")
        switcher_layout.addWidget(lbl_2d)

        self.btn_prev_2d = QPushButton("◀", self.switcher_container)
        self.btn_prev_2d.setFixedSize(22, 22)
        self.btn_prev_2d.setToolTip("Previous selected 2D map")
        self.btn_prev_2d.clicked.connect(lambda: self._step_2d_candidate(-1))
        switcher_layout.addWidget(self.btn_prev_2d)

        self.cbb_2d_map = QComboBox(self.switcher_container)
        self.cbb_2d_map.setMinimumWidth(140)
        self.cbb_2d_map.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.cbb_2d_map.currentTextChanged.connect(self._on_2d_combo_changed)
        switcher_layout.addWidget(self.cbb_2d_map)

        self.btn_next_2d = QPushButton("▶", self.switcher_container)
        self.btn_next_2d.setFixedSize(22, 22)
        self.btn_next_2d.setToolTip("Next selected 2D map")
        self.btn_next_2d.clicked.connect(lambda: self._step_2d_candidate(1))
        switcher_layout.addWidget(self.btn_next_2d)

        self.switcher_container.setVisible(False)
        header_layout.addWidget(self.switcher_container, stretch=0)

        tok = current_tokens()

        # Exclusion / diagnostic status badge
        self.lbl_context_status = QLabel("", self.header_bar)
        self.lbl_context_status.setStyleSheet(f"font-size: 11px; color: {tok.warning};")
        self.lbl_context_status.setVisible(False)
        header_layout.addWidget(self.lbl_context_status, stretch=1)

        self.header_bar.setVisible(False)
        top_layout.addWidget(self.header_bar, stretch=1)

        self._toolbar_spacer = QWidget(self.top_bar)
        self._toolbar_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_layout.addWidget(self._toolbar_spacer, stretch=1)

        # ── Persistent View Options toolbar ([Show Legend | More view options ⋯]) ──
        self.options_toolbar = QWidget(self.top_bar)
        opts_layout = QHBoxLayout(self.options_toolbar)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.setSpacing(4)

        self.btn_legend = QToolButton(self.options_toolbar)
        self.btn_legend.setObjectName("liveMapsBtnLegend")
        self.btn_legend.setIcon(Icons.legend(color=tok.text_primary))
        self.btn_legend.setIconSize(QSize(16, 16))
        self.btn_legend.setFixedSize(26, 26)
        self.btn_legend.setCheckable(True)
        self.btn_legend.setChecked(self._view_options["show_legend"])
        self.btn_legend.setToolTip(
            "Show / Hide plot legend\n(Double-click a canvas legend box to edit labels)"
        )
        self.btn_legend.toggled.connect(self._on_legend_toggled)
        opts_layout.addWidget(self.btn_legend)

        self.btn_options = QToolButton(self.options_toolbar)
        self.btn_options.setObjectName("liveMapsBtnOptions")
        self.btn_options.setIcon(Icons.settings(color=tok.text_primary))
        self.btn_options.setIconSize(QSize(16, 16))
        self.btn_options.setFixedSize(26, 26)
        self.btn_options.setPopupMode(QToolButton.InstantPopup)
        self.btn_options.setToolTip("More view options")
        self.options_menu = self._create_options_menu()
        self.btn_options.setMenu(self.options_menu)
        opts_layout.addWidget(self.btn_options)

        top_layout.addWidget(self.options_toolbar, stretch=0)
        self.top_bar.setVisible(False)
        main_layout.addWidget(self.top_bar)

        # ── Placeholder when all selectors are off ──
        self.lbl_placeholder = QLabel(self)
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet(f"font-size: 13px; color: {tok.text_muted}; padding: 30px;")
        main_layout.addWidget(self.lbl_placeholder)

        # ── Horizontal container for the 3 subpanels ──
        self.maps_container = QWidget(self)
        maps_layout = QHBoxLayout(self.maps_container)
        maps_layout.setContentsMargins(0, 0, 0, 0)
        maps_layout.setSpacing(6)

        for color_id in SELECTOR_KEYS:
            panel = LiveMapSubpanel(color_id, self.maps_container, mode=self.mode)
            panel.parameter_changed.connect(self.parameter_change_requested.emit)
            panel.legend_edit_requested.connect(self.open_legend_editor)
            panel.view_reset_requested.connect(self._sync_shared_y)
            panel.normalization_changed.connect(self._sync_shared_y)
            panel.spectra_selected.connect(self._on_panel_spectra_selected)
            panel.copy_data_requested.connect(self.copy_data_requested.emit)
            panel.set_view_options(self._view_options, redraw=False)
            panel.setVisible(False)  # Hidden by default until activated
            self.panels[color_id] = panel
            maps_layout.addWidget(panel, stretch=1)

        main_layout.addWidget(self.maps_container, stretch=1)
        self._update_visibility()

    def _on_legend_toggled(self, checked: bool):
        """Keep top toggle button, checkbox, and menu action synchronized on legend toggle."""
        checked = bool(checked)
        self.btn_legend.blockSignals(True)
        self.act_show_legend.blockSignals(True)
        self.chk_show_legend.blockSignals(True)
        self.btn_legend.setChecked(checked)
        self.act_show_legend.setChecked(checked)
        self.chk_show_legend.setChecked(checked)
        self.btn_legend.blockSignals(False)
        self.act_show_legend.blockSignals(False)
        self.chk_show_legend.blockSignals(False)
        self._update_option("show_legend", checked)

    def _update_option(self, key: str, value):
        """Update one view option and apply cleanly to visible panels via fast path."""
        if key in ("plot_theme", "copy_fig_theme"):
            value = _normalize_plot_theme(value)
        if self._view_options.get(key) == value:
            return
        self._view_options[key] = value
        if key == "plot_theme":
            self._current_theme = value
        self._apply_view_options_to_panels(changed_keys={key})
        if not self._is_setting_options:
            self.view_options_changed.emit(self.get_view_options())

    def get_view_options(self) -> dict:
        """Return a deep copy of current Live Maps view options."""
        return _copy_view_options(self._view_options)

    def set_view_options(self, options: dict, emit_signal: bool = False):
        """Apply a partial or full view options dictionary and refresh visible panels."""
        old_opts = self._view_options
        new_opts = _copy_view_options({**old_opts, **(options or {})})
        changed_keys = {k for k in new_opts if new_opts.get(k) != old_opts.get(k)}
        if not changed_keys:
            return
        self._view_options = new_opts
        self._current_theme = self._view_options["plot_theme"]
        self._is_setting_options = True
        try:
            self._sync_options_controls()
            self._apply_view_options_to_panels(changed_keys=changed_keys)
        finally:
            self._is_setting_options = False
        if emit_signal:
            self.view_options_changed.emit(self.get_view_options())

    def reset_view_options(self):
        """Reset all view options and custom legend labels/colors to defaults."""
        self._view_options = _copy_view_options()
        self._current_theme = self._view_options["plot_theme"]
        self._is_setting_options = True
        try:
            self._sync_options_controls()
            self._apply_view_options_to_panels(changed_keys=None)
        finally:
            self._is_setting_options = False
        self.view_options_changed.emit(self.get_view_options())

    def _apply_view_options_to_panels(self, changed_keys: Optional[set] = None):
        """Push view options into all subpanels using fast artist updates when possible."""
        for panel in self.panels.values():
            redraw = bool(panel.isVisible() and panel._display_kind is not None)
            panel.apply_view_options_fast(
                self._view_options,
                changed_keys=changed_keys,
                redraw=redraw,
            )
        if changed_keys is None or (changed_keys & {"share_y", "xscale", "yscale", "text_fontsize"}):
            restore = bool(changed_keys is None or "share_y" in changed_keys)
            self._sync_shared_y(restore_when_disabled=restore)

    def _legend_editor_entries(self) -> List[dict]:
        """Collect editable curve/series entries from active 1D subpanel contents."""
        if self._last_raw_entries and len(self._last_raw_entries) >= 2:
            return [
                {
                    "key": str(e.get("name", f"Map {idx + 1}")),
                    "default_label": str(e.get("name", f"Map {idx + 1}")),
                    "auto_color": DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
                }
                for idx, e in enumerate(self._last_raw_entries)
            ]
        entries = []
        for cid in SELECTOR_KEYS:
            panel = self.panels.get(cid)
            if panel is not None and self._active_states.get(cid, False):
                param = panel.get_selected_parameter()
                entries.append({
                    "key": cid,
                    "default_label": f"{panel.color_name}: {param}",
                    "auto_color": panel.color_hex,
                })
        if not entries:
            for cid in SELECTOR_KEYS:
                panel = self.panels[cid]
                entries.append({
                    "key": cid,
                    "default_label": f"{panel.color_name}: {panel.get_selected_parameter()}",
                    "auto_color": panel.color_hex,
                })
        return entries

    def open_legend_editor(self):
        """Open LiveMapsLegendEditorDialog to customize legend placement, labels, and colors."""
        dlg = LiveMapsLegendEditorDialog(
            entries=self._legend_editor_entries(),
            custom_labels=self._view_options.get("custom_labels"),
            custom_colors=self._view_options.get("custom_colors"),
            legend_loc=self._view_options.get("legend_loc", "best"),
            legend_ncol=self._view_options.get("legend_ncol", 1),
            legend_fontsize=self._view_options.get("legend_fontsize", 8),
            parent=self,
        )
        if dlg.exec():
            res = dlg.result_state()
            self.set_view_options(
                {**res, "show_legend": True},
                emit_signal=True,
            )

    def _sync_shared_y(self, restore_when_disabled: bool = False):
        """Synchronize Y-axis limits across visible 1D subpanels when Share Y-axis is enabled."""
        use_log = self._view_options.get("yscale", "linear") == "log"
        target_scale = "log" if use_log else "linear"
        if not self._view_options.get("share_y", False):
            if restore_when_disabled:
                for cid in SELECTOR_KEYS:
                    panel = self.panels.get(cid)
                    if (
                        panel is not None
                        and panel.isVisible()
                        and panel._display_kind in ("series", "multi_profile")
                        and panel.ax.get_yscale() == target_scale
                    ):
                        b = panel._valid_pos_y_bounds if use_log else panel._valid_y_bounds
                        if b is not None and np.isfinite(b[0]) and np.isfinite(b[1]) and b[1] > b[0]:
                            cur_lo, cur_hi = panel.ax.get_ylim()
                            if not (
                                np.isclose(cur_lo, b[0], rtol=1e-9, atol=1e-12)
                                and np.isclose(cur_hi, b[1], rtol=1e-9, atol=1e-12)
                            ):
                                panel.ax.set_ylim(b[0], b[1])
                                panel.canvas.draw_idle()
            return
        target_panels = []
        bounds = []
        for cid in SELECTOR_KEYS:
            panel = self.panels.get(cid)
            if (
                panel is not None
                and panel.isVisible()
                and panel._display_kind in ("series", "multi_profile")
            ):
                b = panel._valid_pos_y_bounds if use_log else panel._valid_y_bounds
                if b is not None and panel.ax.get_yscale() == target_scale:
                    target_panels.append(panel)
                    bounds.append(b)
        if len(target_panels) < 2:
            return
        g_lo = min(b[0] for b in bounds)
        g_hi = max(b[1] for b in bounds)
        if not (np.isfinite(g_lo) and np.isfinite(g_hi)) or g_hi <= g_lo:
            return
        for panel in target_panels:
            cur_lo, cur_hi = panel.ax.get_ylim()
            if not (
                np.isclose(cur_lo, g_lo, rtol=1e-9, atol=1e-12)
                and np.isclose(cur_hi, g_hi, rtol=1e-9, atol=1e-12)
            ):
                panel.ax.set_ylim(g_lo, g_hi)
                panel.canvas.draw_idle()

    def set_mode(self, mode: str):
        """Configure tab for 'map' (2D spatial heatmaps) or 'spectra' (1D variation graphs) mode."""
        self.mode = mode
        if mode != "map":
            self._spatial_sites = None
            self._selected_points = []
            self._selected_map_name = None
        if mode == "spectra":
            self.lbl_placeholder.setText(
                "Live Spectral Mapping\n\n"
                "Activate one or more spectral range selectors (Red, Blue, Green)\n"
                "in the Spectra viewer right panel to display live parameter variation graphs."
            )
        else:
            self.lbl_placeholder.setText(
                "Live Spectral Mapping\n\n"
                "Activate one or more spectral range selectors (Red, Blue, Green)\n"
                "in the Spectra viewer right panel to display live 2D heatmaps\n"
                "(or line plots for 1D profile maps)."
            )
        for panel in self.panels.values():
            panel.set_mode(mode)
            if mode != "map":
                panel.set_spatial_sites(None)
                panel.set_selected_points([])

    def active_site_map_name(self) -> Optional[str]:
        """Return the name of the map currently backing interactive Live Maps selection."""
        return self._spatial_sites.get("map_name") if self._spatial_sites else None

    def set_2d_data_provider(self, provider: Optional[Callable[[str], Optional[tuple]]]):
        """Register optional ``provider(color_id) -> (x, y, values)`` on all R/B/G subpanels."""
        for panel in self.panels.values():
            panel.set_2d_data_provider(provider)

    def set_spatial_sites(self, sites: Optional[dict]):
        """Propagate current map spatial site metadata to all R/B/G subpanels."""
        self._spatial_sites = sites
        cur_map = sites.get("map_name") if sites else None
        show_pts = (
            self._selected_points
            if (cur_map and (self._selected_map_name is None or self._selected_map_name == cur_map))
            else []
        )
        for panel in self.panels.values():
            panel.set_spatial_sites(sites)
            panel.set_selected_points(show_pts)

    def set_selected_points(self, points: list, map_name: Optional[str] = None):
        """Highlight (X, Y) spectrum points across all R/B/G subpanels without feedback emission."""
        pts = list(dict.fromkeys((float(pt[0]), float(pt[1])) for pt in (points or [])))
        self._selected_points = pts
        self._selected_map_name = map_name
        active_map = self.active_site_map_name()
        effective = (
            pts
            if (map_name is None or active_map is None or active_map == map_name)
            else []
        )
        for panel in self.panels.values():
            panel.set_selected_points(effective)

    def _on_panel_spectra_selected(self, points: list):
        """Sync one subpanel's interactive XY selection across R/B/G panels and notify workspace."""
        pts = list(dict.fromkeys((float(pt[0]), float(pt[1])) for pt in (points or [])))
        self._selected_points = pts
        self._selected_map_name = self.active_site_map_name()
        for panel in self.panels.values():
            panel.set_selected_points(pts)
        self.spectra_selected.emit(list(pts))

    def set_selector_active(self, color_id: str, is_active: bool):
        """Update visibility of a subpanel based on selector state."""
        self._active_states[color_id] = is_active
        panel = self.panels.get(color_id)
        if panel:
            panel.setVisible(is_active)
        self._update_visibility()

    def set_selector_range(self, color_id: str, x_min: float, x_max: float):
        """Update range label on a subpanel."""
        self._ranges[color_id] = (x_min, x_max)
        panel = self.panels.get(color_id)
        if panel:
            panel.set_range_text(x_min, x_max)

    def _update_visibility(self):
        """Show or hide top bar, container, and placeholder based on active selectors."""
        any_active = any(self._active_states.values())
        self.lbl_placeholder.setVisible(not any_active)
        self.maps_container.setVisible(any_active)
        self._update_header_visibility(any_active)

    def _update_header_visibility(self, any_active: Optional[bool] = None):
        """Show top options toolbar whenever selectors are active; show context header only when needed."""
        if any_active is None:
            any_active = any(self._active_states.values())
        has_switcher = len(self._2d_candidates) >= 2
        has_status = bool(self._context_status)
        show_context_header = any_active and (has_switcher or has_status)
        self.switcher_container.setVisible(has_switcher)
        self.lbl_context_status.setVisible(has_status)
        self.header_bar.setVisible(show_context_header)
        self._toolbar_spacer.setVisible(any_active and not show_context_header)
        self.top_bar.setVisible(any_active)

    def _on_2d_combo_changed(self, name: str):
        if name:
            self.active_2d_map_requested.emit(name)
            self._sync_2d_nav_buttons()

    def _step_2d_candidate(self, delta: int):
        idx = self.cbb_2d_map.currentIndex() + delta
        if 0 <= idx < self.cbb_2d_map.count():
            self.cbb_2d_map.setCurrentIndex(idx)

    def _sync_2d_nav_buttons(self):
        count = self.cbb_2d_map.count()
        idx = self.cbb_2d_map.currentIndex()
        self.btn_prev_2d.setEnabled(count > 1 and idx > 0)
        self.btn_next_2d.setEnabled(count > 1 and 0 <= idx < count - 1)

    def update_2d_candidates(
        self,
        candidates: List[str],
        active_name: Optional[str],
        status_msg: str = "",
    ):
        """Update the 2D switcher combo box and exclusion/diagnostic badge."""
        self._2d_candidates = list(candidates) if candidates else []
        self._context_status = status_msg or ""
        self.lbl_context_status.setText(self._context_status)

        self.cbb_2d_map.blockSignals(True)
        try:
            self.cbb_2d_map.clear()
            for name in self._2d_candidates:
                self.cbb_2d_map.addItem(name)
            if active_name:
                idx = self.cbb_2d_map.findText(active_name)
                if idx >= 0:
                    self.cbb_2d_map.setCurrentIndex(idx)
        finally:
            self.cbb_2d_map.blockSignals(False)

        self._sync_2d_nav_buttons()
        self._update_header_visibility()

    def update_map_data(
        self,
        color_id: str,
        grid: Optional[np.ndarray],
        extent: list[float],
        vmin: float,
        vmax: float,
        status_msg: str = "",
    ):
        """Forward calculated 2D grid data to matching subpanel."""
        self._last_raw_entries = []
        panel = self.panels.get(color_id)
        if panel:
            plot_theme = self._view_options.get("plot_theme", self._current_theme)
            panel.update_map(grid, extent, vmin, vmax, status_msg, theme=plot_theme)

    def update_series_data(
        self,
        color_id: str,
        values: Optional[np.ndarray],
        labels: list[str],
        status_msg: str = "",
    ):
        """Forward calculated 1D parameter series data to matching subpanel."""
        self._last_raw_entries = []
        panel = self.panels.get(color_id)
        if panel:
            plot_theme = self._view_options.get("plot_theme", self._current_theme)
            panel.update_series(values, labels, status_msg, theme=plot_theme)
        self._sync_shared_y()

    def update_profile_data(
        self,
        color_id: str,
        positions: Optional[np.ndarray],
        values: Optional[np.ndarray],
        axis_label: str = "X",
        status_msg: str = "",
    ):
        """Forward a single 1D profile (values along the profile axis) to the matching subpanel."""
        self._last_raw_entries = []
        panel = self.panels.get(color_id)
        if panel:
            plot_theme = self._view_options.get("plot_theme", self._current_theme)
            panel.update_series(
                values, None, status_msg, theme=plot_theme,
                x_values=positions if positions is not None else np.array([]),
                x_label=f"{axis_label or 'Position'} (mm)",
            )
        self._sync_shared_y()

    def update_multi_profile_data(
        self,
        color_id: str,
        entries: List[dict],
        shared_x_label: str = "X",
        status_msg: str = "",
    ):
        """Forward stacked multi-1D profile entries and update status."""
        self._last_raw_entries = list(entries) if entries else []
        if status_msg and status_msg != "OK":
            self._context_status = status_msg
            self.lbl_context_status.setText(self._context_status)
        self._update_header_visibility()
        panel = self.panels.get(color_id)
        if panel:
            plot_theme = self._view_options.get("plot_theme", self._current_theme)
            panel.update_multi_profile(
                entries, shared_x_label=shared_x_label, status_msg=status_msg,
                theme=plot_theme,
            )
        self._sync_shared_y()

    def clear_all(self):
        """Clear all subpanel artists, 2D candidate switcher, and status."""
        self._last_raw_entries = []
        self._2d_candidates = []
        self._context_status = ""
        self._spatial_sites = None
        self._selected_points = []
        self._selected_map_name = None
        self.lbl_context_status.setText("")
        self.cbb_2d_map.blockSignals(True)
        self.cbb_2d_map.clear()
        self.cbb_2d_map.blockSignals(False)
        self._sync_2d_nav_buttons()
        self._update_header_visibility()
        for panel in self.panels.values():
            panel.set_spatial_sites(None)
            panel.set_selected_points([])
            panel.clear_plot()

    def apply_global_theme(self, theme_name: str):
        """Update Qt toolbar and subpanel icon colors without overriding Plot theme."""
        tok = current_tokens()
        icon_color = tok.text_primary
        self._icon_color = icon_color
        self.btn_legend.setIcon(Icons.legend(color=icon_color))
        self.btn_options.setIcon(Icons.settings(color=icon_color))
        if hasattr(self, "lbl_context_status"):
            self.lbl_context_status.setStyleSheet(f"font-size: 11px; color: {tok.warning};")
        if hasattr(self, "lbl_placeholder"):
            self.lbl_placeholder.setStyleSheet(f"font-size: 13px; color: {tok.text_muted}; padding: 30px;")
        for panel in self.panels.values():
            panel.apply_icon_theme(icon_color)

    def apply_theme(self, theme_name: str):
        """Update Qt UI icons and explicitly switch the Matplotlib Plot theme."""
        self.apply_global_theme(theme_name)
        self.set_view_options({"plot_theme": _normalize_plot_theme(theme_name)}, emit_signal=False)


__all__ = [
    "AVAILABLE_PALETTES",
    "DEFAULT_PALETTES",
    "LEGEND_LOCATIONS",
    "LiveMapSubpanel",
    "LiveMapsLegendEditorDialog",
    "VLiveMaps",
    "_copy_view_options",
    "_get_cached_style_dict",
    "_normalize_1d_values",
    "_normalize_plot_theme",
    "_parse_norm_bound",
    "_resolve_style_path",
]

