"""Unit tests for VLiveMaps and LiveMapSubpanel."""

import numpy as np
from matplotlib.colors import LogNorm

from spectroview.model.spectral_parameters import (
    PARAM_INTEGRATED_INTENSITY,
    PARAM_PEAK_POSITION,
    PARAM_RB_RATIO,
    SELECTOR_RED,
    SELECTOR_BLUE,
    SELECTOR_GREEN,
)
from spectroview.view.components.v_live_maps import (
    VLiveMaps,
    LiveMapSubpanel,
)
from spectroview.view.components.v_live_maps_legend_editor import (
    LiveMapsLegendEditorDialog,
)


def test_live_map_subpanel_creation(qapp):
    panel = LiveMapSubpanel(SELECTOR_RED)
    assert panel.color_id == SELECTOR_RED
    assert panel.cbb_param.currentText() == PARAM_INTEGRATED_INTENSITY

    # Signal on parameter change
    emitted = []
    panel.parameter_changed.connect(lambda cid, p: emitted.append((cid, p)))
    panel.set_selected_parameter(PARAM_PEAK_POSITION)
    assert len(emitted) == 1
    assert emitted[0] == (SELECTOR_RED, PARAM_PEAK_POSITION)

    # Range label update
    panel.set_range_text(100.5, 250.8)
    assert "[100.5, 250.8]" in panel.lbl_range.text()


def test_live_map_subpanel_render(qapp):
    panel = LiveMapSubpanel(SELECTOR_BLUE)
    grid = np.arange(12, dtype=float).reshape(3, 4)
    extent = [0.0, 4.0, 0.0, 3.0]

    panel.update_map(grid, extent, 0.0, 11.0, status_msg="OK", theme="Light Mode")
    assert panel.img is not None
    assert panel.cbar is not None
    orig_img = panel.img
    orig_cbar = panel.cbar

    # In-place update with new grid preserves existing image and colorbar (no redraw bounce)
    grid2 = np.arange(12, dtype=float).reshape(3, 4) * 2.0
    panel.update_map(grid2, extent, 0.0, 22.0, status_msg="OK", theme="Light Mode")
    assert panel.img is orig_img
    assert panel.cbar is orig_cbar

    # Render None / empty grid gracefully
    panel.update_map(None, extent, 0.0, 1.0, status_msg="No data", theme="Dark Mode")
    assert panel.cbar is None


def test_v_live_maps_visibility_toggling(qapp):
    live_maps = VLiveMaps()
    live_maps.show()

    # Initially placeholder visible, maps container hidden
    assert live_maps.lbl_placeholder.isVisible() is True
    assert live_maps.maps_container.isVisible() is False
    assert live_maps.panels[SELECTOR_RED].isVisible() is False

    # Activate Red
    live_maps.set_selector_active(SELECTOR_RED, True)
    assert live_maps.lbl_placeholder.isVisible() is False
    assert live_maps.maps_container.isVisible() is True
    assert live_maps.panels[SELECTOR_RED].isVisible() is True
    assert live_maps.panels[SELECTOR_BLUE].isVisible() is False
    assert live_maps.panels[SELECTOR_GREEN].isVisible() is False

    # Activate Blue
    live_maps.set_selector_active(SELECTOR_BLUE, True)
    assert live_maps.panels[SELECTOR_RED].isVisible() is True
    assert live_maps.panels[SELECTOR_BLUE].isVisible() is True
    assert live_maps.panels[SELECTOR_GREEN].isVisible() is False

    # Deactivate both
    live_maps.set_selector_active(SELECTOR_RED, False)
    live_maps.set_selector_active(SELECTOR_BLUE, False)
    assert live_maps.lbl_placeholder.isVisible() is True
    assert live_maps.maps_container.isVisible() is False


def test_v_live_maps_update_data_and_theme(qapp):
    live_maps = VLiveMaps()
    live_maps.set_selector_active(SELECTOR_GREEN, True)

    grid = np.ones((5, 5))
    live_maps.update_map_data(SELECTOR_GREEN, grid, [0, 5, 0, 5], 0.0, 1.0, "OK")
    assert live_maps.panels[SELECTOR_GREEN].img is not None

    # Apply dark theme
    live_maps.apply_theme("dark")
    assert live_maps._current_theme == "dark"


def test_live_map_subpanel_spectra_series_render(qapp):
    """Test LiveMapSubpanel and VLiveMaps in 'spectra' mode (1D parameter variation graph)."""
    live_maps = VLiveMaps(mode="spectra")
    live_maps.show()
    red_panel = live_maps.panels[SELECTOR_RED]

    # In spectra mode, 2D colormap selector is hidden
    assert red_panel.cbb_palette.isHidden() is True

    live_maps.set_selector_active(SELECTOR_RED, True)
    vals = np.array([10.0, 20.0, 15.0, 30.0])
    labels = ["S1", "S2", "S3", "S4"]

    live_maps.update_series_data(SELECTOR_RED, vals, labels, "OK")
    assert red_panel.line is not None
    orig_line = red_panel.line
    assert np.allclose(red_panel.line.get_ydata(), vals)

    # Fast in-place update when dragging bracket (same spectrum count) preserves line artist
    vals2 = np.array([12.0, 24.0, 18.0, 36.0])
    live_maps.update_series_data(SELECTOR_RED, vals2, labels, "OK")
    assert red_panel.line is orig_line
    assert np.allclose(red_panel.line.get_ydata(), vals2)

    # Empty / NaN series renders status message gracefully and clears line artist
    live_maps.update_series_data(SELECTOR_RED, np.array([]), [], "No spectra selected")
    assert red_panel.line is None


# ── 1D profile maps & R/B ratio ──────────────────────────────────────────────


def test_live_map_profile_render_in_map_mode(qapp):
    """1D profile maps render a line plot vs. position and hide the colormap selector."""
    live_maps = VLiveMaps(mode="map")
    live_maps.show()
    live_maps.set_selector_active(SELECTOR_RED, True)
    panel = live_maps.panels[SELECTOR_RED]
    assert panel.cbb_palette.isHidden() is False

    pos = np.array([0.0, 0.5, 1.0, 1.5])
    vals = np.array([1.0, 3.0, 2.0, 5.0])
    live_maps.update_profile_data(SELECTOR_RED, pos, vals, "X", "OK")
    assert panel.line is not None
    assert panel.img is None and panel.cbar is None
    np.testing.assert_allclose(panel.line.get_xdata(), pos)
    np.testing.assert_allclose(panel.line.get_ydata(), vals)
    assert panel.ax.get_xlabel() == "X (mm)"
    assert panel.cbb_palette.isHidden() is True
    xmin, xmax = panel.ax.get_xlim()
    assert xmin < 0.0 and xmax > 1.5

    # Fast in-place update while dragging keeps the same artist
    orig_line = panel.line
    live_maps.update_profile_data(SELECTOR_RED, pos, vals * 2, "X", "OK")
    assert panel.line is orig_line
    np.testing.assert_allclose(panel.line.get_ydata(), vals * 2)

    # Same length, different profile axis (map switch) still updates x data/label
    live_maps.update_profile_data(SELECTOR_RED, pos * 10, vals, "Y", "OK")
    np.testing.assert_allclose(panel.line.get_xdata(), pos * 10)
    assert panel.ax.get_xlabel() == "Y (mm)"


def test_live_map_switch_heatmap_profile_heatmap(qapp):
    live_maps = VLiveMaps(mode="map")
    live_maps.resize(900, 280)
    live_maps.show()
    live_maps.set_selector_active(SELECTOR_BLUE, True)
    panel = live_maps.panels[SELECTOR_BLUE]

    live_maps.update_map_data(SELECTOR_BLUE, np.ones((3, 3)), [-20, 20, -20, 20], 0.0, 1.0, "OK")
    panel.canvas.draw()
    assert panel.img is not None and panel.line is None
    assert panel.ax.get_aspect() in ("equal", 1.0)

    # Switching from 2D heatmap (aspect="equal") to 1D profile with large Y vs X
    # span must reset aspect to "auto" so the axes does not collapse into a line.
    x_pos = np.linspace(-20.0, 20.0, 41)
    y_vals = np.linspace(5000.0, 40000.0, 41)
    live_maps.update_profile_data(SELECTOR_BLUE, x_pos, y_vals, "X", "OK")
    panel.canvas.draw()
    assert panel.line is not None and panel.img is None and panel.cbar is None
    assert panel.cbb_palette.isHidden() is True
    assert panel.ax.get_aspect() == "auto"
    pos_bbox = panel.ax.get_position()
    assert pos_bbox.width > 0.5 and pos_bbox.height > 0.5

    live_maps.update_map_data(SELECTOR_BLUE, np.ones((3, 3)) * 2, [-20, 20, -20, 20], 0.0, 2.0, "OK")
    panel.canvas.draw()
    assert panel.img is not None and panel.cbar is not None and panel.line is None
    assert panel.cbb_palette.isHidden() is False
    assert panel.ax.get_aspect() in ("equal", 1.0)


def test_live_map_profile_empty_and_nan_values(qapp):
    live_maps = VLiveMaps(mode="map")
    live_maps.set_selector_active(SELECTOR_RED, True)
    panel = live_maps.panels[SELECTOR_RED]

    live_maps.update_profile_data(SELECTOR_RED, np.array([]), np.array([]), "X", "Unable")
    assert panel.line is None

    # NaN (masked) points are drawn as gaps without breaking axis limits
    live_maps.update_profile_data(
        SELECTOR_RED, np.arange(4.0), np.array([1.0, np.nan, 2.0, 3.0]), "X", "OK"
    )
    assert panel.line is not None
    ymin, ymax = panel.ax.get_ylim()
    assert np.isfinite(ymin) and np.isfinite(ymax)


def test_live_map_theme_reapply_keeps_profile_plot(qapp):
    live_maps = VLiveMaps(mode="map")
    live_maps.show()
    live_maps.set_selector_active(SELECTOR_RED, True)
    panel = live_maps.panels[SELECTOR_RED]
    pos = np.array([2.0, 4.0, 6.0])
    live_maps.update_profile_data(SELECTOR_RED, pos, np.array([1.0, 2.0, 3.0]), "Distance", "OK")

    live_maps.apply_theme("dark")
    assert panel.line is not None and panel.img is None
    np.testing.assert_allclose(panel.line.get_xdata(), pos)
    assert panel.ax.get_xlabel() == "Distance (mm)"


def test_green_panel_offers_rb_ratio_option(qapp):
    green = LiveMapSubpanel(SELECTOR_GREEN)
    red = LiveMapSubpanel(SELECTOR_RED)
    assert green.cbb_param.findText(PARAM_RB_RATIO) >= 0
    assert red.cbb_param.findText(PARAM_RB_RATIO) < 0
    # Default remains the standard Green intensity
    assert green.cbb_param.currentText() == PARAM_INTEGRATED_INTENSITY

    emitted = []
    green.parameter_changed.connect(lambda cid, p: emitted.append((cid, p)))
    green.set_range_text(800.0, 900.0)
    green.set_selected_parameter(PARAM_RB_RATIO)
    assert emitted[-1] == (SELECTOR_GREEN, PARAM_RB_RATIO)
    assert green.is_ratio_display()
    assert "Red" in green.lbl_range.text() and "Blue" in green.lbl_range.text()

    # Range updates while in ratio mode keep the ratio label; switching back restores range
    green.set_range_text(810.0, 910.0)
    assert "Red" in green.lbl_range.text()
    green.set_selected_parameter(PARAM_INTEGRATED_INTENSITY)
    assert "[810.0, 910.0]" in green.lbl_range.text()


def test_v_live_maps_multi_profile_legend_theme_and_fast_path(qapp):
    live_maps = VLiveMaps(mode="map")
    live_maps.show()
    live_maps.set_selector_active(SELECTOR_RED, True)
    panel = live_maps.panels[SELECTOR_RED]

    # Switching from 2D heatmap -> multi-1D must reset aspect="auto" and show legend header
    live_maps.update_map_data(SELECTOR_RED, np.ones((4, 4)), [-10, 10, -10, 10], 0.0, 1.0, "OK")
    entries = [
        {"name": "p1", "positions": np.array([0.0, 1.0, 2.0]), "values": np.array([10.0, 20.0, 30.0]), "is_primary": True, "axis_label": "X", "out_of_range": False, "msg": "OK"},
        {"name": "p2", "positions": np.array([0.0, 1.0, 2.0]), "values": np.array([5.0, np.nan, 15.0]), "is_primary": False, "axis_label": "X", "out_of_range": False, "msg": "OK"},
        {"name": "p3_disjoint", "positions": np.array([0.0, 1.0]), "values": np.array([np.nan, np.nan]), "is_primary": False, "axis_label": "X", "out_of_range": True, "msg": "Out of range"},
    ]
    live_maps.update_multi_profile_data(SELECTOR_RED, entries, shared_x_label="X", status_msg="Excluded 2D map(s): m2d")
    panel.canvas.draw()

    assert panel.img is None
    assert panel.ax.get_aspect() == "auto"
    assert len(panel.lines) == 3
    assert panel.lines[0].get_linewidth() > panel.lines[1].get_linewidth()
    ymin, ymax = panel.ax.get_ylim()
    assert np.isfinite(ymin) and np.isfinite(ymax)
    assert live_maps.header_bar.isVisible() is True
    assert live_maps.legend_scroll.isVisible() is True
    assert "Excluded 2D" in live_maps.lbl_context_status.text()

    # Fast path reuses Line2D objects in-place on value change
    first_line = panel.lines[0]
    entries_moved = [dict(e) for e in entries]
    entries_moved[0]["values"] = np.array([12.0, 24.0, 36.0])
    live_maps.update_multi_profile_data(SELECTOR_RED, entries_moved, shared_x_label="X", status_msg="Excluded 2D map(s): m2d")
    assert panel.lines[0] is first_line

    # Theme switch preserves multi-profile curves
    live_maps.apply_theme("dark")
    assert len(panel.lines) == 3

    # Clear resets all artists & header
    live_maps.clear_all()
    assert panel.lines == []
    assert live_maps.header_bar.isVisible() is False


def test_v_live_maps_2d_switcher_emits_active_map_requested(qapp):
    live_maps = VLiveMaps(mode="map")
    live_maps.show()
    live_maps.set_selector_active(SELECTOR_RED, True)

    requested = []
    live_maps.active_2d_map_requested.connect(requested.append)

    live_maps.update_2d_candidates(["m1", "m2", "m3"], "m1", "Excluded 1D profile(s): p1")
    assert live_maps.switcher_container.isVisible() is True
    assert live_maps.btn_prev_2d.isEnabled() is False
    assert live_maps.btn_next_2d.isEnabled() is True

    live_maps.btn_next_2d.click()
    assert requested[-1] == "m2"
    assert live_maps.btn_prev_2d.isEnabled() is True
    live_maps.btn_next_2d.click()
    assert requested[-1] == "m3"
    assert live_maps.btn_next_2d.isEnabled() is False

    # Shrinking to 1 candidate without exclusion hides header
    live_maps.update_2d_candidates(["m1"], "m1", "")
    assert live_maps.switcher_container.isVisible() is False
    assert live_maps.header_bar.isVisible() is False


def test_v_live_maps_view_options_plot_styles_and_scales(qapp):
    """Plot styles (line/dot/line+dots), line width, dot size, and Linear/Log X & Y scaling."""
    live_maps = VLiveMaps(mode="map")
    live_maps.show()
    assert live_maps.top_bar.isVisible() is False

    live_maps.set_selector_active(SELECTOR_RED, True)
    assert live_maps.top_bar.isVisible() is True
    assert live_maps.btn_legend.isVisible() is True
    assert live_maps.btn_options.isVisible() is True
    assert live_maps.header_bar.isVisible() is False  # context bar stays hidden for single profile

    pos = np.array([1.0, 2.0, 4.0, 8.0])
    vals = np.array([10.0, 100.0, 500.0, 1000.0])
    live_maps.update_profile_data(SELECTOR_RED, pos, vals, axis_label="X", status_msg="OK")

    panel = live_maps.panels[SELECTOR_RED]
    assert panel.line.get_linestyle() == "-"
    assert panel.line.get_marker() == "o"

    # Switch to Line only with thicker width
    live_maps.cbb_plot_style.setCurrentText("Line")
    live_maps.spb_linewidth.setValue(3.2)
    assert panel.line.get_linestyle() == "-"
    assert panel.line.get_marker() in ("", "None")
    assert panel.line.get_linewidth() == 3.2

    # Switch to Dot only with larger dot size
    live_maps.cbb_plot_style.setCurrentText("Dot")
    live_maps.spb_dotsize.setValue(8.0)
    assert panel.line.get_linestyle() in ("None", "none")
    assert panel.line.get_marker() == "o"
    assert panel.line.get_markersize() == 8.0

    # Switch X and Y scales to Log on strictly positive data
    live_maps.cbb_xscale.setCurrentText("Log")
    live_maps.cbb_yscale.setCurrentText("Log")
    assert panel.ax.get_xscale() == "log"
    assert panel.ax.get_yscale() == "log"

    # Non-positive values gracefully fall back to linear instead of raising
    live_maps.update_profile_data(
        SELECTOR_RED, np.array([-2.0, -1.0, 0.0]), np.array([-5.0, 0.0, -1.0]), axis_label="X", status_msg="OK"
    )
    assert panel.ax.get_xscale() == "linear"
    assert panel.ax.get_yscale() == "linear"

    # Reset restores defaults cleanly
    live_maps.reset_view_options()
    opts = live_maps.get_view_options()
    assert opts["xscale"] == "linear"
    assert opts["yscale"] == "linear"
    assert opts["plot_style"] == "line+dots"


def test_v_live_maps_2d_view_options_zscale_smoothing_colorbar_and_grid(qapp):
    """2D heatmaps support LogNorm Z-scale, bilinear smoothing, colorbar toggle, and grid."""
    live_maps = VLiveMaps(mode="map")
    live_maps.show()
    live_maps.set_selector_active(SELECTOR_RED, True)

    grid = np.array([[1.0, 10.0], [100.0, 1000.0]])
    live_maps.update_map_data(SELECTOR_RED, grid, [1.0, 3.0, 1.0, 3.0], 1.0, 1000.0, "OK")
    panel = live_maps.panels[SELECTOR_RED]
    assert panel.img is not None
    assert panel.cbar is not None
    assert panel.img.get_interpolation() == "none"

    live_maps.cbb_zscale.setCurrentText("Log")
    assert isinstance(panel.img.norm, LogNorm)

    live_maps.act_smoothing.setChecked(True)
    assert panel.img.get_interpolation() == "bilinear"

    live_maps.act_show_colorbar.setChecked(False)
    assert panel.cbar is None

    live_maps.act_show_colorbar.setChecked(True)
    assert panel.cbar is not None

    # Non-positive 2D grid safely avoids invalid LogNorm
    neg_grid = np.array([[-5.0, -1.0], [0.0, -2.0]])
    live_maps.update_map_data(SELECTOR_RED, neg_grid, [0.0, 2.0, 0.0, 2.0], -5.0, 0.0, "OK")
    assert not isinstance(panel.img.norm, LogNorm)


def test_v_live_maps_share_y_and_legend_customization(qapp, monkeypatch):
    """Share Y-axis synchronizes 1D limits and legend editor updates canvas labels, colors, and header chips."""
    live_maps = VLiveMaps(mode="map")
    live_maps.show()
    live_maps.set_selector_active(SELECTOR_RED, True)
    live_maps.set_selector_active(SELECTOR_BLUE, True)

    entries_r = [
        {"name": "p1", "positions": np.array([0.0, 1.0]), "values": np.array([10.0, 20.0]), "is_primary": True, "axis_label": "X", "out_of_range": False, "msg": "OK"},
        {"name": "p2", "positions": np.array([0.0, 1.0]), "values": np.array([15.0, 30.0]), "is_primary": False, "axis_label": "X", "out_of_range": False, "msg": "OK"},
    ]
    entries_b = [
        {"name": "p1", "positions": np.array([0.0, 1.0]), "values": np.array([100.0, 500.0]), "is_primary": True, "axis_label": "X", "out_of_range": False, "msg": "OK"},
        {"name": "p2", "positions": np.array([0.0, 1.0]), "values": np.array([200.0, 1000.0]), "is_primary": False, "axis_label": "X", "out_of_range": False, "msg": "OK"},
    ]
    live_maps.update_multi_profile_data(SELECTOR_RED, entries_r, shared_x_label="X", status_msg="OK")
    live_maps.update_multi_profile_data(SELECTOR_BLUE, entries_b, shared_x_label="X", status_msg="OK")

    panel_r = live_maps.panels[SELECTOR_RED]
    panel_b = live_maps.panels[SELECTOR_BLUE]
    assert panel_r.ax.get_ylim() != panel_b.ax.get_ylim()

    live_maps.act_share_y.setChecked(True)
    assert panel_r.ax.get_ylim() == panel_b.ax.get_ylim()

    # Toggle legend on & verify button + action sync + box creation
    live_maps.btn_legend.click()
    assert live_maps.act_show_legend.isChecked() is True
    assert panel_r._legend is not None

    # Location, columns, font size
    live_maps.cbb_legend_loc.setCurrentText("outside right")
    live_maps.spb_legend_ncol.setValue(2)
    live_maps.spb_legend_fontsize.setValue(11)
    assert panel_r._legend is not None
    assert round(panel_r._legend.get_texts()[0].get_fontsize()) == 11

    # Mock dialog edit via LiveMapsLegendEditorDialog
    def _fake_exec(self):
        self.cbb_loc.setCurrentText("upper left")
        self._row_widgets[0][4].setText("Sample Alpha")
        self._row_widgets[0][3].setCurrentIndex(1)
        return 1

    monkeypatch.setattr(LiveMapsLegendEditorDialog, "exec", _fake_exec)
    live_maps.open_legend_editor()

    texts = [t.get_text() for t in panel_r._legend.get_texts()]
    assert "Sample Alpha" in texts
    assert live_maps._legend_entries[0][0] == "Sample Alpha"
    assert live_maps.get_view_options()["custom_colors"].get("p1") is not None

    # Double-clicking inside rendered legend box emits legend_edit_requested
    edited = []
    panel_r.legend_edit_requested.connect(lambda: edited.append(True))
    panel_r.canvas.draw()
    bbox = panel_r._legend.get_window_extent(renderer=panel_r.canvas.get_renderer())
    cx = 0.5 * (bbox.x0 + bbox.x1)
    cy = 0.5 * (bbox.y0 + bbox.y1)
    fake_event = type("Evt", (), {"dblclick": True, "x": cx, "y": cy})()
    panel_r._on_canvas_button_press(fake_event)
    assert edited == [True]

    # Toggle legend off hides canvas legend box cleanly
    live_maps.btn_legend.click()
    assert panel_r._legend is None

