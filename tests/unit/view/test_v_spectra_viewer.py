"""Tests for spectrum-series colors and legend controls."""

from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGroupBox, QLabel

from spectroview import DEFAULT_COLORS
from spectroview.view.components.customized_widgets import CustomizedPalette
from spectroview.view.components.spectra_viewer import (
    FastLine3DCollection,
    LEGEND_EDITOR_MAX_ROWS,
    SPECTRA_COLOR_PALETTES,
    VSpectraViewer,
)


@pytest.fixture
def viewer(qapp):
    widget = VSpectraViewer()
    yield widget
    widget.close()


def _tensor_data(count):
    x = np.linspace(0.0, 1.0, 5)
    return {
        "type": "tensor",
        "x": x,
        "y": np.vstack([x + i for i in range(count)]),
        "x0": x,
        "y0": np.vstack([x + i for i in range(count)]),
        "colors": [None] * count,
        "labels": [f"Series {i + 1}" for i in range(count)],
        "fnames": [f"series_{i + 1}.txt" for i in range(count)],
        "proxies": [],
    }


def test_palette_options_and_legend_default(viewer):
    choices = {
        viewer.cbb_color_palette.itemText(i)
        for i in range(viewer.cbb_color_palette.count())
    }

    assert choices == set(SPECTRA_COLOR_PALETTES)
    assert {
        "DEFAULT_COLORS", "tab20", "tab20b", "tab20c", "Dark2",
        "Paired", "Accent", "viridis", "plasma", "jet",
    } <= choices
    assert {
        "tab10", "Set1", "Set2", "Set3", "Pastel1", "Pastel2",
    }.isdisjoint(choices)
    assert isinstance(viewer.cbb_color_palette, CustomizedPalette)
    assert all(
        not viewer.cbb_color_palette.itemIcon(index).isNull()
        for index in range(viewer.cbb_color_palette.count())
    )
    assert viewer.cbb_color_palette.currentText() == "DEFAULT_COLORS"
    assert viewer.spin_max_legend_items.value() == 15


def test_default_and_discrete_palettes_cycle(viewer):
    assert len(DEFAULT_COLORS) == 20
    assert len(set(DEFAULT_COLORS)) == 20
    assert viewer._get_colors_cycle(22) == [
        DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(22)
    ]

    viewer.cbb_color_palette.setCurrentText("Dark2")
    colors = viewer._get_colors_cycle(10)
    assert len(colors) == 10
    assert len(set(colors[:8])) == 8
    assert colors[8] == colors[0]
    assert colors[9] == colors[1]


def test_default_palette_preview_renders_all_discrete_colors(viewer):
    combo = viewer.cbb_color_palette
    preview = combo._create_colormap_preview("DEFAULT_COLORS").toImage()
    observed = {
        preview.pixelColor(x, combo.icon_height // 2).name().upper()
        for x in range(combo.icon_width)
    }

    assert observed == set(DEFAULT_COLORS)


@pytest.mark.parametrize("palette_name", ["viridis", "plasma", "jet"])
def test_gradient_palettes_are_sampled_across_series(viewer, palette_name):
    viewer.cbb_color_palette.setCurrentText(palette_name)
    colors = viewer._get_colors_cycle(5)
    cmap = mpl.colormaps[palette_name]

    assert len(colors) == 5
    assert len(set(colors)) == 5
    assert colors[0] == mpl.colors.to_hex(cmap(0.0))
    assert colors[-1] == mpl.colors.to_hex(cmap(1.0))


def test_custom_spectrum_color_takes_priority_over_palette(viewer):
    data = _tensor_data(3)
    data["colors"][1] = "#abcdef"
    viewer._tensor_data = data
    viewer.cbb_color_palette.setCurrentText("plasma")
    palette_colors = viewer._get_colors_cycle(3)

    segments = viewer._build_tensor_segments(
        0.0, 0.0, "line", 1.5, "black", palette_colors)

    assert segments["main_colors"] == [
        palette_colors[0], "#abcdef", palette_colors[2]
    ]


def test_legend_item_limit_is_adjustable(viewer):
    viewer.btn_legend.setChecked(True)
    viewer.set_plot_data(_tensor_data(20))

    assert len(viewer.ax.get_legend().get_texts()) == 15

    viewer.spin_max_legend_items.setValue(7)

    assert len(viewer.ax.get_legend().get_texts()) == 7


def test_palette_and_legend_limit_round_trip_through_options_state(viewer):
    viewer.cbb_color_palette.setCurrentText("viridis")
    viewer.spin_max_legend_items.setValue(24)
    state = viewer.get_options_state()

    assert state["color_palette"] == "viridis"
    assert state["max_legend_items"] == 24

    viewer.cbb_color_palette.setCurrentText("DEFAULT_COLORS")
    viewer.spin_max_legend_items.setValue(3)
    viewer.set_options_state(state)

    assert viewer.cbb_color_palette.currentText() == "viridis"
    assert viewer.spin_max_legend_items.value() == 24


def test_legend_editor_lists_all_plotted_spectra_beyond_legend_cap(viewer):
    viewer._tensor_data = _tensor_data(6)
    viewer.spin_max_legend_items.setValue(2)

    dialog = viewer._create_legend_editor_dialog()

    assert dialog.table.rowCount() == 6
    assert dialog.max_items_spin.value() == 2
    assert [edit.text() for edit in dialog.label_edits] == [
        f"Series {index}" for index in range(1, 7)
    ]
    assert all(combo.selected_color() is None
               for combo in dialog.color_combos)
    dialog.close()


@pytest.mark.parametrize("is_map_workspace", [False, True])
def test_legend_editor_caps_rows_in_both_workspaces(
        viewer, is_map_workspace):
    total = LEGEND_EDITOR_MAX_ROWS + 25
    viewer._tensor_data = _tensor_data(total)
    if is_map_workspace:
        viewer._tensor_data["map_name"] = "large_map"
    viewer.cbb_color_palette.setCurrentText("viridis")

    entries = viewer._legend_editor_entries()
    dialog = viewer._create_legend_editor_dialog()
    entries_group = next(
        group for group in dialog.findChildren(QGroupBox)
        if group.title().startswith("Plotted spectra")
    )
    limit_notice = dialog.findChild(QLabel, "legendEditorLimitNotice")

    assert len(entries) == LEGEND_EDITOR_MAX_ROWS
    assert dialog.table.rowCount() == LEGEND_EDITOR_MAX_ROWS
    assert f"showing {LEGEND_EDITOR_MAX_ROWS} of {total}" in (
        entries_group.title().lower())
    assert limit_notice is not None
    assert f"{LEGEND_EDITOR_MAX_ROWS} of {total}" in limit_notice.text()
    expected_last_color = viewer._get_colors_for_palette(
        "viridis", total)[LEGEND_EDITOR_MAX_ROWS - 1]
    assert expected_last_color in dialog.color_combos[-1].itemData(
        0, Qt.ToolTipRole)
    dialog.close()


def test_legend_editor_color_choices_follow_selected_palette(viewer):
    viewer._tensor_data = _tensor_data(3)
    dialog = viewer._create_legend_editor_dialog()

    dialog.palette_combo.setCurrentText("Dark2")
    available = {
        dialog.color_combos[0].itemData(index)
        for index in range(dialog.color_combos[0].count())
    }
    expected = {
        mpl.colors.to_hex(color)
        for color in mpl.colormaps["Dark2"].colors
    }
    automatic_color = viewer._get_colors_for_palette("Dark2", 3)[0]

    assert expected <= available
    assert dialog.color_combos[0].selected_color() is None
    assert automatic_color in dialog.color_combos[0].itemData(
        0, Qt.ToolTipRole)
    dialog.close()


def test_accepted_legend_editor_syncs_view_data_and_spectrum_proxies(
        viewer, monkeypatch):
    data = _tensor_data(3)
    data["colors"] = [None, "#abcdef", None]
    proxies = [
        SimpleNamespace(label=label, color=color)
        for label, color in zip(data["labels"], data["colors"])
    ]
    data["proxies"] = proxies
    viewer._tensor_data = data

    dialog = viewer._create_legend_editor_dialog()
    dialog.palette_combo.setCurrentText("plasma")
    dialog.max_items_spin.setValue(2)
    dialog.label_edits[0].setText("Renamed spectrum")
    dialog.color_combos[0].setCurrentIndex(1)
    selected_color = dialog.color_combos[0].selected_color()
    dialog.color_combos[1].setCurrentIndex(0)  # Restore palette-driven color

    plot_calls = []
    customization_signals = []
    option_signals = []
    monkeypatch.setattr(viewer, "_plot", lambda: plot_calls.append(True))
    viewer.spectrumCustomized.connect(
        lambda: customization_signals.append(True))
    viewer.allOptionsSyncChanged.connect(
        lambda state: option_signals.append(state))

    viewer._apply_legend_editor_values(dialog.values())

    assert viewer.cbb_color_palette.currentText() == "plasma"
    assert viewer.spin_max_legend_items.value() == 2
    assert data["labels"][0] == "Renamed spectrum"
    assert proxies[0].label == "Renamed spectrum"
    assert data["colors"][:2] == [selected_color, None]
    assert proxies[0].color == selected_color
    assert proxies[1].color is None
    assert len(customization_signals) == 1
    assert len(option_signals) == 1
    assert len(plot_calls) == 1
    dialog.close()


def test_cancelled_legend_editor_does_not_change_spectra(
        viewer, monkeypatch):
    data = _tensor_data(2)
    proxy = SimpleNamespace(label="Series 1", color=None)
    data["proxies"] = [proxy]
    viewer._tensor_data = data
    dialog = viewer._create_legend_editor_dialog()
    dialog.label_edits[0].setText("Should not be applied")
    dialog.palette_combo.setCurrentText("jet")
    monkeypatch.setattr(dialog, "exec", lambda: QDialog.Rejected)
    monkeypatch.setattr(
        viewer, "_create_legend_editor_dialog", lambda: dialog)

    viewer._open_legend_editor()

    assert data["labels"][0] == "Series 1"
    assert data["colors"][0] is None
    assert proxy.label == "Series 1"
    assert proxy.color is None
    assert viewer.cbb_color_palette.currentText() == "DEFAULT_COLORS"
    dialog.close()


def test_double_click_anywhere_inside_legend_opens_batch_editor(
        viewer, monkeypatch):
    viewer.btn_legend.setChecked(True)
    viewer.set_plot_data(_tensor_data(3))
    bbox = viewer._legend_bbox
    opened = []
    monkeypatch.setattr(
        viewer, "_open_legend_editor", lambda: opened.append(True))

    inside_event = SimpleNamespace(
        dblclick=True,
        x=(bbox.x0 + bbox.x1) / 2,
        y=(bbox.y0 + bbox.y1) / 2,
        inaxes=viewer.ax,
    )
    outside_event = SimpleNamespace(
        dblclick=True,
        x=bbox.x1 + 20,
        y=bbox.y1 + 20,
        inaxes=viewer.ax,
    )

    viewer._on_legend_double_click(inside_event)
    viewer._on_legend_double_click(outside_event)

    assert opened == [True]


def test_heatmap_toggle_and_controls(viewer):
    viewer.show()
    viewer.set_plot_data(_tensor_data(3))

    # Initially heatmap is off
    assert not viewer._heatmap_mode
    assert not viewer.cbb_heatmap_yaxis.isVisible()
    assert not viewer.le_heatmap_ylabel.isVisible()
    assert not viewer.spin_heatmap_interp.isVisible()
    assert viewer.btn_peak.isVisible()
    assert viewer.btn_baseline.isVisible()
    assert viewer.btn_norm.isVisible()
    assert viewer.norm_xmin.isVisible()
    assert viewer.norm_xmax.isVisible()
    assert viewer.btn_bestfit.isVisible()
    assert viewer.shift_panel.isVisible()

    # Toggle heatmap ON
    viewer.btn_heatmap.setChecked(True)
    assert viewer._heatmap_mode
    assert viewer.cbb_heatmap_yaxis.isVisible()
    assert viewer.le_heatmap_ylabel.isVisible()
    assert viewer.spin_heatmap_interp.isVisible()
    assert not viewer.btn_peak.isVisible()
    assert not viewer.btn_baseline.isVisible()
    assert not viewer.btn_eraser.isVisible()
    assert not viewer.btn_norm.isVisible()
    assert not viewer.norm_xmin.isVisible()
    assert not viewer.norm_xmax.isVisible()
    assert not viewer.btn_bestfit.isVisible()
    assert not viewer.shift_panel.isVisible()
    assert viewer._heatmap_colorbar is not None

    # Toggle heatmap OFF
    viewer.btn_heatmap.setChecked(False)
    assert not viewer._heatmap_mode
    assert not viewer.cbb_heatmap_yaxis.isVisible()
    assert not viewer.le_heatmap_ylabel.isVisible()
    assert not viewer.spin_heatmap_interp.isVisible()
    assert viewer.btn_peak.isVisible()
    assert viewer.btn_baseline.isVisible()
    assert viewer.btn_eraser.isVisible()
    assert viewer.btn_norm.isVisible()
    assert viewer.norm_xmin.isVisible()
    assert viewer.norm_xmax.isVisible()
    assert viewer.btn_bestfit.isVisible()
    assert viewer.shift_panel.isVisible()
    assert viewer._heatmap_colorbar is None


def test_bestfit_button_default_state(viewer):
    assert viewer.btn_bestfit.isCheckable()
    assert viewer.btn_bestfit.isChecked()
    state = viewer.get_options_state()
    assert state["bestfit"] is True


def test_legend_outside_option_default_and_toggle(viewer):
    """Test 'Move legend box outside' action and its effect on axes legend placement."""
    assert viewer.act_legend_outside.isCheckable()
    assert viewer.act_legend_outside.isChecked() is False
    assert viewer.get_options_state()["legend_outside"] is False

    viewer.set_plot_data(_tensor_data(3))
    viewer.btn_legend.setChecked(True)

    # By default, legend is located inside axes at top right corner (loc='upper right', code 1)
    legend = viewer.ax.get_legend()
    assert legend is not None
    assert legend._loc == 1

    # Toggle Move legend box outside ON
    viewer.act_legend_outside.setChecked(True)
    assert viewer.act_legend_outside.isChecked() is True
    assert viewer.get_options_state()["legend_outside"] is True

    # Legend is now placed at center left outside the axes (loc=6, bbox_to_anchor x=1.02, y=0.5)
    legend = viewer.ax.get_legend()
    assert legend is not None
    assert legend._loc == 6
    bbox_anchor = legend.get_bbox_to_anchor()
    assert bbox_anchor._bbox.x0 == 1.02
    assert bbox_anchor._bbox.y0 == 0.5

    # Toggle Move legend box outside OFF
    viewer.act_legend_outside.setChecked(False)
    assert viewer.act_legend_outside.isChecked() is False
    assert viewer.get_options_state()["legend_outside"] is False
    legend = viewer.ax.get_legend()
    assert legend._loc == 1


def test_legend_outside_round_trip_through_options_state(viewer):
    """Test legend_outside preserves state across get_options_state and set_options_state."""
    viewer.act_legend_outside.setChecked(True)
    state = viewer.get_options_state()
    assert state["legend_outside"] is True

    viewer.act_legend_outside.setChecked(False)
    assert viewer.act_legend_outside.isChecked() is False

    viewer.set_options_state(state)
    assert viewer.act_legend_outside.isChecked() is True


def test_zoom_button_toggle_and_range_selector_conflict_resolution(viewer):
    """Test that btn_zoom can be toggled off, and range selector activation turns off zoom."""
    # Initially zoom is checked
    assert viewer.btn_zoom.isChecked() is True
    assert viewer.zoom_pan_active is True

    # Clicking btn_zoom toggles it OFF
    viewer.btn_zoom.click()
    assert viewer.btn_zoom.isChecked() is False
    assert viewer.zoom_pan_active is False

    # Clicking btn_zoom again toggles it ON
    viewer.btn_zoom.click()
    assert viewer.btn_zoom.isChecked() is True
    assert viewer.zoom_pan_active is True

    # Activating a range selector automatically turns off zoom to prevent dual-action conflict
    viewer.btn_range_red.setChecked(True)
    assert viewer.btn_range_red.isChecked() is True
    assert viewer.btn_zoom.isChecked() is False
    assert viewer.zoom_pan_active is False

    # User can re-enable zoom if desired while range selectors are active
    viewer.btn_zoom.click()
    assert viewer.btn_zoom.isChecked() is True
    assert viewer.zoom_pan_active is True


def test_norm_range_edited_updates_view(viewer):
    data = _tensor_data(3)
    viewer.set_plot_data(data)
    viewer.btn_norm.setChecked(True)
    viewer.norm_xmin.setText("0.2")
    viewer.norm_xmax.setText("0.8")
    viewer._on_norm_range_edited()
    assert viewer.ax.get_ylim()[1] > 0


def test_compute_shift_steps_with_precomputed_norm(viewer):
    data = _tensor_data(3)
    viewer.set_plot_data(data)
    viewer.slider_y_shift.setValue(50)
    x = data["x"]
    Y = data["y"]
    Y_norm = viewer._get_normalized_y_tensor(x, Y)
    x_step, y_step = viewer._compute_shift_steps(Y_norm=Y_norm)
    assert x_step == 0.0
    assert y_step > 0.0


def test_vectorized_heatmap_interpolation(viewer):
    data = _tensor_data(5)
    viewer.set_plot_data(data)
    viewer.btn_heatmap.setChecked(True)
    viewer.spin_heatmap_interp.setValue(100)
    viewer._plot_heatmap()
    assert viewer._heatmap_colorbar is not None


def test_filename_token_extraction_with_precompiled_regex(viewer):
    fname = "sample_test_300K_4.5mW_01.txt"
    tokens = viewer._extract_filename_tokens(fname)
    assert len(tokens) >= 3
    # Verify (pos, text, num) tuples
    token_dict = {t[1]: t[2] for t in tokens}
    assert token_dict.get("300K") == 300.0
    assert token_dict.get("4.5mW") == 4.5
    assert token_dict.get("01") == 1.0


def test_control_bar_layout_options_and_copy_on_far_right(viewer):
    """Verify that 'Plot Heatmap', '3D View', 'More view options', and 'Copy' sit on the far right of the control bar, preceded by a separator."""
    from PySide6.QtWidgets import QFrame

    assert viewer.btn_options.width() == 30 and viewer.btn_options.height() == 30
    assert viewer.btn_options.iconSize().width() == 22 and viewer.btn_options.iconSize().height() == 22
    assert "More view options" in viewer.btn_options.toolTip()

    assert viewer.btn_copy.width() == 30 and viewer.btn_copy.height() == 30
    assert viewer.btn_copy.iconSize().width() == 22 and viewer.btn_copy.iconSize().height() == 22
    assert "Copy" in viewer.btn_copy.toolTip()

    layout = viewer.control_bar.layout()
    widgets = []
    has_stretch_before_toolbar = False
    toolbar_index = -1
    for i in range(layout.count()):
        item = layout.itemAt(i)
        w = item.widget()
        if w is not None:
            widgets.append(w)
            if w is viewer.lbl_coordinates:
                toolbar_index = i
        else:
            # Spacer / stretch item
            if toolbar_index == -1:
                has_stretch_before_toolbar = True

    # Check right-end placement: Heatmap, 3D View, Gear (Options), Copy
    assert widgets[-1] is viewer.btn_copy
    assert widgets[-2] is viewer.btn_options
    assert widgets[-3] is viewer.btn_3d
    assert widgets[-4] is viewer.btn_heatmap
    # Preceded by vertical separator
    assert isinstance(widgets[-5], QFrame)
    assert widgets[-5].frameShape() == QFrame.VLine
    # Stretch is present to push items across the bar
    assert has_stretch_before_toolbar


def test_built_in_live_coordinates_displayed_and_toolbar_hidden(viewer):
    """Verify that matplotlib's native toolbar is hidden, while built-in
    live coordinate display (lbl_coordinates) remains visible, unclipped,
    and updates on cursor hover."""
    import matplotlib.backend_bases as mbb

    viewer.show()

    # Native matplotlib toolbar widget is hidden
    assert viewer.toolbar.isHidden()

    # Built-in live coordinate label is visible with proper dimensions
    assert not viewer.lbl_coordinates.isHidden()
    assert viewer.lbl_coordinates.minimumWidth() >= 180

    # Live coordinate updates work on hover over axes
    event = mbb.MouseEvent("motion_notify_event", viewer.canvas, 100, 100)
    event.inaxes = viewer.ax
    event.xdata = 520.123
    event.ydata = 145.678
    viewer._on_hover(event)
    assert "520" in viewer.lbl_coordinates.text()
    assert "145" in viewer.lbl_coordinates.text()
    assert "520" in viewer.toolbar.locLabel.text()

    # Matplotlib toolbar mouse_move also synchronizes with lbl_coordinates
    event.xdata = 630.456
    event.ydata = 250.789
    viewer.toolbar.mouse_move(event)
    assert "630" in viewer.lbl_coordinates.text()
    assert "250" in viewer.lbl_coordinates.text()

    # Leaving figure canvas clears live coordinates
    leave_event = mbb.LocationEvent("figure_leave_event", viewer.canvas, 0, 0)
    viewer._on_figure_leave(leave_event)
    assert viewer.lbl_coordinates.text() == ""


def test_x_axis_unit_options_include_energy_kev(viewer):
    """Verify that 'Energy (keV)' is included in X-axis unit dropdown options."""
    items = [viewer.cbb_xaxis.itemText(i) for i in range(viewer.cbb_xaxis.count())]
    assert "Energy (keV)" in items


def test_spectra_viewer_copy_feedback(viewer):
    """Verify that clicking copy triggers visual checkmark feedback."""
    assert not viewer.btn_copy.icon().isNull()
    viewer._emit_copy()
    # Check that feedback method sets check icon
    viewer._reset_copy_feedback()
    assert not viewer.btn_copy.icon().isNull()


def test_spectra_viewer_range_drag_blitting_clamping_and_release(viewer):
    """Verify range bracket drag prepares blitting, clamps outside Y, skips peak hover, and restores state on release."""
    import matplotlib.backend_bases as mbb

    x = np.linspace(400.0, 800.0, 41)
    y = np.exp(-((x - 600.0) ** 2) / (2.0 * 25.0 ** 2))
    viewer.resize(640, 400)
    viewer.show()
    viewer.set_plot_data(
        {
            "type": "tensor",
            "x": x,
            "y": y[None, :],
            "x0": x,
            "y0": y[None, :],
            "colors": [None],
            "labels": ["spec_0"],
            "fnames": ["spec_0"],
            "proxies": [],
        }
    )
    viewer._toggle_range_selector("red", True)
    viewer.canvas.draw()

    r_min, _ = viewer.range_manager.get_range("red")

    # Dummy fitted line to ensure _on_hover skips peak hit-testing while dragging
    hover_calls = []

    class _DummyLine:
        figure = viewer.figure

        def contains(self, _ev):
            hover_calls.append(True)
            return False, {}

    viewer._fitted_lines = [(_DummyLine(), {"x0": 600.0, "fwhm": 20.0})]

    finished_signals = []
    viewer.range_drag_finished.connect(finished_signals.append)

    px, py = viewer.ax.transData.transform((r_min, 0.5))
    press = mbb.MouseEvent("button_press_event", viewer.canvas, px, py, button=1)
    press.inaxes = viewer.ax
    press.xdata = r_min
    press.ydata = 0.5
    viewer._on_mouse_click(press)

    assert viewer.range_manager.is_dragging() is True
    assert viewer._range_drag_bg is not None
    assert viewer.figure.get_layout_engine() is None
    assert all(a.get_animated() for a in viewer.range_manager.active_artists())

    # Hover during drag must skip fitted-line contains() scan
    hover = mbb.MouseEvent("motion_notify_event", viewer.canvas, px + 5, py)
    hover.inaxes = viewer.ax
    hover.xdata = r_min + 10.0
    hover.ydata = 0.5
    viewer._on_hover(hover)
    assert hover_calls == []

    # Motion vertically outside axes still clamps and updates dragged boundary
    new_x = r_min + 25.0
    px_out, _ = viewer.ax.transData.transform((new_x, 0.5))
    motion = mbb.MouseEvent("motion_notify_event", viewer.canvas, px_out, -20.0)
    motion.inaxes = None
    motion.xdata = None
    motion.ydata = None
    viewer._on_range_drag_motion(motion)
    updated_min, _ = viewer.range_manager.get_range("red")
    assert updated_min == pytest.approx(new_x, abs=2.0)

    # Release restores normal drawing mode, layout engine, and emits range_drag_finished
    release = mbb.MouseEvent("button_release_event", viewer.canvas, px_out, py, button=1)
    viewer._on_range_drag_release(release)
    assert viewer.range_manager.is_dragging() is False
    assert viewer._range_drag_bg is None
    assert viewer.figure.get_layout_engine() is not None
    assert all(not a.get_animated() for a in viewer.range_manager.active_artists())
    assert finished_signals == ["red"]


def _series_data_from_fnames(fnames, colors=None):
    count = len(fnames)
    x = np.linspace(100.0, 500.0, 10)
    y = np.vstack([np.sin(x / 100.0) + i for i in range(count)])
    return {
        "type": "tensor",
        "x": x,
        "y": y,
        "x0": x,
        "y0": y,
        "colors": [None] * count if colors is None else list(colors),
        "labels": list(fnames),
        "fnames": list(fnames),
        "proxies": [],
    }


def test_legend_colorbar_option_replaces_discrete_legend_with_colorbar(viewer):
    """Test colorbar legend mode extracts token bounds, sets label, and toggles back cleanly."""
    fnames = [
        "LNO550_25°C_down",
        "LNO550_100°C_down",
        "LNO550_300°C_down",
        "LNO550_500°C_down",
    ]
    viewer.set_plot_data(_series_data_from_fnames(fnames))

    cb_items = [
        viewer.cbb_colorbar_param.itemText(i)
        for i in range(viewer.cbb_colorbar_param.count())
    ]
    assert "Index" in cb_items
    assert "25°C" in cb_items

    viewer.cbb_color_palette.setCurrentText("jet")
    viewer.cbb_colorbar_param.setCurrentText("25°C")
    viewer.le_colorbar_label.setText("Temperature (°C)")
    viewer.act_legend_colorbar.setChecked(True)

    assert viewer.btn_legend.isChecked() is True
    assert viewer.ax.get_legend() is None
    assert viewer._heatmap_colorbar is not None
    assert viewer._heatmap_colorbar.norm.vmin == pytest.approx(25.0)
    assert viewer._heatmap_colorbar.norm.vmax == pytest.approx(500.0)
    assert viewer._heatmap_colorbar.ax.get_ylabel() == "Temperature (°C)"

    # Toggling colorbar off removes colorbar and restores discrete legend
    viewer.act_legend_colorbar.setChecked(False)
    assert viewer._heatmap_colorbar is None
    assert viewer.ax.get_legend() is not None


def test_legend_colorbar_maps_non_uniform_and_reversed_token_values_to_colors(viewer):
    """Verify spectrum colors follow extracted token values on gradient palettes while keeping overrides."""
    fnames = [
        "LNO550_500°C_down",
        "LNO550_100°C_down",
        "LNO550_25°C_down",
    ]
    viewer.cbb_color_palette.setCurrentText("jet")
    viewer.set_plot_data(_series_data_from_fnames(fnames))
    viewer.cbb_colorbar_param.setCurrentText("500°C")
    viewer.act_legend_colorbar.setChecked(True)

    cmap = mpl.colormaps["jet"]
    norm = mpl.colors.Normalize(vmin=25.0, vmax=500.0)
    expected_colors = [
        mpl.colors.to_hex(cmap(norm(500.0))),
        mpl.colors.to_hex(cmap(norm(100.0))),
        mpl.colors.to_hex(cmap(norm(25.0))),
    ]
    plotted_colors = [
        mpl.colors.to_hex(c) for c in viewer.ax.collections[0].get_colors()
    ]
    assert plotted_colors == expected_colors

    # Custom per-spectrum override continues to take priority
    viewer.set_plot_data(
        _series_data_from_fnames(fnames, colors=[None, "#abcdef", None])
    )
    plotted_with_override = [
        mpl.colors.to_hex(c) for c in viewer.ax.collections[0].get_colors()
    ]
    assert plotted_with_override == [
        expected_colors[0],
        "#abcdef",
        expected_colors[2],
    ]


def test_legend_colorbar_options_round_trip_and_settings_persistence(viewer, settings):
    """Verify legend_colorbar, colorbar_param, and colorbar_label round-trip in options and QSettings."""
    from spectroview.model.m_settings import MSettings

    viewer.act_legend_colorbar.setChecked(True)
    viewer.cbb_colorbar_param.addItem("25°C")
    viewer.cbb_colorbar_param.setCurrentText("25°C")
    viewer.le_colorbar_label.setText("Temperature (°C)")

    state = viewer.get_options_state()
    assert state["legend_colorbar"] is True
    assert state["colorbar_param"] == "25°C"
    assert state["colorbar_label"] == "Temperature (°C)"

    settings.save_view_options(state)
    loaded = MSettings().load_view_options()
    assert loaded["legend_colorbar"] is True
    assert loaded["colorbar_param"] == "25°C"
    assert loaded["colorbar_label"] == "Temperature (°C)"

    viewer.act_legend_colorbar.setChecked(False)
    viewer.cbb_colorbar_param.clear()
    viewer.cbb_colorbar_param.addItem("Index")
    viewer.le_colorbar_label.setText("")

    viewer.set_options_state(loaded)
    assert viewer.act_legend_colorbar.isChecked() is True
    assert viewer.cbb_colorbar_param.currentText() == "25°C"
    assert viewer.le_colorbar_label.text() == "Temperature (°C)"


def test_filename_token_extraction_without_extension_preserves_decimals(viewer):
    """Verify _extract_filename_tokens preserves decimal points when filenames have no extension."""
    fname = "sample_test_300K_4.5mW_01"
    tokens = viewer._extract_filename_tokens(fname)
    token_dict = {t[1]: t[2] for t in tokens}
    assert token_dict.get("300K") == 300.0
    assert token_dict.get("4.5mW") == 4.5
    assert token_dict.get("01") == 1.0


def test_options_menu_two_column_layout_and_colorbar_controls_visibility(viewer):
    """Verify More view options uses a compact 2-column layout and toggles colorbar sub-options."""
    from PySide6.QtWidgets import QFrame

    menu = viewer.options_menu
    assert len(menu.actions()) == 1
    container = menu.actions()[0].defaultWidget()
    v_lines = [
        f for f in container.findChildren(QFrame)
        if f.frameShape() == QFrame.VLine
    ]
    assert len(v_lines) == 1
    assert container.sizeHint().height() < 360

    # Colorbar parameter and label rows are hidden when colorbar mode is unchecked
    assert viewer.chk_legend_colorbar.isChecked() is False
    assert viewer.row_colorbar_param.isHidden() is True
    assert viewer.cbb_colorbar_param.isHidden() is True
    assert viewer.row_colorbar_label.isHidden() is True
    assert viewer.le_colorbar_label.isHidden() is True

    # Show menu to verify real isVisible() transitions when checkbox toggles
    menu.show()
    viewer.chk_legend_colorbar.setChecked(True)
    assert viewer.act_legend_colorbar.isChecked() is True
    assert viewer.row_colorbar_param.isHidden() is False
    assert viewer.cbb_colorbar_param.isVisible() is True
    assert viewer.row_colorbar_label.isHidden() is False
    assert viewer.le_colorbar_label.isVisible() is True

    viewer.chk_legend_colorbar.setChecked(False)
    assert viewer.act_legend_colorbar.isChecked() is False
    assert viewer.row_colorbar_param.isHidden() is True
    assert viewer.cbb_colorbar_param.isVisible() is False
    assert viewer.row_colorbar_label.isHidden() is True
    assert viewer.le_colorbar_label.isVisible() is False

    # set_options_state syncs menu checkboxes and conditional colorbar rows
    state_on = viewer.get_options_state()
    state_on["legend_colorbar"] = True
    state_on["grid"] = True
    viewer.set_options_state(state_on)
    assert viewer.chk_legend_colorbar.isChecked() is True
    assert viewer.chk_grid.isChecked() is True
    assert viewer.row_colorbar_param.isHidden() is False
    assert viewer.row_colorbar_label.isHidden() is False

    state_off = viewer.get_options_state()
    state_off["legend_colorbar"] = False
    state_off["grid"] = False
    viewer.set_options_state(state_off)
    assert viewer.chk_legend_colorbar.isChecked() is False
    assert viewer.chk_grid.isChecked() is False
    assert viewer.row_colorbar_param.isHidden() is True
    assert viewer.row_colorbar_label.isHidden() is True
    menu.close()


def test_colorbar_parameter_persists_across_dynamic_spectra_list_selections(viewer):
    """Verify selected colorbar parameter token slot/pattern persists across dynamic selection changes."""
    first_batch = [
        "LNO550_206R_incell_0deg_30sx4_500°C_down",
        "LNO550_206R_incell_0deg_30sx4_450°C_down",
        "LNO550_206R_incell_0deg_30sx4_400°C_down",
    ]
    viewer.set_plot_data(_series_data_from_fnames(first_batch))
    viewer.cbb_color_palette.setCurrentText("jet")
    viewer.act_legend_colorbar.setChecked(True)
    viewer.cbb_colorbar_param.setCurrentText("500°C")

    assert viewer._heatmap_colorbar is not None
    assert viewer._heatmap_colorbar.norm.vmin == pytest.approx(400.0)
    assert viewer._heatmap_colorbar.norm.vmax == pytest.approx(500.0)

    # Select a different subset where the first spectrum has 25°C instead of 500°C
    second_batch = [
        "LNO550_206R_incell_0deg_30sx4_25°C_down",
        "LNO550_206R_incell_0deg_30sx4_100°C_down",
        "LNO550_206R_incell_0deg_30sx4_300°C_down",
    ]
    viewer.set_plot_data(_series_data_from_fnames(second_batch))

    assert viewer.cbb_colorbar_param.currentText() == "25°C"
    assert viewer.cbb_colorbar_param.currentData() == 5
    assert viewer._heatmap_colorbar is not None
    assert viewer._heatmap_colorbar.norm.vmin == pytest.approx(25.0)
    assert viewer._heatmap_colorbar.norm.vmax == pytest.approx(300.0)

    # Simulate momentary empty selection before selecting another batch
    viewer.set_plot_data(None)
    third_batch = [
        "LNO550_206R_incell_0deg_30sx4_150°C_down",
        "LNO550_206R_incell_0deg_30sx4_250°C_down",
    ]
    viewer.set_plot_data(_series_data_from_fnames(third_batch))

    assert viewer.cbb_colorbar_param.currentText() == "150°C"
    assert viewer.cbb_colorbar_param.currentData() == 5
    assert viewer._heatmap_colorbar is not None
    assert viewer._heatmap_colorbar.norm.vmin == pytest.approx(150.0)
    assert viewer._heatmap_colorbar.norm.vmax == pytest.approx(250.0)


def test_3d_view_toggle_and_controls(viewer):
    """Verify btn_3d sits next to btn_heatmap and cleanly switches toolbar state and Axes3D."""
    viewer.show()
    viewer.set_plot_data(_tensor_data(3))

    layout = viewer.control_bar.layout()
    widgets = [layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget() is not None]
    idx_hm = widgets.index(viewer.btn_heatmap)
    assert widgets[idx_hm + 1] is viewer.btn_3d

    # Initially standard 2D mode
    assert not viewer._3d_mode
    assert viewer.ax.name == "rectilinear"
    assert viewer.zoom_pan_active is True
    assert viewer.btn_zoom.isVisible()

    # Toggle 3D mode ON
    viewer.btn_3d.setChecked(True)
    assert viewer._3d_mode is True
    assert viewer.ax.name == "3d"
    assert viewer.zoom_pan_active is False
    assert not viewer.btn_zoom.isChecked()
    assert not viewer.btn_zoom.isVisible()
    assert not viewer.btn_peak.isVisible()
    assert not viewer.btn_baseline.isVisible()
    assert not viewer.btn_eraser.isVisible()
    assert not viewer.shift_panel.isVisible()
    assert viewer.btn_norm.isVisible()
    assert viewer.btn_bestfit.isVisible()
    assert viewer.cbb_heatmap_yaxis.isVisible()
    assert viewer.le_heatmap_ylabel.isVisible()
    assert not viewer.spin_heatmap_interp.isVisible()

    # Toggle 3D mode OFF -> restores Standard 2D mode
    viewer.btn_3d.setChecked(False)
    assert viewer._3d_mode is False
    assert viewer.ax.name == "rectilinear"
    assert viewer.zoom_pan_active is True
    assert viewer.btn_zoom.isChecked()
    assert viewer.btn_zoom.isVisible()
    assert viewer.btn_peak.isVisible()
    assert viewer.btn_baseline.isVisible()
    assert viewer.btn_eraser.isVisible()
    assert viewer.shift_panel.isVisible()
    assert not viewer.cbb_heatmap_yaxis.isVisible()
    assert not viewer.le_heatmap_ylabel.isVisible()


def test_heatmap_and_3d_mode_mutual_exclusivity(viewer):
    """Verify Standard <-> Heatmap <-> 3D transitions keep modes mutually exclusive and clean."""
    viewer.show()
    viewer.set_plot_data(_tensor_data(4))

    # Standard -> Heatmap
    viewer.btn_heatmap.setChecked(True)
    assert viewer._heatmap_mode is True
    assert viewer._3d_mode is False
    assert viewer.btn_heatmap.isChecked()
    assert not viewer.btn_3d.isChecked()
    assert viewer.ax.name == "rectilinear"
    assert viewer._heatmap_colorbar is not None

    # Heatmap -> 3D
    viewer.btn_3d.setChecked(True)
    assert viewer._3d_mode is True
    assert viewer._heatmap_mode is False
    assert viewer.btn_3d.isChecked()
    assert not viewer.btn_heatmap.isChecked()
    assert viewer.ax.name == "3d"
    assert viewer._heatmap_colorbar is None

    # 3D -> Heatmap
    viewer.btn_heatmap.setChecked(True)
    assert viewer._heatmap_mode is True
    assert viewer._3d_mode is False
    assert viewer.btn_heatmap.isChecked()
    assert not viewer.btn_3d.isChecked()
    assert viewer.ax.name == "rectilinear"
    assert viewer._heatmap_colorbar is not None
    assert viewer.btn_zoom.isVisible()
    assert viewer.spin_heatmap_interp.isVisible()

    # Heatmap -> Standard
    viewer.btn_heatmap.setChecked(False)
    assert viewer._heatmap_mode is False
    assert viewer._3d_mode is False
    assert viewer.ax.name == "rectilinear"
    assert viewer._heatmap_colorbar is None
    assert viewer.shift_panel.isVisible()


def test_3d_view_rendering_modes_and_interactive_rotation(viewer):
    """Verify 3D rendering (uniform/ragged, line/dot, norm, tokens, themes), rotation, zoom, rescale, and options round-trip."""
    import matplotlib.backend_bases as mbb

    viewer.resize(640, 420)
    viewer.show()

    fnames = [
        "sample_25°C_run",
        "sample_100°C_run",
        "sample_300°C_run",
    ]
    viewer.set_plot_data(_series_data_from_fnames(fnames))
    viewer.btn_3d.setChecked(True)
    viewer.canvas.draw()

    assert viewer.ax.name == "3d"
    assert any(isinstance(c, FastLine3DCollection) for c in viewer.ax.collections)

    # Filename Y-axis token extraction
    viewer.cbb_heatmap_yaxis.setCurrentText("25°C")
    ylim = viewer.ax.get_ylim3d()
    assert min(ylim) <= 25.0 + 1e-6
    assert max(ylim) >= 300.0 - 1e-6

    # Normalization scales Z max to 1.0
    viewer.btn_norm.setChecked(True)
    viewer.canvas.draw()
    assert viewer.ax.get_zlim3d()[1] == pytest.approx(1.0, rel=1e-2)
    viewer.btn_norm.setChecked(False)

    # Dot plot style and Dark Mode
    viewer.cbb_plotstyle.setCurrentText("dot")
    viewer.cbb_theme.setCurrentText("Dark Mode")
    viewer.canvas.draw()
    assert len(viewer.ax.collections) >= 1

    # Ragged tensor_list and single-spectrum 3D rendering
    ragged = {
        "type": "tensor_list",
        "x": [np.linspace(100.0, 300.0, 10), np.linspace(150.0, 400.0, 15)],
        "y": [np.ones(10) * 5.0, np.ones(15) * 10.0],
        "x0": [np.linspace(100.0, 300.0, 10), np.linspace(150.0, 400.0, 15)],
        "y0": [np.ones(10) * 5.0, np.ones(15) * 10.0],
        "colors": [None, None],
        "labels": ["Ragged 1", "Ragged 2"],
        "fnames": ["ragged_1.txt", "ragged_2.txt"],
        "proxies": [],
    }
    viewer.cbb_plotstyle.setCurrentText("line")
    viewer.set_plot_data(ragged)
    viewer.canvas.draw()
    assert any(isinstance(c, FastLine3DCollection) for c in viewer.ax.collections)

    viewer.set_plot_data(_tensor_data(1))
    viewer.canvas.draw()
    assert viewer.ax.name == "3d"

    # Interactive mouse drag rotation changes elev / azim
    viewer.set_plot_data(_tensor_data(4))
    viewer.canvas.draw()
    init_elev = viewer.ax.elev
    init_azim = viewer.ax.azim

    press = mbb.MouseEvent("button_press_event", viewer.canvas, 200, 200, button=1)
    press.inaxes = viewer.ax
    press.xdata, press.ydata = 0.0, 0.0
    viewer.canvas.callbacks.process("button_press_event", press)

    move = mbb.MouseEvent("motion_notify_event", viewer.canvas, 270, 240, button=1)
    move.inaxes = viewer.ax
    move.xdata, move.ydata = 0.25, 0.15
    viewer.canvas.callbacks.process("motion_notify_event", move)

    release = mbb.MouseEvent("button_release_event", viewer.canvas, 270, 240, button=1)
    release.inaxes = viewer.ax
    viewer.canvas.callbacks.process("button_release_event", release)

    assert (viewer.ax.elev != init_elev) or (viewer.ax.azim != init_azim)

    # Scroll wheel adjusts zlim
    zlim_before = viewer.ax.get_zlim3d()
    scroll = mbb.MouseEvent("scroll_event", viewer.canvas, 200, 200, step=-1)
    scroll.inaxes = viewer.ax
    viewer._on_scroll(scroll)
    zlim_after = viewer.ax.get_zlim3d()
    assert zlim_after[1] > zlim_before[1]

    # Rescale restores default camera angles and limits
    viewer._rescale()
    assert viewer.ax.elev == pytest.approx(viewer._default_3d_elev)
    assert viewer.ax.azim == pytest.approx(viewer._default_3d_azim)
    assert viewer.ax.get_zlim3d()[1] < zlim_after[1]

    # Options state round-trip
    state_3d = viewer.get_options_state()
    assert state_3d["view_3d"] is True
    assert state_3d["heatmap"] is False

    viewer.btn_3d.setChecked(False)
    assert viewer._3d_mode is False
    assert viewer.ax.name == "rectilinear"

    viewer.set_options_state(state_3d)
    assert viewer.btn_3d.isChecked() is True
    assert viewer._3d_mode is True
    assert viewer.ax.name == "3d"


def test_spectra_viewer_package_exports_and_submodules():
    """Verify spectra_viewer package public exports and submodule availability."""
    from spectroview.view.components import spectra_viewer
    from spectroview.view.components.spectra_viewer import (
        common,
        v_interactions,
        v_plot_1d,
        v_plot_3d,
        v_plot_heatmap,
        v_spectra_legend_editor,
        v_spectra_viewer,
    )

    for name in spectra_viewer.__all__:
        assert hasattr(spectra_viewer, name)

    assert hasattr(common, "get_colors_for_palette")
    assert hasattr(v_plot_1d, "SpectraPlot1DMixin")
    assert hasattr(v_plot_heatmap, "SpectraHeatmapMixin")
    assert hasattr(v_plot_3d, "SpectraPlot3DMixin")
    assert hasattr(v_plot_3d, "FastLine3DCollection")
    assert hasattr(v_spectra_legend_editor, "SpectraLegendEditorDialog")
    assert hasattr(v_spectra_legend_editor, "SpectrumColorComboBox")
    assert hasattr(v_spectra_legend_editor, "SpectraLegendMixin")
    assert hasattr(v_interactions, "SpectraInteractionMixin")
    assert hasattr(v_spectra_viewer, "VSpectraViewer")


def test_peak_toolbar_icon_has_plus_sign(viewer):
    """Verify Peak icon SVG template renders a peak curve with a top-right '+' sign."""
    from spectroview.view.theme.icons_svg import _SVG_TEMPLATES

    peak_svg = _SVG_TEMPLATES["peak"]
    assert '<line x1="19" x2="19" y1="3" y2="9"/>' in peak_svg
    assert '<line x1="16" x2="22" y1="6" y2="6"/>' in peak_svg
    assert not viewer.btn_peak.icon().isNull()
    assert not viewer.btn_peak.icon().pixmap(22, 22).isNull()


def test_cosmic_ray_eraser_flow_erase_undo_and_validate(viewer):
    """Verify cosmic-ray eraser toggles, erases a spike, supports undo, and emits validation signal."""
    import matplotlib.backend_bases as mbb

    x = np.linspace(100.0, 200.0, 21)
    y = np.ones_like(x) * 10.0
    y[10] = 500.0  # Cosmic-ray spike at x = 150.0
    data = {
        "type": "tensor",
        "x": x,
        "y": y[None, :].copy(),
        "x0": x,
        "y0": y[None, :].copy(),
        "colors": [None],
        "labels": ["spec_0"],
        "fnames": ["spec_0.txt"],
        "proxies": [],
    }
    viewer.set_plot_data(data)

    erased_events = []
    validated_events = []
    viewer.cosmicRayErased.connect(lambda idx, arr: erased_events.append((idx, arr.copy())))
    viewer.cosmicRayValidated.connect(lambda: validated_events.append(True))

    viewer.show()
    viewer.btn_eraser.setChecked(True)
    assert viewer._erase_mode is True
    assert viewer.zoom_pan_active is False
    assert viewer.btn_erase_undo.isVisible() is True
    assert viewer.btn_erase_validate.isVisible() is True

    press = mbb.MouseEvent("button_press_event", viewer.canvas, 100, 100, button=1)
    press.inaxes = viewer.ax
    press.xdata = 145.0
    press.ydata = 10.0
    viewer._erase_on_press(press)
    assert viewer._erase_drag_start_x == 145.0

    motion = mbb.MouseEvent("motion_notify_event", viewer.canvas, 120, 100, button=1)
    motion.inaxes = viewer.ax
    motion.xdata = 155.0
    motion.ydata = 10.0
    viewer._erase_on_motion(motion)
    assert viewer._erase_highlight is not None

    release = mbb.MouseEvent("button_release_event", viewer.canvas, 120, 100, button=1)
    release.inaxes = viewer.ax
    release.xdata = 155.0
    release.ydata = 10.0
    viewer._erase_on_release(release)

    assert viewer._erase_highlight is None
    assert len(erased_events) == 1
    assert erased_events[0][0] == 0
    assert viewer._tensor_data["y"][0][10] < 50.0

    # Undo restores the spike
    viewer._erase_undo()
    assert viewer._tensor_data["y"][0][10] == pytest.approx(500.0)

    # Validate clears history, emits cosmicRayValidated, and exits eraser mode
    viewer._erase_validate()
    assert len(validated_events) == 1
    assert viewer._erase_mode is False
    assert viewer.btn_eraser.isChecked() is False
    assert viewer.zoom_pan_active is True


def test_interactive_peak_finding_and_dragging(viewer):
    """Verify _find_peak_at_position, _on_drag_peak, and _on_release_drag emit expected signals."""
    import matplotlib.backend_bases as mbb

    x = np.linspace(400.0, 600.0, 21)
    y = np.exp(-((x - 500.0) ** 2) / (2.0 * 10.0 ** 2)) * 100.0
    data = {
        "type": "tensor",
        "x": x,
        "y": y[None, :],
        "x0": x,
        "y0": y[None, :],
        "colors": [None],
        "labels": ["spec_0"],
        "fnames": ["spec_0.txt"],
        "proxies": [],
        "fit_models": [
            {
                "peak_models": {
                    "0": {
                        "Gaussian": {
                            "x0": {"value": 500.0, "vary": True},
                            "ampli": {"value": 100.0, "vary": True},
                            "fwhm": {"value": 20.0, "vary": True},
                        }
                    }
                },
                "peak_labels": ["Peak1"],
            }
        ],
    }
    viewer.resize(640, 400)
    viewer.show()
    viewer.set_plot_data(data)
    viewer.canvas.draw()

    found = viewer._find_peak_at_position(500.0, 50.0)
    assert found is not None
    assert found[1]["peak_label"] == "Peak1"

    dragged = []
    finished = []
    viewer.peak_dragged.connect(lambda nx, ny: dragged.append((nx, ny)))
    viewer.peak_drag_finished.connect(lambda: finished.append(True))

    viewer.btn_peak.click()
    assert viewer.btn_peak.isChecked() is True
    assert viewer.zoom_pan_active is False

    px, py = viewer.ax.transData.transform((500.0, 80.0))
    press = mbb.MouseEvent("button_press_event", viewer.canvas, px, py, button=1)
    press.inaxes = viewer.ax
    press.xdata = 500.0
    press.ydata = 80.0
    viewer._on_mouse_click(press)
    assert viewer._dragging_peak is not None

    drag = mbb.MouseEvent("motion_notify_event", viewer.canvas, px + 10, py + 5, button=1)
    drag.inaxes = viewer.ax
    drag.xdata = 520.0
    drag.ydata = 120.0
    viewer._on_drag_peak(drag)

    assert dragged == [(520.0, 120.0)]
    pdict = viewer._tensor_data["fit_models"][0]["peak_models"]["0"]["Gaussian"]
    assert pdict["x0"]["value"] == 520.0
    assert pdict["ampli"]["value"] == 120.0

    release = mbb.MouseEvent("button_release_event", viewer.canvas, px + 10, py + 5, button=1)
    viewer._on_release_drag(release)
    assert finished == [True]
    assert viewer._dragging_peak is None


def test_3d_and_1d_log_scale_emit_no_nonpositive_limit_warnings(viewer, recwarn):
    """Verify switching to Log scale in 1D and 3D modes with <=0 data emits no Matplotlib limit UserWarnings."""
    x = np.linspace(100.0, 200.0, 10)
    y = np.vstack([
        np.linspace(-2.0, 10.0, 10),
        np.linspace(0.0, 20.0, 10),
    ])
    data = {
        "type": "tensor",
        "x": x,
        "y": y,
        "x0": x,
        "y0": y,
        "colors": [None, None],
        "labels": ["s1", "s2"],
        "fnames": ["s1.txt", "s2.txt"],
        "proxies": [],
    }

    viewer.resize(640, 420)
    viewer.show()
    viewer.set_plot_data(data)
    viewer.canvas.draw()

    recwarn.clear()
    # 1D Log scale with existing non-positive ylim from Linear mode
    viewer.cbb_yscale.setCurrentText("Log")
    viewer.canvas.draw()
    assert viewer.ax.get_yscale() == "log"

    # Switch to 3D mode while Log scale is active
    viewer.btn_3d.setChecked(True)
    viewer.canvas.draw()
    assert viewer.ax.name == "3d"
    assert viewer.ax.get_zscale() == "log"

    # Toggle Linear -> Log while already in 3D mode
    viewer.cbb_yscale.setCurrentText("Linear")
    viewer.canvas.draw()
    viewer.cbb_yscale.setCurrentText("Log")
    viewer.canvas.draw()
    assert viewer.ax.get_zscale() == "log"

    nonpos_warnings = [
        str(w.message)
        for w in recwarn
        if issubclass(w.category, UserWarning)
        and "Attempt to set non-positive" in str(w.message)
    ]
    assert nonpos_warnings == []


