"""Tests for spectrum-series colors and legend controls."""

from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGroupBox, QLabel

from spectroview import DEFAULT_COLORS
from spectroview.view.components.customized_widgets import CustomizedPalette
from spectroview.view.components.v_spectra_viewer import (
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



