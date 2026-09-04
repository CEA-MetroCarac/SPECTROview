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
