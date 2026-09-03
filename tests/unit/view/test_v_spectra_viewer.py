"""Tests for spectrum-series colors and legend controls."""

import matplotlib as mpl
import numpy as np
import pytest

from spectroview import DEFAULT_COLORS
from spectroview.view.components.customized_widgets import CustomizedPalette
from spectroview.view.components.v_spectra_viewer import (
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
