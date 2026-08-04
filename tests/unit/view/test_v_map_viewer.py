"""Tests for VMapViewer's color-scale (linear/log) option on the heatmap."""
import pandas as pd
from matplotlib.colors import LogNorm

from spectroview.view.components.v_map_viewer import VMapViewer


def _simple_2dmap_df():
    """A minimal, strictly-positive 2x2 map (X, Y, then wavenumber columns)."""
    return pd.DataFrame({
        'X': [0.0, 1.0, 0.0, 1.0],
        'Y': [0.0, 0.0, 1.0, 1.0],
        100.0: [1.0, 2.0, 3.0, 4.0],
        200.0: [2.0, 3.0, 4.0, 5.0],
        300.0: [3.0, 4.0, 5.0, 6.0],
    })


class TestMapViewerColorScale:
    def test_default_is_linear_kwargs(self, qapp):
        viewer = VMapViewer()
        assert viewer.cbb_color_scale.currentText() == "Linear"
        assert viewer._build_color_kwargs(1.0, 10.0) == {"vmin": 1.0, "vmax": 10.0}

    def test_log_scale_builds_lognorm_for_positive_range(self, qapp):
        viewer = VMapViewer()
        viewer.cbb_color_scale.setCurrentText("Log")
        kwargs = viewer._build_color_kwargs(1.0, 100.0)
        assert "vmin" not in kwargs and isinstance(kwargs["norm"], LogNorm)
        assert kwargs["norm"].vmin == 1.0 and kwargs["norm"].vmax == 100.0

    def test_log_scale_falls_back_to_linear_when_not_positive(self, qapp):
        """Log needs strictly-positive data; a <=0 lower bound (fit params,
        clipped intensity) degrades to linear instead of raising."""
        viewer = VMapViewer()
        viewer.cbb_color_scale.setCurrentText("Log")
        assert viewer._build_color_kwargs(0.0, 50.0) == {"vmin": 0.0, "vmax": 50.0}
        assert viewer._build_color_kwargs(-5.0, 50.0) == {"vmin": -5.0, "vmax": 50.0}

    def test_plotted_2dmap_image_uses_lognorm_when_log_selected(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))

        viewer = VMapViewer()
        viewer.cbb_map_type.setCurrentText("2Dmap")
        viewer.set_map_data(_simple_2dmap_df(), "m1")
        viewer.cbb_color_scale.setCurrentText("Log")
        viewer._do_plot_heatmap()  # bypass the 100ms debounce timer

        assert viewer.img is not None
        assert isinstance(viewer.img.norm, LogNorm)
