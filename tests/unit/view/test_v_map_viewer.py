"""Tests for VMapViewer's color-scale (linear/log) option on the heatmap."""
import pandas as pd
from matplotlib.colors import LogNorm

from spectroview import PALETTE
from spectroview.view.components.customized_widgets import CustomizedPalette
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
    def test_palette_preview_widget_keeps_map_palette_set(self, qapp):
        viewer = VMapViewer()

        assert isinstance(viewer.cbb_palette, CustomizedPalette)
        assert [
            viewer.cbb_palette.itemText(index)
            for index in range(viewer.cbb_palette.count())
        ] == PALETTE
        assert all(
            not viewer.cbb_palette.itemIcon(index).isNull()
            for index in range(viewer.cbb_palette.count())
        )

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


class TestMapViewerAutoSelection:
    def test_auto_selects_first_point_on_map_load(self, qapp):
        viewer = VMapViewer()
        df = _simple_2dmap_df()
        viewer.set_map_data(df, "m1")
        assert viewer.selected_points == [(0.0, 0.0)]

    def test_auto_selects_first_point_on_map_switch(self, qapp):
        viewer = VMapViewer()
        df1 = _simple_2dmap_df()
        viewer.set_map_data(df1, "m1")
        assert viewer.selected_points == [(0.0, 0.0)]

        df2 = pd.DataFrame({
            'X': [10.0, 20.0],
            'Y': [30.0, 40.0],
            100.0: [1.0, 2.0],
        })
        viewer.set_map_data(df2, "m2")
        assert viewer.selected_points == [(10.0, 30.0)]

    def test_preserves_selected_point_on_same_map_refresh(self, qapp):
        viewer = VMapViewer()
        df = _simple_2dmap_df()
        viewer.set_map_data(df, "m1")
        # User selects point (1.0, 1.0)
        viewer.set_selected_points([(1.0, 1.0)])
        assert viewer.selected_points == [(1.0, 1.0)]

        # Map data refreshed (e.g. after baseline subtraction on m1)
        viewer.set_map_data(df, "m1")
        assert viewer.selected_points == [(1.0, 1.0)]

    def test_explicit_selected_points_honored(self, qapp):
        viewer = VMapViewer()
        df = _simple_2dmap_df()
        viewer.set_map_data(df, "m1", selected_points=[(1.0, 0.0)])
        assert viewer.selected_points == [(1.0, 0.0)]


class TestMapViewerInterpolation:
    def test_wafer_map_default_interpolation_is_nearest_neighbor(self, qapp):
        viewer = VMapViewer()
        viewer.cbb_map_type.setCurrentText("Wafer_200mm")
        df1 = _simple_2dmap_df()
        viewer.set_map_data(df1, "m1")
        viewer._do_plot_heatmap()

        assert viewer.img is not None
        assert viewer.img.get_interpolation() == 'none'

    def test_wafer_map_switch_maintains_consistent_interpolation(self, qapp):
        viewer = VMapViewer()
        viewer.cbb_map_type.setCurrentText("Wafer_200mm")
        df1 = _simple_2dmap_df()
        viewer.set_map_data(df1, "m1")
        viewer._do_plot_heatmap()
        assert viewer.img.get_interpolation() == 'none'

        df2 = pd.DataFrame({
            'X': [0.0, 2.0, 0.0, 2.0],
            'Y': [0.0, 0.0, 2.0, 2.0],
            100.0: [10.0, 20.0, 30.0, 40.0],
            200.0: [10.0, 20.0, 30.0, 40.0],
        })
        viewer.set_map_data(df2, "m2")
        viewer._do_plot_heatmap()
        assert viewer.img.get_interpolation() == 'none'

    def test_wafer_map_smoothing_toggle(self, qapp):
        viewer = VMapViewer()
        viewer.cbb_map_type.setCurrentText("Wafer_200mm")
        df1 = _simple_2dmap_df()
        viewer.set_map_data(df1, "m1")
        viewer.action_smoothing.setChecked(True)
        viewer._do_plot_heatmap()

        assert viewer.img is not None
        assert viewer.img.get_interpolation() == 'bilinear'


class TestMapViewerOptimization:
    def test_remove_outliers_on_numpy_array_and_masked_data(self, qapp):
        viewer = VMapViewer()
        import numpy as np
        # 100 normal values and 2 extreme outliers as a NumPy ndarray
        data = np.ones(100) * 10.0
        data[0] = 1000.0
        data[1] = -500.0
        filtered = viewer._remove_outliers(data)
        assert isinstance(filtered, (np.ndarray, pd.Series))
        assert np.max(filtered) < 500.0
        assert np.min(filtered) > -100.0

    def test_cache_key_generation(self, qapp):
        viewer = VMapViewer()
        viewer.set_map_data(_simple_2dmap_df(), "m1")
        key = viewer._build_cache_key()
        assert key[0] == "m1"
        assert key[1] == "Intensity"
        assert key[3] == "2Dmap"

    def test_stats_box_computation(self, qapp):
        import numpy as np
        viewer = VMapViewer()
        viewer.cbb_map_type.setCurrentText("Wafer_200mm")
        viewer._last_final_z_col = np.array([10.0, 20.0, 30.0, np.nan])
        viewer._draw_stats_box()
        assert viewer._last_stats_text_artist is not None
        text = viewer._last_stats_text_artist.get_text()
        assert "mean: 20.00" in text
        assert "min: 10.00" in text
        assert "max: 30.00" in text

    def test_profile_extraction_between_two_points(self, qapp):
        viewer = VMapViewer()
        viewer.cbb_map_type.setCurrentText("2Dmap")
        viewer.set_map_data(_simple_2dmap_df(), "m1")
        viewer.set_selected_points([(0.0, 0.0), (1.0, 1.0)])
        profile = viewer._extract_profile()
        assert profile is not None
        assert len(profile) == 100
        assert "distance" in profile.columns
        assert "values" in profile.columns


class TestMapViewerNavigationToolbar:
    """Tests for the redesigned navigation buttons on VMapViewer."""

    def test_navigation_buttons_initialization(self, qapp):
        viewer = VMapViewer()
        assert hasattr(viewer, "btn_home")
        assert hasattr(viewer, "btn_pan")
        assert hasattr(viewer, "btn_zoom")
        assert hasattr(viewer, "btn_rescale")

        assert viewer.btn_pan.isCheckable()
        assert viewer.btn_zoom.isCheckable()
        assert not viewer.btn_home.isCheckable()

        assert not viewer.btn_home.icon().isNull()
        assert not viewer.btn_pan.icon().isNull()
        assert not viewer.btn_zoom.icon().isNull()

        assert viewer.btn_home.width() == 28 and viewer.btn_home.height() == 28
        assert viewer.btn_pan.width() == 28 and viewer.btn_pan.height() == 28
        assert viewer.btn_zoom.width() == 28 and viewer.btn_zoom.height() == 28

    def test_pan_zoom_toggle_and_sync(self, qapp):
        viewer = VMapViewer()

        # Initial state
        assert not viewer.btn_pan.isChecked()
        assert not viewer.btn_zoom.isChecked()
        assert viewer.toolbar.mode == ""

        # Toggle pan
        viewer.btn_pan.click()
        assert viewer.btn_pan.isChecked()
        assert not viewer.btn_zoom.isChecked()
        assert viewer.toolbar.mode == "pan/zoom"

        # Toggle zoom while pan active -> zoom should activate, pan deactivate
        viewer.btn_zoom.click()
        assert not viewer.btn_pan.isChecked()
        assert viewer.btn_zoom.isChecked()
        assert viewer.toolbar.mode == "zoom rect"

        # Toggle zoom off
        viewer.btn_zoom.click()
        assert not viewer.btn_pan.isChecked()
        assert not viewer.btn_zoom.isChecked()
        assert viewer.toolbar.mode == ""

    def test_apply_global_theme_updates_icons(self, qapp):
        viewer = VMapViewer()
        viewer.apply_global_theme("dark")
        assert not viewer.btn_home.icon().isNull()
        assert not viewer.btn_pan.icon().isNull()
        assert not viewer.btn_zoom.icon().isNull()

        viewer.apply_global_theme("light")
        assert not viewer.btn_home.icon().isNull()
        assert not viewer.btn_pan.icon().isNull()
        assert not viewer.btn_zoom.icon().isNull()

    def test_built_in_live_coordinates_displayed_and_toolbar_hidden(self, qapp):
        viewer = VMapViewer()
        viewer.show()

        # Matplotlib toolbar must be hidden (background interaction helper)
        assert viewer.toolbar.isHidden()

        # Built-in live coordinate display must be visible and have stable minimum width
        assert not viewer.lbl_coordinates.isHidden()
        assert viewer.lbl_coordinates.minimumWidth() >= 180

        # locLabel must exist and update on set_message
        assert hasattr(viewer.toolbar, "locLabel")
        viewer.toolbar.set_message("x=12.5  y=34.2")
        assert "12.5" in viewer.toolbar.locLabel.text()
        assert "12.5" in viewer.lbl_coordinates.text()

        # Leaving figure canvas clears live coordinates
        import matplotlib.backend_bases as mbb
        leave_event = mbb.LocationEvent("figure_leave_event", viewer.canvas, 0, 0)
        viewer._on_figure_leave(leave_event)
        assert viewer.lbl_coordinates.text() == ""

    def test_map_viewer_copy_feedback(self, qapp):
        viewer = VMapViewer()
        assert not viewer.btn_copy.icon().isNull()
        viewer._copy_figure_to_clipboard()
        viewer._reset_copy_feedback()
        assert not viewer.btn_copy.icon().isNull()


class TestMapViewerCoordinateDrift:
    """Regression tests verifying zero coordinate drift between data points, heatmap pixels, and selection overlay."""

    def test_irregular_2dmap_has_zero_coordinate_drift(self, qapp):
        """Wafer or irregular coordinates in 2Dmap mode must render color pixels at true (X, Y) without drift."""
        from matplotlib.collections import PatchCollection
        import numpy as np
        viewer = VMapViewer()
        viewer.cbb_map_type.setCurrentText("2Dmap")

        # Irregular circular wafer points (e.g. wafer1_process1 geometry)
        df = pd.DataFrame({
            'X': [0.0, 29.7, 21.0, 0.0, -21.0, -29.7, -21.0, 0.0, 21.0],
            'Y': [0.0, 0.0, 20.978, 29.667, 20.978, 0.0, -20.977, -29.667, -20.977],
            100.0: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        })
        viewer.set_map_data(df, "irregular_wafer")
        viewer.set_selected_points([(-21.0, -20.977)])
        viewer._do_plot_heatmap()

        # Image must use PatchCollection to place rectangles at exact coordinates
        assert isinstance(viewer.img, PatchCollection)

        # Check selection rectangle center
        sel_rect = viewer._selection_collection.get_paths()[0]
        sel_bbox = sel_rect.get_extents()
        sel_cx = (sel_bbox.x0 + sel_bbox.x1) / 2
        sel_cy = (sel_bbox.y0 + sel_bbox.y1) / 2

        # Check patch center in PatchCollection
        paths = viewer.img.get_paths()
        matching_patch = None
        for p in paths:
            bb = p.get_extents()
            cx = (bb.x0 + bb.x1) / 2
            cy = (bb.y0 + bb.y1) / 2
            if np.isclose(cx, -21.0, atol=1e-3) and np.isclose(cy, -20.977, atol=1e-3):
                matching_patch = (cx, cy)
                break

        assert matching_patch is not None, "Color pixel patch at (-21.0, -20.977) must exist"
        drift = np.sqrt((matching_patch[0] - sel_cx) ** 2 + (matching_patch[1] - sel_cy) ** 2)
        assert drift < 1e-4, f"Expected 0 drift, got {drift}"

    def test_regular_2dmap_retains_axes_image_rendering(self, qapp):
        """Regular uniform Cartesian grids must continue to use fast AxesImage rendering."""
        from matplotlib.image import AxesImage
        viewer = VMapViewer()
        viewer.cbb_map_type.setCurrentText("2Dmap")
        viewer.set_map_data(_simple_2dmap_df(), "regular_map")
        viewer._do_plot_heatmap()

        assert isinstance(viewer.img, AxesImage)
        extent = viewer.img.get_extent()
        # 2x2 grid from 0 to 1 with dx=1 -> extent is [-0.5, 1.5, -0.5, 1.5]
        assert extent == [-0.5, 1.5, -0.5, 1.5]

    def test_wafer_map_extent_symmetric_and_centered(self, qapp):
        """Wafer map interpolation extent must be symmetric and isotropic around origin."""
        viewer = VMapViewer()
        viewer.cbb_map_type.setCurrentText("Wafer_200mm")
        viewer.set_map_data(_simple_2dmap_df(), "wafer_test")
        viewer._do_plot_heatmap()

        extent = viewer.img.get_extent()
        # Radius is 100mm, grid size is 80 -> half_step = 100/79 ≈ 1.2658mm
        # Extent must be exactly symmetric: -extent[0] == extent[1] and -extent[2] == extent[3]
        assert abs(extent[0] + extent[1]) < 1e-6
        assert abs(extent[2] + extent[3]) < 1e-6
        assert abs(extent[0] - extent[2]) < 1e-6



