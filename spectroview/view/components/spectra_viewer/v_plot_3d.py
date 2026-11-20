"""Interactive 3D spectrum visualization and projection management for SpectraViewer."""

from __future__ import annotations

import warnings

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from spectroview import PLOT_POLICY_DARK, PLOT_POLICY_LIGHT
from spectroview.fit_engine.evaluator import eval_peak_initial
from spectroview.i18n import combo_data_or_text, tr
from spectroview.view.components.spectra_viewer.common import (
    get_cached_style_dict,
    non_singular_bounds,
)


class FastLine3DCollection(Line3DCollection):
    """Vectorized Line3DCollection projecting all 3D segments in a single pass."""

    def __init__(self, segments, *args, **kwargs):
        self._flat_3d = None
        self._seg_slices = None
        super().__init__(segments, *args, **kwargs)

    def set_segments(self, segments):
        if segments is None or len(segments) == 0:
            self._segments3d = []
            self._flat_3d = None
            self._seg_slices = None
            super().set_segments([])
            return

        if isinstance(segments, np.ndarray) and segments.ndim == 3:
            self._segments3d = np.ascontiguousarray(segments, dtype=float)
            n_segs, n_pts, _ = self._segments3d.shape
            self._flat_3d = self._segments3d.reshape(-1, 3)
            self._seg_slices = None
            self._paths = [
                mpath.Path(np.empty((n_pts, 2), dtype=float))
                for _ in range(n_segs)
            ]
            self.stale = True
        else:
            segs_list = [
                np.ascontiguousarray(s, dtype=float)
                for s in segments
                if len(s) > 0
            ]
            self._segments3d = segs_list
            if segs_list:
                lengths = [len(s) for s in segs_list]
                self._flat_3d = np.vstack(segs_list)
                splits = np.cumsum(lengths)
                starts = np.concatenate(([0], splits[:-1]))
                self._seg_slices = list(zip(starts, splits))
                self._paths = [
                    mpath.Path(np.empty((length, 2), dtype=float))
                    for length in lengths
                ]
            else:
                self._flat_3d = None
                self._seg_slices = None
                self._paths = []
            self.stale = True

    def do_3d_projection(self):
        flat = getattr(self, "_flat_3d", None)
        if not getattr(self, "_axlim_clip", False) and flat is not None and flat.size > 0:
            xs, ys, zs = proj3d._proj_trans_points(flat, self.axes.M)
            pts_2d = np.column_stack([xs, ys])
            if self._seg_slices is None:
                n_segs, n_pts, _ = self._segments3d.shape
                segs_2d = pts_2d.reshape(n_segs, n_pts, 2)
                for path, seg_2d in zip(self._paths, segs_2d):
                    path._vertices = seg_2d
            else:
                for path, (start, end) in zip(self._paths, self._seg_slices):
                    path._vertices = pts_2d[start:end]
            self.stale = True
            finite_zs = zs[np.isfinite(zs)]
            return float(np.min(finite_zs)) if finite_zs.size else 1e9
        return super().do_3d_projection()


class SpectraPlot3DMixin:
    """2D <-> 3D Axes3D projection management and vectorized 3D plot rendering."""

    def _apply_2d_axes_style(self, style_dict: dict):
        """Apply light/dark style colors to 2D rectilinear spines, ticks, and labels."""
        self.ax.set_facecolor(style_dict.get("axes.facecolor", "white"))
        edge_color = style_dict.get("axes.edgecolor", "black")
        for spine in self.ax.spines.values():
            spine.set_color(edge_color)

        tick_color = style_dict.get("xtick.color", "black")
        self.ax.tick_params(colors=tick_color, which="both")

        label_color = style_dict.get("axes.labelcolor", "black")
        self.ax.xaxis.label.set_color(label_color)
        self.ax.yaxis.label.set_color(label_color)
        self.ax.title.set_color(label_color)

    def _apply_3d_axes_style(self):
        """Apply light/dark style colors to Axes3D panes, axis lines, ticks, and labels."""
        if not hasattr(self.ax, "zaxis"):
            return
        style_name = (
            combo_data_or_text(self.cbb_theme)
            if hasattr(self, "cbb_theme")
            else "Light Mode"
        )
        style_path = (
            PLOT_POLICY_DARK if style_name == "Dark Mode" else PLOT_POLICY_LIGHT
        )
        style_dict = get_cached_style_dict(style_path)

        axes_bg = style_dict.get("axes.facecolor", "white")
        edge_color = style_dict.get("axes.edgecolor", "black")
        tick_color = style_dict.get("xtick.color", "black")
        label_color = style_dict.get("axes.labelcolor", "black")

        self.ax.set_facecolor(axes_bg)
        if style_name == "Dark Mode":
            pane_colors = (
                (0.18, 0.18, 0.20, 0.55),
                (0.16, 0.16, 0.18, 0.55),
                (0.14, 0.14, 0.16, 0.55),
            )
            grid_color = (1.0, 1.0, 1.0, 0.18)
        else:
            pane_colors = (
                (0.95, 0.95, 0.95, 0.50),
                (0.90, 0.90, 0.90, 0.50),
                (0.925, 0.925, 0.925, 0.50),
            )
            grid_color = (0.0, 0.0, 0.0, 0.15)

        for axis, pane_col in zip(
            (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis), pane_colors
        ):
            axis.set_pane_color(pane_col)
            axis.pane.set_edgecolor(edge_color)
            axis.line.set_color(edge_color)
            axis._axinfo["grid"]["color"] = grid_color
            axis.label.set_color(label_color)

        for dim in ("x", "y", "z"):
            self.ax.tick_params(axis=dim, colors=tick_color, which="both")
        self.ax.title.set_color(label_color)

    def _ensure_axes_projection(self, is_3d: bool):
        """Switch figure axes between 2D rectilinear and 3D projection when needed."""
        currently_3d = getattr(self.ax, "name", None) == "3d"
        if currently_3d == bool(is_3d):
            return

        self._remove_heatmap_colorbar()
        if hasattr(self, "range_manager"):
            self.range_manager.clear_artists()
        self._fitted_lines.clear()
        self._legend_obj = None
        self._legend_bbox = None
        self._highlighted_line = None

        old_ax = self.ax
        if currently_3d:
            # Disconnect Axes3D mouse callbacks before deleting to avoid leaks
            for _sig, cb_dict in list(self.canvas.callbacks.callbacks.items()):
                for cid, ref in list(cb_dict.items()):
                    func = ref() if callable(ref) else None
                    if getattr(func, "__self__", None) is old_ax:
                        self.canvas.mpl_disconnect(cid)
            if hasattr(old_ax, "disable_mouse_rotation"):
                old_ax.disable_mouse_rotation()

        if old_ax in self.figure.axes:
            self.figure.delaxes(old_ax)

        style_name = (
            combo_data_or_text(self.cbb_theme)
            if hasattr(self, "cbb_theme")
            else "Light Mode"
        )
        style_path = (
            PLOT_POLICY_DARK if style_name == "Dark Mode" else PLOT_POLICY_LIGHT
        )
        with plt.style.context(style_path):
            if is_3d:
                self.ax = self.figure.add_subplot(
                    111,
                    projection="3d",
                    elev=self._default_3d_elev,
                    azim=self._default_3d_azim,
                )
                self._apply_3d_axes_style()
                self.ax.set_navigate_mode(None)
            else:
                self.ax = self.figure.add_subplot(111)
                self._apply_2d_axes_style(get_cached_style_dict(style_path))

    @staticmethod
    def _non_singular_bounds(
        vmin: float, vmax: float, pad: float = 0.5
    ) -> tuple[float, float]:
        """Ensure axis bounds are finite and non-singular."""
        return non_singular_bounds(vmin, vmax, pad=pad)

    def _draw_3d_bestfit(self, y_values: np.ndarray, lw: float, fg_color):
        """Render bestfit curves and peak components in 3D space."""
        n_specs = len(self._tensor_data.get("fnames", []))
        is_tensor_list = self._tensor_data.get("type") == "tensor_list"
        x_val = self._tensor_data.get("x")
        y_peaks_list = self._tensor_data.get("y_peaks", [])
        colorful = (
            not hasattr(self, "act_bestfit_colorful")
            or self.act_bestfit_colorful.isChecked()
        )

        peak_segs, peak_cols = [], []
        bestfit_segs = []

        for spec_idx in range(min(n_specs, self.spin_max_legend_items.value())):
            x = np.asarray(
                x_val[spec_idx] if isinstance(x_val, list) else x_val, dtype=float
            )
            if len(x) == 0:
                continue
            y_pos = float(y_values[spec_idx])
            y_base = self._get_tensor_item("y_baseline", spec_idx, is_tensor_list)
            fit_model = self._get_tensor_item("fit_models", spec_idx, is_tensor_list)

            if fit_model and fit_model.get("peak_models") and len(x) > 1:
                n_fine_pts = max(1000, len(x) * 3)
                x_fine = np.linspace(x.min(), x.max(), n_fine_pts)
                sorted_keys = sorted(
                    fit_model["peak_models"].keys(), key=lambda k: int(k)
                )
                y_fit_fine = np.zeros_like(x_fine)
                for i, k in enumerate(sorted_keys):
                    y_peak_fine = eval_peak_initial(x_fine, fit_model["peak_models"][k])
                    y_fit_fine += y_peak_fine
                    peak_segs.append(
                        np.column_stack([
                            x_fine,
                            np.full(n_fine_pts, y_pos, dtype=float),
                            y_peak_fine,
                        ])
                    )
                    peak_cols.append(f"C{(i + 1) % 10}" if colorful else fg_color)

                if y_base is not None:
                    y_fit_fine += np.interp(x_fine, x, y_base)
                bestfit_segs.append(
                    np.column_stack([
                        x_fine,
                        np.full(n_fine_pts, y_pos, dtype=float),
                        y_fit_fine,
                    ])
                )
                continue

            if y_peaks_list:
                y_peaks = (
                    y_peaks_list[spec_idx]
                    if is_tensor_list and spec_idx < len(y_peaks_list)
                    else (
                        [p[spec_idx] for p in y_peaks_list]
                        if not is_tensor_list
                        else None
                    )
                )
                if y_peaks is not None:
                    for i, y_peak in enumerate(y_peaks):
                        peak_segs.append(
                            np.column_stack([
                                x,
                                np.full(len(x), y_pos, dtype=float),
                                np.asarray(y_peak, dtype=float),
                            ])
                        )
                        peak_cols.append(f"C{(i + 1) % 10}" if colorful else fg_color)

            y_bestfit = self._get_tensor_item("y_bestfit", spec_idx, is_tensor_list)
            if y_bestfit is not None:
                bestfit_segs.append(
                    np.column_stack([
                        x,
                        np.full(len(x), y_pos, dtype=float),
                        np.asarray(y_bestfit, dtype=float),
                    ])
                )

        if peak_segs:
            self.ax.add_collection3d(
                FastLine3DCollection(
                    peak_segs, colors=peak_cols, linewidths=(lw * 0.6)
                ),
                autolim=False,
            )
        if bestfit_segs:
            self.ax.add_collection3d(
                FastLine3DCollection(
                    bestfit_segs,
                    colors=[fg_color] * len(bestfit_segs),
                    linewidths=(lw * 0.6),
                ),
                autolim=False,
            )

    def _plot_3d(self):
        """Render the selected spectra in an interactive 3D plot."""
        if getattr(self, "_reset_3d_view", False) or not hasattr(self.ax, "elev"):
            elev = self._default_3d_elev
            azim = self._default_3d_azim
            roll = 0.0
            self._reset_3d_view = False
        else:
            elev = self.ax.elev
            azim = self.ax.azim
            roll = getattr(self.ax, "roll", 0.0)

        fg_color = plt.rcParams.get("axes.labelcolor", "black")
        self.ax.clear()
        self._remove_heatmap_colorbar()
        self._fitted_lines.clear()
        self._apply_3d_axes_style()
        self.ax.view_init(elev=elev, azim=azim, roll=roll)
        self.ax.set_navigate_mode(None)
        self._update_r2_display()

        Y_data = self._tensor_data.get("y", [])
        x_data = self._tensor_data.get("x")
        N = len(Y_data)
        if N == 0 or x_data is None:
            self.canvas.draw_idle()
            return

        Z_norm = self._get_normalized_y_tensor(x_data, Y_data)
        y_values, y_label = self._get_heatmap_y_values()
        if len(y_values) != N:
            y_values = np.arange(N, dtype=float)
        else:
            y_values = np.where(
                np.isfinite(y_values), y_values, np.arange(N, dtype=float)
            )

        plot_style = combo_data_or_text(self.cbb_plotstyle)
        lw = self.spin_lw.value()
        dot_size = self.spin_dotsize.value()
        main_colors = self._resolve_series_colors(N)

        is_list = isinstance(Z_norm, list) or isinstance(x_data, list)
        x_mins, x_maxs, z_mins, z_maxs, pos_z_mins = [], [], [], [], []

        if plot_style == "dot":
            all_x, all_y, all_z, all_c = [], [], [], []
            if not is_list:
                x_arr = np.asarray(x_data, dtype=float)
                Z_arr = np.asarray(Z_norm, dtype=float)
                if Z_arr.ndim == 1:
                    Z_arr = Z_arr[None, :]
                if x_arr.size > 0 and Z_arr.size > 0:
                    X_3d = np.broadcast_to(x_arr[None, :], Z_arr.shape)
                    Y_3d = np.broadcast_to(y_values[:, None], Z_arr.shape)
                    all_x.append(X_3d.ravel())
                    all_y.append(Y_3d.ravel())
                    all_z.append(Z_arr.ravel())
                    for i in range(N):
                        all_c.extend([main_colors[i]] * len(x_arr))
                    x_mins.append(float(np.nanmin(x_arr)))
                    x_maxs.append(float(np.nanmax(x_arr)))
                    z_mins.append(float(np.nanmin(Z_arr)))
                    z_maxs.append(float(np.nanmax(Z_arr)))
                    pos_z = Z_arr[Z_arr > 0]
                    if pos_z.size:
                        pos_z_mins.append(float(np.min(pos_z)))
            else:
                for i in range(N):
                    xi = np.asarray(
                        x_data[i] if isinstance(x_data, list) else x_data,
                        dtype=float,
                    )
                    zi = np.asarray(Z_norm[i], dtype=float)
                    if xi.size == 0 or zi.size == 0:
                        continue
                    all_x.append(xi)
                    all_y.append(np.full(len(xi), y_values[i], dtype=float))
                    all_z.append(zi)
                    all_c.extend([main_colors[i]] * len(xi))
                    x_mins.append(float(np.nanmin(xi)))
                    x_maxs.append(float(np.nanmax(xi)))
                    z_mins.append(float(np.nanmin(zi)))
                    z_maxs.append(float(np.nanmax(zi)))
                    pos_z = zi[zi > 0]
                    if pos_z.size:
                        pos_z_mins.append(float(np.min(pos_z)))
            if all_x:
                self.ax.scatter(
                    np.concatenate(all_x),
                    np.concatenate(all_y),
                    np.concatenate(all_z),
                    s=dot_size,
                    c=all_c,
                    marker="o",
                    edgecolors="none",
                    depthshade=False,
                )
        else:
            if not is_list:
                x_arr = np.asarray(x_data, dtype=float)
                Z_arr = np.asarray(Z_norm, dtype=float)
                if Z_arr.ndim == 1:
                    Z_arr = Z_arr[None, :]
                if x_arr.size > 0 and Z_arr.size > 0:
                    X_3d = np.broadcast_to(x_arr[None, :], Z_arr.shape)
                    Y_3d = np.broadcast_to(y_values[:, None], Z_arr.shape)
                    segs_3d = np.stack([X_3d, Y_3d, Z_arr], axis=-1)
                    lc = FastLine3DCollection(
                        segs_3d, colors=main_colors, linewidths=lw
                    )
                    self.ax.add_collection3d(lc, autolim=False)
                    x_mins.append(float(np.nanmin(x_arr)))
                    x_maxs.append(float(np.nanmax(x_arr)))
                    z_mins.append(float(np.nanmin(Z_arr)))
                    z_maxs.append(float(np.nanmax(Z_arr)))
                    pos_z = Z_arr[Z_arr > 0]
                    if pos_z.size:
                        pos_z_mins.append(float(np.min(pos_z)))
            else:
                segs_3d = []
                seg_colors = []
                for i in range(N):
                    xi = np.asarray(
                        x_data[i] if isinstance(x_data, list) else x_data,
                        dtype=float,
                    )
                    zi = np.asarray(Z_norm[i], dtype=float)
                    if xi.size == 0 or zi.size == 0:
                        continue
                    segs_3d.append(
                        np.column_stack([
                            xi,
                            np.full(len(xi), y_values[i], dtype=float),
                            zi,
                        ])
                    )
                    seg_colors.append(main_colors[i])
                    x_mins.append(float(np.nanmin(xi)))
                    x_maxs.append(float(np.nanmax(xi)))
                    z_mins.append(float(np.nanmin(zi)))
                    z_maxs.append(float(np.nanmax(zi)))
                    pos_z = zi[zi > 0]
                    if pos_z.size:
                        pos_z_mins.append(float(np.min(pos_z)))
                if segs_3d:
                    lc = FastLine3DCollection(
                        segs_3d, colors=seg_colors, linewidths=lw
                    )
                    self.ax.add_collection3d(lc, autolim=False)

        # ── Raw overlay ──
        if self.act_raw.isChecked():
            x0 = self._tensor_data.get("x0")
            Y0 = self._tensor_data.get("y0")
            if x0 is not None and Y0 is not None:
                if not isinstance(Y0, list) and not isinstance(x0, list):
                    x0_arr = np.asarray(x0, dtype=float)
                    Y0_arr = np.asarray(Y0, dtype=float)
                    if Y0_arr.ndim == 1:
                        Y0_arr = Y0_arr[None, :]
                    if x0_arr.size > 0 and Y0_arr.size > 0:
                        raw_segs = np.stack([
                            np.broadcast_to(x0_arr[None, :], Y0_arr.shape),
                            np.broadcast_to(y_values[:, None], Y0_arr.shape),
                            Y0_arr,
                        ], axis=-1)
                        self.ax.add_collection3d(
                            FastLine3DCollection(
                                raw_segs,
                                colors=[fg_color] * len(Y0_arr),
                                linewidths=0.8,
                                alpha=0.8,
                            ),
                            autolim=False,
                        )
                        z_mins.append(float(np.nanmin(Y0_arr)))
                        z_maxs.append(float(np.nanmax(Y0_arr)))
                else:
                    raw_segs = []
                    for i in range(N):
                        xi = np.asarray(
                            x0[i] if isinstance(x0, list) else x0, dtype=float
                        )
                        yi = np.asarray(Y0[i], dtype=float)
                        if xi.size > 0 and yi.size > 0:
                            raw_segs.append(
                                np.column_stack([
                                    xi,
                                    np.full(len(xi), y_values[i], dtype=float),
                                    yi,
                                ])
                            )
                            z_mins.append(float(np.nanmin(yi)))
                            z_maxs.append(float(np.nanmax(yi)))
                    if raw_segs:
                        self.ax.add_collection3d(
                            FastLine3DCollection(
                                raw_segs,
                                colors=[fg_color] * len(raw_segs),
                                linewidths=0.8,
                                alpha=0.8,
                            ),
                            autolim=False,
                        )

        # ── Bestfit overlay (un-normalized view) ──
        if self.btn_bestfit.isChecked() and not self.btn_norm.isChecked():
            self._draw_3d_bestfit(y_values, lw, fg_color)

        # ── Axes limits & scale ──
        x_lo, x_hi = self._non_singular_bounds(
            min(x_mins) if x_mins else 0.0,
            max(x_maxs) if x_maxs else 1.0,
        )
        y_lo, y_hi = self._non_singular_bounds(
            float(np.nanmin(y_values)),
            float(np.nanmax(y_values)),
        )
        z_lo, z_hi = self._non_singular_bounds(
            min(z_mins) if z_mins else 0.0,
            max(z_maxs) if z_maxs else 1.0,
        )
        self.ax.set_xlim(x_lo, x_hi)
        self.ax.set_ylim(y_lo, y_hi)

        if combo_data_or_text(self.cbb_yscale) == "Log" and z_hi > 0:
            z_pos_lo = min(pos_z_mins) if pos_z_mins else max(z_hi * 1e-4, 1e-9)
            if z_pos_lo >= z_hi:
                z_pos_lo = z_hi * 0.1
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Attempt to set non-positive .*lim on a log-scaled axis.*",
                    category=UserWarning,
                )
                self.ax.set_zlim(z_pos_lo, z_hi)
                self.ax.set_zscale("log")
                self.ax.set_zlim(z_pos_lo, z_hi)
        else:
            self.ax.set_zscale("linear")
            self.ax.set_zlim(z_lo, z_hi)

        # ── Legend / Colorbar ──
        self._legend_obj = None
        self._legend_bbox = None
        if self.btn_legend.isChecked():
            if self._is_legend_colorbar_active():
                param_values, cbar_label = self._get_colorbar_values()
                cmap, norm, _ = self._resolve_colorbar_cmap_and_norm(N, param_values)
                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = self.figure.colorbar(sm, ax=self.ax, pad=0.1, shrink=0.75)
                cbar.set_label(cbar_label, color=fg_color)
                cbar.ax.tick_params(colors=fg_color)
                self._heatmap_colorbar = cbar
            else:
                t_labels = self._tensor_data.get("labels", [])
                t_fnames = self._tensor_data.get("fnames", [])
                proxies = self._tensor_data.get("proxies", [])
                max_items = self.spin_max_legend_items.value()
                handles = []
                labels = []
                for spec_idx in range(min(N, max_items)):
                    label_str = (
                        t_labels[spec_idx]
                        if spec_idx < len(t_labels) and t_labels[spec_idx]
                        else (
                            t_fnames[spec_idx]
                            if spec_idx < len(t_fnames)
                            else tr("Spectrum {index}", index=spec_idx + 1)
                        )
                    )
                    proxy_line = mlines.Line2D(
                        [], [], color=main_colors[spec_idx], label=label_str, lw=lw
                    )
                    if spec_idx < len(proxies):
                        proxy_line._spectrum_ref = proxies[spec_idx]
                    handles.append(proxy_line)
                    labels.append(label_str)
                if handles:
                    if (
                        getattr(self, "act_legend_outside", None)
                        and self.act_legend_outside.isChecked()
                    ):
                        legend = self.ax.legend(
                            handles,
                            labels,
                            loc="center left",
                            bbox_to_anchor=(1.02, 0.5),
                        )
                    else:
                        legend = self.ax.legend(handles, labels, loc="upper right")
                    self._make_legend_pickable(legend)

        self.ax.grid(self.act_grid.isChecked())
        self.ax.set_xlabel(self.cbb_xaxis.currentText())
        self.ax.set_ylabel(y_label)
        self.ax.set_zlabel(self.cbb_yaxis.currentText())
        self.canvas.draw_idle()

    def _toggle_3d_mode(self, checked: bool):
        """Activate or deactivate interactive 3D spectra visualization mode."""
        self._3d_mode = bool(checked)
        if checked:
            self._reset_3d_view = True
            if getattr(self, "_heatmap_mode", False) or (
                hasattr(self, "btn_heatmap") and self.btn_heatmap.isChecked()
            ):
                self._heatmap_mode = False
                if hasattr(self, "btn_heatmap"):
                    self.btn_heatmap.blockSignals(True)
                    self.btn_heatmap.setChecked(False)
                    self.btn_heatmap.blockSignals(False)

        self._sync_mode_toolbar_state()
        self._emit_view_options()
