"""Preprocessing, baseline, spectral range, and cosmic-ray mixin for VMWorkspaceSpectra."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from spectroview.fit_engine.baseline import eval_baseline_batch
from spectroview.i18n import tr
from spectroview.viewmodel.common.spectral_utils import closest_index


class SpectraPreprocessingMixin:
    """Preprocessing operations (X/Y corrections, ROI cropping, baseline, cosmic-ray removal)."""

    @staticmethod
    def _reset_mapdata_arrays(md):
        """Reset x/Y to their raw x0/Y0, and clear baseline + peak-curve state.

        Shared by reinit_spectra(), apply_spectral_range(), and
        _apply_fit_model_to_mapdata()'s "clean reset" branch. Each caller
        still owns the fields it deliberately doesn't clear here (fit_model,
        range_min/max, colors/labels) since they either want to preserve
        them or are about to repopulate them right afterward.
        """
        if md.x0 is not None:
            md.x = md.x0.copy()
        if md.Y0 is not None:
            md.Y = md.Y0.copy()
        md.baseline_config = None
        md.Y_baseline = None
        md.peak_params = None
        md.Y_peaks = None
        md.Y_bestfit = None
        md.is_baseline_subtracted = False

    def apply_cosmic_ray_erase(self, md_row: int, modified_y):
        """Update the processed spectrum array with the interactively erased data.

        Called on every completed drag-erase so the model stays in sync with
        the viewer's live preview (but Y0 is NOT touched yet — only Y).

        Args:
            md_row:     The actual row index in md.Y to overwrite.  In the
                        Spectra workspace this is always 0; in the Maps
                        workspace it is the spectrum's position in the map tensor.
            modified_y: 1-D array of corrected intensity values.
        """
        md = self._get_interactive_map_data()
        if md is None:
            return

        y_arr = np.asarray(modified_y, dtype=np.float32)

        # Ensure md.Y exists (initialise from Y0 if needed)
        if md.Y is None:
            if md.Y0 is not None:
                md.Y = md.Y0.copy()
            else:
                return

        if md_row < md.Y.shape[0] and y_arr.shape == md.Y[md_row].shape:
            md.Y[md_row] = y_arr
            md.touch()

    def validate_cosmic_ray_erase(self):
        """Permanently commit cosmic-ray erasures to both Y and Y0.

        Called when the user clicks the validate (✓) button.  After this call
        the erasure survives reinit because Y0 (the raw reference) is also
        updated.
        """
        md = self._get_interactive_map_data()
        if md is None:
            return

        # Persist processed data into raw reference so Reinit keeps the correction
        if md.Y is not None and md.Y0 is not None:
            if md.Y.shape == md.Y0.shape:
                md.Y0 = md.Y.copy()
            else:
                # Shapes differ (e.g. spectral range was cropped) — only update
                # the overlapping region to avoid dimension mismatches.
                m = min(md.Y.shape[1], md.Y0.shape[1])
                md.Y0[:, :m] = md.Y[:, :m]

        self.notify.emit(tr("Cosmic ray erasure validated and saved."))
        self._emit_selected_spectra()

    def add_baseline_point(self, x: float, y: float):
        """Add an anchor point to the current spectrum's baseline and re-evaluate."""
        md = self._get_interactive_map_data()
        if not md:
            return

        is_sub = getattr(md, "is_baseline_subtracted", False)
        is_sub_any = is_sub.any() if isinstance(is_sub, np.ndarray) else bool(is_sub)
        if is_sub_any:
            self.notify.emit(tr("Baseline already subtracted. Please reinit first."))
            return

        if md.baseline_config is None:
            # Use current settings if available, else default
            attached = True
            mode = "Linear"
            if hasattr(self, "_baseline_settings") and self._baseline_settings:
                attached = bool(self._baseline_settings.get("attached", True))
                mode = self._baseline_settings.get("mode", "Linear")
            md.baseline_config = {"points": [[], []], "mode": mode, "attached": attached}

        pts = md.baseline_config.get("points", [[], []])
        if not pts or len(pts) < 2:
            pts = [[], []]
            md.baseline_config["points"] = pts

        pts[0].append(x)
        pts[1].append(y)

        # Sort by x
        sorted_pts = sorted(zip(pts[0], pts[1]))
        md.baseline_config["points"] = (
            [list(t) for t in zip(*sorted_pts)] if sorted_pts else [[], []]
        )

        # Dynamically evaluate the baseline curve
        x_arr = md.x_axis
        y_arr = md.y_matrix
        md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)

        self._emit_selected_spectra()

    def remove_baseline_point(self, x: float):
        """Remove the baseline anchor point closest to *x* and re-evaluate."""
        md = self._get_interactive_map_data()
        if not md or not md.baseline_config:
            return

        pts = md.baseline_config.get("points", [[], []])
        if not pts or not pts[0]:
            return

        xs, ys = pts[0], pts[1]
        idx = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
        xs.pop(idx)
        ys.pop(idx)

        md.baseline_config["points"] = [xs, ys]

        # Dynamically evaluate the baseline curve
        x_arr = md.x_axis
        y_arr = md.y_matrix
        md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)

        self._emit_selected_spectra()

    def apply_x_correction(self, measured_peak: float):
        """Shift X-axis so *measured_peak* aligns with the 520.7 cm^-1 Si reference."""
        if not self.selected_fnames:
            self.notify.emit(tr("No spectrum selected."))
            return

        fnames = self._get_selected_spectra()
        si_ref = 520.7
        delta_x = si_ref - measured_peak

        mds = self._get_unique_map_data(fnames)
        for md in mds:
            prev_delta = md.xcorrection_value
            if prev_delta != 0.0:
                if md.x is not None:
                    md.x = md.x - prev_delta
                md.x0 = md.x0 - prev_delta

            if md.x is not None:
                md.x = md.x + delta_x
            md.x0 = md.x0 + delta_x
            md.xcorrection_value = delta_x

            if "xcorrection_value" in md.map_metadata:
                del md.map_metadata["xcorrection_value"]

        # Trigger plot refresh
        self.show_xcorrection_value.emit(mds[0].xcorrection_value)
        self._emit_selected_spectra()

    def undo_x_correction(self):
        """Undo X-axis correction for selected spectra."""
        if not self.selected_fnames:
            self.notify.emit(tr("No spectrum selected."))
            return

        fnames = self._get_selected_spectra()

        for md in self._get_unique_map_data(fnames):
            delta = md.xcorrection_value
            if delta != 0.0:
                if md.x is not None:
                    md.x = md.x - delta
                md.x0 = md.x0 - delta
                md.xcorrection_value = 0.0

            if "xcorrection_value" in md.map_metadata:
                del md.map_metadata["xcorrection_value"]

        self.show_xcorrection_value.emit(0.0)
        self._emit_selected_spectra()

    def apply_y_normalization(self, norm_factor: float, apply_all: bool = False):
        """Apply intensity normalization for selected spectra."""
        fnames = (
            self._get_active_spectra() if apply_all else self._get_selected_spectra()
        )

        for md in self._get_unique_map_data(fnames):
            # Undo existing normalization if there is one
            if (
                getattr(md, "intensity_norm_factor", 1.0) != 1.0
                and getattr(md, "intensity_norm_factor", 1.0) != 0
            ):
                if md.Y is not None:
                    md.Y = md.Y * md.intensity_norm_factor
                if md.Y0 is not None:
                    md.Y0 = md.Y0 * md.intensity_norm_factor

            # Apply new normalization (divide by factor)
            md.intensity_norm_factor = norm_factor
            if norm_factor != 0:
                if md.Y is not None:
                    md.Y = md.Y / norm_factor
                if md.Y0 is not None:
                    md.Y0 = md.Y0 / norm_factor

        self._emit_selected_spectra()

    def undo_y_normalization(self, apply_all: bool = False):
        """Undo intensity normalization for selected spectra."""
        fnames = (
            self._get_active_spectra() if apply_all else self._get_selected_spectra()
        )

        for md in self._get_unique_map_data(fnames):
            if (
                getattr(md, "intensity_norm_factor", 1.0) != 1.0
                and getattr(md, "intensity_norm_factor", 1.0) != 0
            ):
                if md.Y is not None:
                    md.Y = md.Y * md.intensity_norm_factor
                if md.Y0 is not None:
                    md.Y0 = md.Y0 * md.intensity_norm_factor
                md.intensity_norm_factor = 1.0

        self._emit_selected_spectra()

    def reinit_spectra(self, apply_all: bool = False):
        """Reinitialize spectra to original data."""
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all:
                self.notify.emit(tr("No spectrum selected."))
            return

        for md in mds:
            self._reset_mapdata_arrays(md)
            md.fit_model = None
            md.range_min = None
            md.range_max = None
            md.colors = [None] * len(md.fnames)
            md.labels = [None] * len(md.fnames)

            self._on_map_data_changed(md, "reinit_spectra")

        self._post_bulk_action(apply_all, "reinit_spectra")

    def apply_spectral_range(self, xmin: float, xmax: float, apply_all: bool):
        """Crop spectra to the [xmin, xmax] wavenumber interval."""
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all:
                self.notify.emit(tr("No spectrum selected."))
            return

        if xmin > xmax:
            xmin, xmax = xmax, xmin

        for md in mds:
            # 1) Reinit completely to avoid dimension mismatches if previously cropped with baseline/peaks
            self._reset_mapdata_arrays(md)
            md.fit_model = None

            # 2) Perform crop
            if md.x is None:
                curr_x = md.x0
                curr_y = md.Y0
            else:
                curr_x = md.x
                curr_y = md.Y

            i_min = closest_index(curr_x, xmin)
            i_max = closest_index(curr_x, xmax)
            if i_min > i_max:
                i_min, i_max = i_max, i_min

            md.x = curr_x[i_min : i_max + 1].copy()
            md.Y = curr_y[:, i_min : i_max + 1].copy()
            md.range_min = float(md.x[0])
            md.range_max = float(md.x[-1])
            self._on_map_data_changed(md, "apply_spectral_range")

        self._post_bulk_action(apply_all, "apply_spectral_range")

    def copy_baseline(self) -> bool:
        """Copy baseline configuration of the primary selected spectrum."""
        if not self.selected_fnames:
            self.notify.emit(tr("No spectrum selected."))
            return False

        md = self._get_primary_selected_map_data()
        if not (md and md.baseline_config):
            self.notify.emit(tr("No baseline to copy."))
            return False

        self._baseline_clipboard = deepcopy(md.baseline_config)
        self.clipboard_action_succeeded.emit("baseline_copy")
        return True

    def paste_baseline(self, apply_all: bool = False) -> bool:
        """Paste the copied baseline configuration onto target spectra."""
        if self._baseline_clipboard is None:
            self.notify.emit(tr("No baseline copied."))
            return False

        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all:
                self.notify.emit(tr("No spectrum selected."))
            return False

        baseline_data = deepcopy(self._baseline_clipboard)
        for md in mds:
            md.baseline_config = deepcopy(baseline_data)

            x_arr = md.x_axis
            y_arr = md.y_matrix
            md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)
            self._on_map_data_changed(md, "paste_baseline")

        self._post_bulk_action(apply_all, "paste_baseline")
        self.clipboard_action_succeeded.emit("baseline_paste")
        return True

    def subtract_baseline(self, apply_all: bool = False):
        """Subtract evaluated baseline from target spectra."""
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all:
                self.notify.emit(tr("No spectrum selected."))
            return

        subtracted_any = False
        for md in mds:
            if md and md.baseline_config is not None and md.Y_baseline is not None:
                is_sub = getattr(md, "is_baseline_subtracted", False)

                if isinstance(is_sub, np.ndarray):
                    indices_to_sub = [i for i in range(len(is_sub)) if not is_sub[i]]
                    if indices_to_sub:
                        if md.Y is None:
                            md.Y = md.Y0.copy()
                        if md.x is None:
                            md.x = md.x0.copy()
                        md.Y[indices_to_sub] = (
                            md.Y[indices_to_sub] - md.Y_baseline[indices_to_sub]
                        )
                        md.Y_baseline[indices_to_sub] = 0.0
                        md.is_baseline_subtracted[indices_to_sub] = True
                        subtracted_any = True
                        self._on_map_data_changed(md, "subtract_baseline")
                else:
                    if not is_sub:
                        if md.Y is None:
                            md.Y = md.Y0.copy()
                        if md.x is None:
                            md.x = md.x0.copy()
                        md.Y = md.Y - md.Y_baseline
                        md.Y_baseline = None
                        md.is_baseline_subtracted = True
                        subtracted_any = True
                        self._on_map_data_changed(md, "subtract_baseline")

        if subtracted_any:
            self._post_bulk_action(apply_all, "subtract_baseline")
        else:
            self.notify.emit(tr("Baseline already subtracted for all selected spectra."))

    def delete_baseline(self, apply_all: bool = False):
        """Undo baseline subtraction: reinitialise spectrum data, re-apply crop range."""
        mds = self._get_target_mds(apply_all)
        if not mds:
            if not apply_all:
                self.notify.emit(tr("No spectrum selected."))
            return

        for md in mds:
            if md:
                md.baseline_config = None
                md.Y_baseline = None
                md.is_baseline_subtracted = False

                if md.x is not None:
                    xmin = md.x[0]
                    xmax = md.x[-1]
                    i_min = closest_index(md.x0, xmin)
                    i_max = closest_index(md.x0, xmax)
                    if i_min > i_max:
                        i_min, i_max = i_max, i_min

                    md.Y = md.Y0[:, i_min : i_max + 1].copy()
                else:
                    md.Y = md.Y0.copy()
                self._on_map_data_changed(md, "delete_baseline")

        self._post_bulk_action(apply_all, "delete_baseline")

    # ── Baseline mode / settings helpers ──────────────────────────────────

    def _apply_baseline_settings(self, settings: dict, fnames):
        """Push mode/params from 'settings' dict onto each MapData baseline_config."""
        mode = settings.get("mode")
        coef = float(settings.get("coef", 5.0))
        order = int(settings.get("order_max", 1))
        sigma = int(settings.get("sigma", 4))
        attached = bool(settings.get("attached", True))

        for md in self._get_unique_map_data(fnames):
            if md.baseline_config is None:
                md.baseline_config = {"points": [[], []]}
            bl = md.baseline_config
            bl["mode"] = mode
            bl["coef"] = coef
            bl["order_max"] = order
            bl["sigma"] = sigma
            bl["attached"] = attached

            x_arr = md.x_axis
            y_arr = md.y_matrix
            md.Y_baseline = eval_baseline_batch(x_arr, y_arr, md.baseline_config)

    def set_baseline_settings(self, settings: dict):
        """Store baseline settings and apply to currently selected spectra."""
        self._baseline_settings = settings
        if not self.selected_fnames:
            return
        fnames = self._get_selected_spectra()
        self._apply_baseline_settings(settings, fnames)

    def preview_baseline(self, settings: dict):
        """Apply baseline settings to selected spectra and emit plot refresh."""
        self._baseline_settings = settings
        if not self.selected_fnames:
            return

        fnames = self._get_selected_spectra()
        self._apply_baseline_settings(settings, fnames)

        self._emit_selected_spectra()
