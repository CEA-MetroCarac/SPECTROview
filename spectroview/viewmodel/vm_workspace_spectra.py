# ─── spectroview/viewmodel/vm_workspace_spectra.py ───
from copy import deepcopy
import numpy as np

from PySide6.QtCore import QObject, Signal
from pathlib import Path

from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_settings import MSettings

from spectroview.model.m_io import load_spectrum_file

from spectroview.viewmodel.utils import baseline_to_dict, dict_to_baseline


class VMWorkspaceSpectra(QObject):
    # ───── ViewModel → View signals ─────
    spectra_list_changed = Signal(list)      # list[str]
    spectra_selection_changed = Signal(list) # list[dict] → plot data
    count_changed = Signal(int)
    show_xcorrection_value = Signal(float)  # ΔX of first selected spectrum
    spectral_range_changed = Signal(float, float)

    notify = Signal(str)  # general notifications
    
    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings
        self.spectra = MSpectra()
        self.selected_indices = []
        self._baseline_clipboard = None  # for copy/paste baseline
        self._peaks_clipboard = None    # for copy/paste peaks


    # View → ViewModel slots
    def load_files(self, paths: list[str]):
        existing_paths = {s.source_path for s in self.spectra}

        skipped = []

        for p in paths:
            path = str(Path(p).resolve())
            if path in existing_paths:
                skipped.append(Path(p).name)
                continue

            spectrum = load_spectrum_file(Path(p))
            self.spectra.add(spectrum)

        if skipped:
            self.notify.emit(
                f"Already loaded and skipped:\n" + "\n".join(skipped)
            )
        self._emit_list_update()


    def set_selected_indices(self, indices: list[int]):
        """Set currently selected spectra (via Listwidget) by their indices."""
        self.selected_indices = indices
        self._emit_selected_spectra()

    def _emit_selected_spectra(self):
        """Prepare and emit data for plotting the selected spectra."""
        selected_spectra = self.spectra.get(self.selected_indices)

        if not selected_spectra:
            self.spectra_selection_changed.emit([])
            return
        # emit list of the selected spectra to plot in View
        self.spectra_selection_changed.emit(selected_spectra)    

        # emit x-correction of first spectrum to show in GUI
        self.show_xcorrection_value.emit(selected_spectra[0].xcorrection_value)

        # emit spectral range of first selected spectrum to show in GUI
        s = selected_spectra[0]
        xmin = float(s.x[0])
        xmax = float(s.x[-1])
        self.spectral_range_changed.emit(xmin, xmax)
    
    def reorder_spectra(self, new_order: list[int]):
        """new_order = list of old indices in new visual order"""
        self.spectra.reorder(new_order)

        # After reorder, selection must be re-emitted
        self._emit_list_update()
        self._emit_selected_spectra()

    
    
            
    def remove_selected_spectra(self):
        """Remove currently selected spectra."""
        if not self.selected_indices:
            self.notify.emit("No spectra selected.")
            return
        old_selection = set(self.selected_indices)
        old_count = len(self.spectra)
        # Remove from model
        self.spectra.remove(self.selected_indices)
        
        new_count = len(self.spectra)
        self._emit_list_update()

        if new_count == 0:
            self.selected_indices = []
            self.spectra_selection_changed.emit([])
            return
        # Find closest valid index
        min_removed = min(old_selection)
        new_index = min(min_removed, new_count - 1)

        self.selected_indices = [new_index]
        self._emit_selected_spectra()
        
    # Internal helpers
    def _emit_list_update(self):
        """Emit updated list of spectra names and count."""
        names = [s.fname for s in self.spectra]
        self.spectra_list_changed.emit(names)
        self.count_changed.emit(len(self.spectra))

    def add_peak_at(self, x: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]
        
        fit_settings = self.settings.load_fit_settings()

        maxshift = fit_settings.get("maxshift", 20.0)
        maxfwhm = fit_settings.get("maxfwhm", 200.0)

        spectrum.add_peak_model(
            spectrum.peak_model if hasattr(spectrum, "peak_model") else "Lorentzian",
            x,
            dx0=(maxshift, maxshift),
            dfwhm=maxfwhm,
        )
        self._emit_selected_spectra()

    def remove_peak_at(self, x: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if not spectrum.peak_models:
            return

        idx = min(
            range(len(spectrum.peak_models)),
            key=lambda i: abs(
                spectrum.peak_models[i].param_hints["x0"]["value"] - x
            )
        )

        del spectrum.peak_models[idx]
        del spectrum.peak_labels[idx]
        self._emit_selected_spectra()

    def set_baseline_settings(self, settings: dict):
        if not self.selected_indices:
            return

        for spectrum in self.spectra.get(self.selected_indices):
            bl = spectrum.baseline
            bl.attached = settings["attached"]
            bl.sigma = settings["noise"]

            if settings["mode"] == "Linear":
                bl.mode = "Linear"
            else:
                bl.mode = "Polynomial"
                bl.order_max = settings["order"]

        self._emit_selected_spectra()
 

    def add_baseline_point(self, x: float, y: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if spectrum.baseline.is_subtracted:
            self.notify.emit("Baseline already subtracted.")
            return

        spectrum.baseline.add_point(x, y)
        self._emit_selected_spectra()


    def remove_baseline_point(self, x: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if not spectrum.baseline.points:
            return

        xs, ys = spectrum.baseline.points
        if not xs:
            return

        idx = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
        xs.pop(idx)
        ys.pop(idx)

        self._emit_selected_spectra()

    def apply_x_correction(self, measured_peak: float):
        """
        Apply X-axis correction to selected spectra.
        delta_x: user-entered correction value
        """
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectra = self.spectra.get(self.selected_indices)

        SI_REF = 520.7
        delta_x = SI_REF - measured_peak 

        for s in spectra:
            s.apply_xcorrection(delta_x)

        # Trigger plot refresh
        self.show_xcorrection_value.emit(spectra[0].xcorrection_value)
        self._emit_selected_spectra()


    def undo_x_correction(self):
        """Undo X-axis correction for selected spectra."""
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectra = self.spectra.get(self.selected_indices)

        for spectrum in spectra:
            spectrum.undo_xcorrection()

        self.show_xcorrection_value.emit(spectra[0].xcorrection_value)
        self._emit_selected_spectra()

    def reinit_spectra(self, apply_all: bool = False):
        """Reinitialize spectra to original data."""
        if apply_all:
            spectra = self.spectra
        else:
            if not self.selected_indices:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self.spectra.get(self.selected_indices)

        for spectrum in spectra:
            spectrum.reinit()
        
        self._emit_selected_spectra() # Refresh plot 


    def apply_spectral_range(self, xmin: float, xmax: float, apply_all: bool):
        if not self.selected_indices:
            return

        if xmin > xmax:
            xmin, xmax = xmax, xmin

        spectra = (
            self.spectra
            if apply_all
            else self.spectra.get(self.selected_indices)
        )

        for spectrum in spectra:
            spectrum.reinit()

            i_min = self._closest_index(spectrum.x0, xmin)
            i_max = self._closest_index(spectrum.x0, xmax)

            spectrum.x = spectrum.x0[i_min:i_max + 1].copy()
            spectrum.y = spectrum.y0[i_min:i_max + 1].copy()

        self._emit_selected_spectra()

    def copy_baseline(self):
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self.spectra.get(self.selected_indices)[0]
        self._baseline_clipboard = deepcopy(baseline_to_dict(spectrum))

    def paste_baseline(self, apply_all: bool = False):
        if self._baseline_clipboard is None:
            self.notify.emit("No baseline copied.")
            return

        if apply_all:
            spectra = self.spectra
        else:
            if not self.selected_indices:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self.spectra.get(self.selected_indices)

        dict_to_baseline(
            deepcopy(self._baseline_clipboard),
            spectra
        )

        self._emit_selected_spectra()

    def subtract_baseline(self, apply_all: bool = False):
        if apply_all:
            spectra = self.spectra
        else:
            if not self.selected_indices:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self.spectra.get(self.selected_indices)

        for spectrum in spectra:
            if not spectrum.baseline.is_subtracted:
                spectrum.subtract_baseline()

        self._emit_selected_spectra()

    def delete_baseline(self, apply_all: bool = False):
        """Delete baseline (points + subtraction state)."""

        if apply_all:
            spectra = self.spectra
        else:
            if not self.selected_indices:
                self.notify.emit("No spectrum selected.")
                return
            spectra = self.spectra.get(self.selected_indices)

        for spectrum in spectra:
            bl = spectrum.baseline

            # Clear baseline points
            if bl.points:
                xs, ys = bl.points
                xs.clear()
                ys.clear()

            # Reset subtraction state
            bl.is_subtracted = False

        self._emit_selected_spectra()

    def copy_peaks(self):
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if not spectrum.peak_models:
            self.notify.emit("No peaks to copy.")
            return

        # IMPORTANT: save(), not manual extraction
        self._peaks_clipboard = deepcopy(spectrum.save())



    def paste_peaks(self, apply_all: bool = False):
        if not hasattr(self, "_peaks_clipboard") or self._peaks_clipboard is None:
            self.notify.emit("No peaks copied.")
            return

        spectra = (
            self.spectra
            if apply_all
            else self.spectra.get(self.selected_indices)
        )

        for spectrum in spectra:
            spectrum.set_attributes(
                {
                    "peak_labels": self._peaks_clipboard.get("peak_labels", []),
                    "peak_models": deepcopy(
                        self._peaks_clipboard.get("peak_models", {})
                    ),
                }
            )

        self._emit_selected_spectra()


    def delete_peaks(self, apply_all: bool = False):
        spectra = (
            self.spectra
            if apply_all
            else self.spectra.get(self.selected_indices)
        )

        if not spectra:
            self.notify.emit("No spectrum selected.")
            return

        for spectrum in spectra:
            if spectrum.peak_models:
                spectrum.remove_models()

        self._emit_selected_spectra()


    def _closest_index(self, array, value):
        return int(np.abs(array - value).argmin())
    

 