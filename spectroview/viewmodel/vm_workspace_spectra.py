# ─── spectroview/viewmodel/vm_workspace_spectra.py ───
from PySide6.QtCore import QObject, Signal
from pathlib import Path

from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_settings import MSettings

from spectroview.model.m_io import load_spectrum_file

class VMWorkspaceSpectra(QObject):
    # ───── ViewModel → View signals ─────
    spectra_list_changed = Signal(list)      # list[str]
    spectra_selection_changed = Signal(list) # list[dict] → plot data
    count_changed = Signal(int)
    show_xcorrection_value = Signal(float)  # ΔX of first selected spectrum

    notify = Signal(str)  # general notifications
    
    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings
        self.spectra = MSpectra()
        self.selected_indices = []

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

    def _emit_selected_spectra(self):
        """Prepare and emit data for plotting the selected spectra."""
        selected_spectra = self.spectra.get(self.selected_indices)

        if not selected_spectra:
            self.spectra_selection_changed.emit([])
            return
        
        # emit x-correction of first spectrum to show in GUI
        self.show_xcorrection_value.emit(selected_spectra[0].xcorrection_value)

        # emit list of the selected spectra to plot in View
        self.spectra_selection_changed.emit(selected_spectra)


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
