from PySide6.QtCore import QObject, Signal
from pathlib import Path

from PySide6.QtWidgets import QFileDialog


from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_io import load_spectrum_file


class VMSpectra(QObject):
    # ───── ViewModel → View signals ─────
    spectra_list_changed = Signal(list)      # list[str]
    spectra_selection_changed = Signal(list) # list[dict] → plot data
    count_changed = Signal(int)
    show_xcorrection_value = Signal(float)  # ΔX of first selected spectrum

    notify = Signal(str)  # general notifications
    
    def __init__(self):
        super().__init__()
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
        self._emit_selection_plot()
        
    def file_open_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            None,
            "Open spectra",
            "",
            "Data (*.txt *.csv)"
        )
        if paths:
            self.load_files(paths)   
            
            
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
        self._emit_selection_plot()
        
    # Internal helpers
    def _emit_list_update(self):
        names = [s.fname for s in self.spectra]
        self.spectra_list_changed.emit(names)
        self.count_changed.emit(len(self.spectra))

    def _emit_selection_plot(self):
        """Prepare and emit data for plotting the selected spectra."""
        spectra = self.spectra.get(self.selected_indices)

        if not spectra:
            self.spectra_selection_changed.emit([])
            return
        
        # 🔑 emit x-correction of first spectrum to show in GUI
        first = spectra[0]
        self.show_xcorrection_value.emit(first.xcorrection_value)

        lines = []
        for s in spectra:
            lines.append({
                "x": s.x,
                "y": s.y,
                "label": s.label or s.fname,
                "color": s.color,
                "_spectrum_ref": s, 
            })

        self.spectra_selection_changed.emit(lines)

    def _plot_baseline(self, spectrum):
        baseline = spectrum.baseline
        if not baseline or not baseline.points:
            return

        xs, ys = baseline.points
        if not xs:
            return

        self.ax.plot(
            xs, ys,
            "o--",
            color="orange",
            ms=5,
            lw=1,
            label="_baseline_points"
        )

    def _plot_peaks(self, spectrum):
        if not hasattr(spectrum, "peak_models"):
            return

        self._fitted_lines = []

        x = spectrum.x

        for peak_model, label in zip(spectrum.peak_models, spectrum.peak_labels):
            y = self._evaluate_peak_model(peak_model, x)

            line, = self.ax.plot(
                x, y,
                lw=self.spin_lw.value(),
                label=label,
                alpha=0.9
            )

            self._fitted_lines.append((line, peak_model))

    def _evaluate_peak_model(self, peak_model, x):
        param_hints_orig = peak_model.param_hints.copy()

        for key in peak_model.param_hints:
            peak_model.param_hints[key]["expr"] = ""

        params = peak_model.make_params()
        y = peak_model.eval(params, x=x)

        peak_model.param_hints = param_hints_orig
        return y



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
        self._emit_selection_plot()


    def undo_x_correction(self):
        """Undo X-axis correction for selected spectra."""
        if not self.selected_indices:
            self.notify.emit("No spectrum selected.")
            return

        spectra = self.spectra.get(self.selected_indices)

        for spectrum in spectra:
            spectrum.undo_xcorrection()

        self.show_xcorrection_value.emit(spectra[0].xcorrection_value)
        self._emit_selection_plot()


    def add_peak_at(self, x: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        maxshift = 20
        maxfwhm = 200

        spectrum.add_peak_model(
            spectrum.peak_model if hasattr(spectrum, "peak_model") else "Lorentzian",
            x,
            dx0=(maxshift, maxshift),
            dfwhm=maxfwhm,
        )
        self._emit_selection_plot()

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
        self._emit_selection_plot()

    
    def add_baseline_point(self, x: float, y: float):
        if not self.selected_indices:
            return

        spectrum = self.spectra.get(self.selected_indices)[0]

        if spectrum.baseline.is_subtracted:
            self.notify.emit("Baseline already subtracted.")
            return

        spectrum.baseline.add_point(x, y)
        self._emit_selection_plot()


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

        self._emit_selection_plot()
