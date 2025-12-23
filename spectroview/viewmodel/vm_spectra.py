from PySide6.QtCore import QObject, Signal
from pathlib import Path

from PySide6.QtWidgets import QFileDialog


from spectroview.model.m_spectra import SpectraM
from spectroview.model.m_io import load_spectrum_file


class SpectraVM(QObject):
    # ───── ViewModel → View signals ─────
    spectra_list_changed = Signal(list)      # list[str]
    spectra_selection_changed = Signal(list) # list[dict] → plot data
    count_changed = Signal(int)

    notify = Signal(str)  # general notifications
    
    def __init__(self):
        super().__init__()
        self.spectra = SpectraM()
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
        self.selected_indices = indices
        self._emit_selection_plot()
        
    def open_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            None,
            "Open spectra",
            "",
            "Data (*.txt *.csv)"
        )
        if paths:
            self.load_files(paths)    
    
    # Internal helpers
    def _emit_list_update(self):
        names = [s.fname for s in self.spectra]
        self.spectra_list_changed.emit(names)
        self.count_changed.emit(len(self.spectra))

    def _emit_selection_plot(self):
        spectra = self.spectra.get(self.selected_indices)

        lines = []
        for s in spectra:
            lines.append({
                "x": s.x,
                "y": s.y,
                "label": s.label or s.fname,
                "color": s.color,
            })

        self.spectra_selection_changed.emit(lines)
