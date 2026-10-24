"""ViewModel for Fit Model Builder - manages fit model selection and loading."""
import json
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from spectroview.model.m_fit_model_manager import MFitModelManager
from spectroview.model.m_settings import MSettings


class VMFitModelBuilder(QObject):
    # ───── VM → View signals ──────────────────────────────
    models_changed = Signal(list)          # list[str]
    model_selected = Signal(str)
    notify = Signal(str)

    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings
        self.model_manager = MFitModelManager()

        self._current_model_name: str | None = None
        self._extra_models: dict[str, Path] = {}  # loaded via "Load"

        # Note: refresh_models() is deliberately not called here. At
        # construction time the View hasn't connected models_changed/
        # model_selected yet, so an early call would scan the folder for no
        # observable effect. The View calls refresh_models() once its
        # connections are wired (see VWorkspaceSpectra.setup_connections()).

    # View → VM
    def refresh_models(self):
        """Reload models from default model folder only."""
        folder = self.settings.get_fit_model_folder()

        if not folder:
            self.models_changed.emit([])
            self.notify.emit("No model folder defined in Settings.")
            return

        base_models = self.model_manager.scan_folder(folder)

        # Merge default + externally loaded models
        all_models = base_models + [
            name for name in self._extra_models
            if name not in base_models
        ]

        self.models_changed.emit(all_models)

        if all_models:
            self._current_model_name = all_models[0]
            self.model_selected.emit(self._current_model_name)

    def load_model_file(self, file_path: str | Path) -> bool:
        """Load a JSON fit model file without changing default model folder."""
        path = Path(file_path)
        if not path.is_file() or path.suffix.lower() != ".json":
            self.notify.emit(f"Invalid fit model file: {path.name}")
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                self.notify.emit(f"Invalid fit model format: {path.name} must be a JSON object.")
                return False
        except Exception as e:
            self.notify.emit(f"Failed to read fit model {path.name}: {e}")
            return False

        model_name = path.name

        # Store external model separately
        self._extra_models[model_name] = path

        # Update combobox WITHOUT rescanning folders
        base_models = self.model_manager.available_models
        all_models = base_models + [
            name for name in self._extra_models
            if name not in base_models
        ]

        self.models_changed.emit(all_models)

        self._current_model_name = model_name
        self.model_selected.emit(model_name)
        return True

    def pick_and_load_model(self):
        """Load a JSON model without changing default model folder."""
        start_dir = self.settings.get_fit_model_folder() or ""

        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Fit Model",
            start_dir,
            "JSON files (*.json)"
        )

        if not file_path:
            return

        self.load_model_file(file_path)

    def set_current_model(self, model_name: str):
        """Called when user changes combobox selection."""
        if not model_name:
            return
        self._current_model_name = model_name

    def get_current_model_path(self) -> Path | None:
        """Return the full path of the currently selected model."""
        if not self._current_model_name:
            return None

        if self._current_model_name in self._extra_models:
            return self._extra_models[self._current_model_name]

        resolved = self.model_manager.resolve_path(self._current_model_name)
        return Path(resolved) if resolved else None