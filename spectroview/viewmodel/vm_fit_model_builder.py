from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog
from pathlib import Path

from spectroview.model.m_fit_model_manager import MFitModelManager
from spectroview.model.m_settings import MSettings


class VMFitModelBuilder(QObject):
    # ───── VM → View signals ──────────────────────────────
    models_changed = Signal(list)          # list[str]
    model_selected = Signal(str)
    notify = Signal(str)
    model_applied = Signal(str)            # full path


    def __init__(self, settings: MSettings):
        super().__init__()
        self.settings = settings
        self.model_manager = MFitModelManager()

        self._current_model_name: str | None = None
        self._extra_models: dict[str, Path] = {}  # loaded via "Load"

        self.refresh_models()

    # ─────────────────────────────────────────────────────
    # View → VM
    # ─────────────────────────────────────────────────────
    def refresh_models(self):
        """Reload models from default model folder only."""
        folder = self.settings.get_model_folder()

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

    def pick_and_load_model(self):
        """Load a JSON model without changing default model folder."""
        start_dir = self.settings.get_model_folder() or ""

        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Fit Model",
            start_dir,
            "JSON files (*.json)"
        )

        if not file_path:
            return

        path = Path(file_path)
        model_name = path.name

        # Store external model separately
        self._extra_models[model_name] = path

        # Update combobox WITHOUT rescanning folders
        all_models = (
            self.model_manager.available_models +
            list(self._extra_models.keys())
        )

        self.models_changed.emit(all_models)

        self._current_model_name = model_name
        self.model_selected.emit(model_name)

    def apply_model(self, model_name: str):
        """Apply selected model (default or external)."""

        print(f"Applying model: {model_name}")
        if not model_name:
            self.notify.emit("No fit model selected.")
            return

        # External model?
        if model_name in self._extra_models:
            model_path = self._extra_models[model_name]
        else:
            model_path = Path(self.model_manager.resolve_path(model_name))

        if not model_path.exists():
            self.notify.emit(f"Model not found:\n{model_path}")
            return

        self._current_model_name = model_name
        self.model_applied.emit(str(model_path))

    # ─────────────────────────────────────────────────────
    def current_model(self) -> str | None:
        return self._current_model_name
