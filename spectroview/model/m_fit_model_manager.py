#spectroview/model/m_fit_model_manager.py

import os
from pathlib import Path

class MFitModelManager:
    def __init__(self):
        self.model_folder = ""
        self.available_models: list[str] = []

    def scan_folder(self, folder: str) -> list[str]:
        self.available_models.clear()
        self.model_folder = folder

        if not folder or not os.path.isdir(folder):
            return []

        for fname in os.listdir(folder):
            if fname.lower().endswith(".json"):
                self.available_models.append(fname)

        self.available_models.sort()
        return self.available_models

    def resolve_path(self, model_name: str) -> str:
        return str(Path(self.model_folder) / model_name)