#spectroview/model/m_plot_template.py

import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class MPlotTemplate:
    """A saved, reusable set of one or more plot configurations.

    Each entry in ``configs`` is a plot-config dict compatible with
    ``VWorkspaceGraphs.create_plot_from_config()`` — either the narrower
    shape produced by the AI agent's tools, or a full ``MGraph.save()``
    dict (minus ``graph_id``, mirroring the existing "replicate graph"
    pattern in ``v_workspace_graphs.py``). Both shapes are already
    consumed by the same code path today, so no conversion is needed.
    """

    def __init__(self, filepath: Optional[str] = None):
        self.id: str = str(uuid.uuid4())
        self.name: str = "New Template"
        self.created_at: str = datetime.now().isoformat()
        self.configs: List[Dict[str, Any]] = []
        self._filepath: Optional[str] = filepath

        if filepath and os.path.exists(filepath):
            self.load(filepath)

    @property
    def graph_count(self) -> int:
        return len(self.configs)

    def rename(self, new_name: str) -> None:
        self.name = new_name

    def duplicate(self) -> "MPlotTemplate":
        """Create a deep copy with a new ID."""
        import copy
        new_tpl = MPlotTemplate()
        new_tpl.name = f"{self.name} (Copy)"
        new_tpl.configs = copy.deepcopy(self.configs)
        return new_tpl

    def save(self, folder: Optional[str] = None) -> None:
        """Write the template to a JSON file."""
        if not self.configs:
            return  # don't save an empty template

        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', self.name)
        safe_name = safe_name[:40].strip('_') or "Untitled"
        new_filename = f"{safe_name}_{self.id}.json"

        if folder:
            new_filepath = os.path.join(folder, new_filename)
        elif self._filepath:
            new_filepath = os.path.join(os.path.dirname(self._filepath), new_filename)
        else:
            return  # can't save without a path

        if self._filepath and self._filepath != new_filepath and os.path.exists(self._filepath):
            try:
                os.remove(self._filepath)
            except Exception:
                pass

        self._filepath = new_filepath
        folder_path = os.path.dirname(self._filepath)
        if folder_path and not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
            except Exception:
                pass

        data = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "configs": self.configs,
        }
        try:
            with open(self._filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving plot template: {e}")

    def load(self, filepath: str) -> None:
        """Deserialize from JSON."""
        self._filepath = filepath
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.id = data.get("id", self.id)
            self.name = data.get("name", self.name)
            self.created_at = data.get("created_at", self.created_at)
            self.configs = data.get("configs", [])
        except Exception as e:
            print(f"Error loading plot template {filepath}: {e}")
