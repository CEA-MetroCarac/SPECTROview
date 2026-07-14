#spectroview/model/m_plot_template_store.py

import glob
import json
import os
from typing import Dict, List, Optional

from spectroview.model.m_plot_template import MPlotTemplate


class PlotTemplateSummary:
    """Lightweight representation of a plot template for lists."""
    def __init__(self, id: str, name: str, created_at: str, graph_count: int, filepath: str = ""):
        self.id = id
        self.name = name
        self.created_at = created_at
        self.graph_count = graph_count
        self.filepath = filepath


class MPlotTemplateStore:
    """Manages all saved plot templates in the configured template folder."""

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self._index: Dict[str, PlotTemplateSummary] = {}

        if self.folder_path and not os.path.exists(self.folder_path):
            try:
                os.makedirs(self.folder_path)
            except Exception as e:
                print(f"Error creating template folder {self.folder_path}: {e}")

        self.scan_folder()

    def scan_folder(self) -> None:
        """Refresh index from disk."""
        self._index.clear()

        if not self.folder_path or not os.path.exists(self.folder_path):
            return

        for filepath in glob.glob(os.path.join(self.folder_path, "*.json")):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                tpl_id = data.get("id")
                if not tpl_id:
                    continue

                self._index[tpl_id] = PlotTemplateSummary(
                    id=tpl_id,
                    name=data.get("name", "Unknown Template"),
                    created_at=data.get("created_at", ""),
                    graph_count=len(data.get("configs", [])),
                    filepath=filepath,
                )
            except Exception as e:
                print(f"Error scanning {filepath}: {e}")

    def list_templates(self) -> List[PlotTemplateSummary]:
        """Returns summaries sorted by created_at descending."""
        summaries = list(self._index.values())
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    def get_summary(self, template_id: str) -> Optional[PlotTemplateSummary]:
        return self._index.get(template_id)

    def load_template(self, template_id: str) -> Optional[MPlotTemplate]:
        """Load a full template from disk by ID."""
        summary = self._index.get(template_id)
        if summary and summary.filepath and os.path.exists(summary.filepath):
            return MPlotTemplate(summary.filepath)
        return None

    def save_template(self, name: str, configs: List[dict]) -> Optional[MPlotTemplate]:
        """Create and persist a new template from a list of plot configs."""
        if not configs:
            return None
        tpl = MPlotTemplate()
        tpl.name = name
        tpl.configs = configs
        tpl.save(self.folder_path)
        self._index[tpl.id] = PlotTemplateSummary(
            id=tpl.id,
            name=tpl.name,
            created_at=tpl.created_at,
            graph_count=len(configs),
            filepath=tpl._filepath or "",
        )
        return tpl

    def delete_template(self, template_id: str) -> bool:
        """Delete template JSON file and remove from index."""
        summary = self._index.get(template_id)
        if not summary or not summary.filepath:
            return False
        try:
            if os.path.exists(summary.filepath):
                os.remove(summary.filepath)
            if template_id in self._index:
                del self._index[template_id]
            return True
        except Exception as e:
            print(f"Error deleting template {template_id}: {e}")
            return False
