# model/m_style_template_store.py
"""Per-graph style templates: save/apply just the "look" of a graph (see
model/graph_style.py for the style/data field partition), not its data
bindings.

A separate, deliberately much simpler persistence mechanism from
MPlotRecipeStore's whole-workspace recipes (a style template is always
exactly one style dict, applied to one target graph at a time) -- kept
separate rather than overloading MPlotRecipe's `configs` list so the two
concepts can never be confused with each other in a browse dialog.
"""
import glob
import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class StyleTemplateSummary:
    """Lightweight representation of a style template for lists."""
    def __init__(self, id: str, name: str, created_at: str, filepath: str = ""):
        self.id = id
        self.name = name
        self.created_at = created_at
        self.filepath = filepath


class MStyleTemplateStore:
    """Manages all saved per-graph style templates in a folder."""

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self._index: Dict[str, StyleTemplateSummary] = {}

        if self.folder_path and not os.path.exists(self.folder_path):
            try:
                os.makedirs(self.folder_path)
            except Exception as e:
                print(f"Error creating style template folder {self.folder_path}: {e}")

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

                self._index[tpl_id] = StyleTemplateSummary(
                    id=tpl_id,
                    name=data.get("name", "Unknown Style"),
                    created_at=data.get("created_at", ""),
                    filepath=filepath,
                )
            except Exception as e:
                print(f"Error scanning {filepath}: {e}")

    def list_templates(self) -> List[StyleTemplateSummary]:
        """Returns summaries sorted by created_at descending."""
        summaries = list(self._index.values())
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    def get_summary(self, template_id: str) -> Optional[StyleTemplateSummary]:
        return self._index.get(template_id)

    def save_template(self, name: str, style: Dict[str, Any]) -> Optional[str]:
        """Persist a style dict under `name`. Returns the new template's id,
        or None if the folder isn't configured or `style` is empty."""
        if not style or not self.folder_path:
            return None

        tpl_id = str(uuid.uuid4())
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:40].strip('_') or "Untitled"
        filepath = os.path.join(self.folder_path, f"{safe_name}_{tpl_id}.json")
        created_at = datetime.now().isoformat()

        data = {"id": tpl_id, "name": name, "created_at": created_at, "style": style}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving style template: {e}")
            return None

        self._index[tpl_id] = StyleTemplateSummary(
            id=tpl_id, name=name, created_at=created_at, filepath=filepath,
        )
        return tpl_id

    def load_style(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved style dict from disk by template id."""
        summary = self._index.get(template_id)
        if not summary or not summary.filepath or not os.path.exists(summary.filepath):
            return None
        try:
            with open(summary.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("style")
        except Exception as e:
            print(f"Error loading style template {template_id}: {e}")
            return None

    def delete_template(self, template_id: str) -> bool:
        """Delete a style template's JSON file and remove it from the index."""
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
            print(f"Error deleting style template {template_id}: {e}")
            return False
