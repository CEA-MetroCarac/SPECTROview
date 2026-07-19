#spectroview/model/m_plot_recipe_store.py

import glob
import json
import os
from typing import Dict, List, Optional

from spectroview.model.m_plot_recipe import MPlotRecipe


class PlotRecipeSummary:
    """Lightweight representation of a plot recipe for lists."""
    def __init__(self, id: str, name: str, created_at: str, graph_count: int, filepath: str = ""):
        self.id = id
        self.name = name
        self.created_at = created_at
        self.graph_count = graph_count
        self.filepath = filepath


class MPlotRecipeStore:
    """Manages all saved plot recipes in the configured recipe folder."""

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self._index: Dict[str, PlotRecipeSummary] = {}

        if self.folder_path and not os.path.exists(self.folder_path):
            try:
                os.makedirs(self.folder_path)
            except Exception as e:
                print(f"Error creating recipe folder {self.folder_path}: {e}")

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

                recipe_id = data.get("id")
                if not recipe_id:
                    continue

                self._index[recipe_id] = PlotRecipeSummary(
                    id=recipe_id,
                    name=data.get("name", "Unknown Recipe"),
                    created_at=data.get("created_at", ""),
                    graph_count=len(data.get("configs", [])),
                    filepath=filepath,
                )
            except Exception as e:
                print(f"Error scanning {filepath}: {e}")

    def list_recipes(self) -> List[PlotRecipeSummary]:
        """Returns summaries sorted by created_at descending."""
        summaries = list(self._index.values())
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    def get_summary(self, recipe_id: str) -> Optional[PlotRecipeSummary]:
        return self._index.get(recipe_id)

    def load_recipe(self, recipe_id: str) -> Optional[MPlotRecipe]:
        """Load a full recipe from disk by ID."""
        summary = self._index.get(recipe_id)
        if summary and summary.filepath and os.path.exists(summary.filepath):
            return MPlotRecipe(summary.filepath)
        return None

    def save_recipe(self, name: str, configs: List[dict]) -> Optional[MPlotRecipe]:
        """Create and persist a new recipe from a list of plot configs."""
        if not configs:
            return None
        recipe = MPlotRecipe()
        recipe.name = name
        recipe.configs = configs
        recipe.save(self.folder_path)
        self._index[recipe.id] = PlotRecipeSummary(
            id=recipe.id,
            name=recipe.name,
            created_at=recipe.created_at,
            graph_count=len(configs),
            filepath=recipe._filepath or "",
        )
        return recipe

    def delete_recipe(self, recipe_id: str) -> bool:
        """Delete recipe JSON file and remove from index."""
        summary = self._index.get(recipe_id)
        if not summary or not summary.filepath:
            return False
        try:
            if os.path.exists(summary.filepath):
                os.remove(summary.filepath)
            if recipe_id in self._index:
                del self._index[recipe_id]
            return True
        except Exception as e:
            print(f"Error deleting recipe {recipe_id}: {e}")
            return False
