"""
Tests for spectroview/model/m_plot_recipe.py and m_plot_recipe_store.py.

Mirrors the existing MConversationStore test patterns (save/load/list/
rename/duplicate/delete) since MPlotRecipeStore is deliberately modeled
on that class.
"""
import json
import os

import pytest

from spectroview.model.m_plot_recipe import MPlotRecipe
from spectroview.model.m_plot_recipe_store import MPlotRecipeStore


SAMPLE_CONFIGS = [
    {"x": "Slot", "y": "fwhm_Si", "plot_style": "point", "df_name": "fit_results", "grid": True},
    {"x": "Slot", "y": "x0_Si", "plot_style": "box", "df_name": "fit_results"},
]


@pytest.fixture
def store(tmp_path):
    return MPlotRecipeStore(str(tmp_path / "recipes"))


class TestMPlotRecipeSaveLoad:
    def test_save_creates_a_json_file(self, tmp_path):
        recipe = MPlotRecipe()
        recipe.name = "My Recipe"
        recipe.configs = SAMPLE_CONFIGS
        recipe.save(str(tmp_path))

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        assert "My_Recipe" in files[0].name

    def test_empty_recipe_is_not_saved(self, tmp_path):
        recipe = MPlotRecipe()
        recipe.name = "Empty"
        recipe.save(str(tmp_path))
        assert list(tmp_path.glob("*.json")) == []

    def test_load_roundtrip(self, tmp_path):
        recipe = MPlotRecipe()
        recipe.name = "Roundtrip"
        recipe.configs = SAMPLE_CONFIGS
        recipe.save(str(tmp_path))

        loaded = MPlotRecipe(recipe._filepath)
        assert loaded.id == recipe.id
        assert loaded.name == "Roundtrip"
        assert loaded.configs == SAMPLE_CONFIGS

    def test_graph_count_reflects_configs(self):
        recipe = MPlotRecipe()
        recipe.configs = SAMPLE_CONFIGS
        assert recipe.graph_count == 2

    def test_duplicate_gets_new_id_and_copied_configs(self):
        recipe = MPlotRecipe()
        recipe.name = "Original"
        recipe.configs = SAMPLE_CONFIGS
        dup = recipe.duplicate()

        assert dup.id != recipe.id
        assert dup.name == "Original (Copy)"
        assert dup.configs == recipe.configs
        dup.configs[0]["x"] = "Changed"
        assert recipe.configs[0]["x"] == "Slot"  # deep copy, not shared

    def test_rename(self):
        recipe = MPlotRecipe()
        recipe.rename("New Name")
        assert recipe.name == "New Name"


class TestMPlotRecipeStore:
    def test_scan_folder_indexes_saved_recipes(self, tmp_path):
        recipe = MPlotRecipe()
        recipe.name = "Indexed"
        recipe.configs = SAMPLE_CONFIGS
        recipe.save(str(tmp_path))

        store = MPlotRecipeStore(str(tmp_path))
        summaries = store.list_recipes()
        assert len(summaries) == 1
        assert summaries[0].name == "Indexed"
        assert summaries[0].graph_count == 2

    def test_save_recipe_creates_and_indexes(self, store):
        recipe = store.save_recipe("Quick Save", SAMPLE_CONFIGS)
        assert recipe is not None
        assert len(store.list_recipes()) == 1
        assert store.get_summary(recipe.id).name == "Quick Save"

    def test_save_recipe_with_empty_configs_returns_none(self, store):
        result = store.save_recipe("Nothing", [])
        assert result is None
        assert store.list_recipes() == []

    def test_load_recipe_returns_full_configs(self, store):
        saved = store.save_recipe("Full", SAMPLE_CONFIGS)
        loaded = store.load_recipe(saved.id)
        assert loaded is not None
        assert loaded.configs == SAMPLE_CONFIGS

    def test_get_summary_unknown_id_returns_none(self, store):
        assert store.get_summary("nonexistent") is None

    def test_delete_recipe_removes_file_and_index_entry(self, store):
        recipe = store.save_recipe("ToDelete", SAMPLE_CONFIGS)
        filepath = recipe._filepath
        assert os.path.exists(filepath)

        result = store.delete_recipe(recipe.id)
        assert result is True
        assert not os.path.exists(filepath)
        assert store.get_summary(recipe.id) is None

    def test_delete_unknown_recipe_returns_false(self, store):
        assert store.delete_recipe("nonexistent") is False

    def test_list_recipes_sorted_newest_first(self, store, monkeypatch):
        import spectroview.model.m_plot_recipe as mpr

        times = iter(["2026-01-01T00:00:00", "2026-01-02T00:00:00", "2026-01-03T00:00:00"])

        class _FixedDatetime:
            @staticmethod
            def now():
                class _D:
                    def isoformat(self_inner):
                        return next(times)
                return _D()

        monkeypatch.setattr(mpr, "datetime", _FixedDatetime)

        store.save_recipe("First", SAMPLE_CONFIGS)
        store.save_recipe("Second", SAMPLE_CONFIGS)
        store.save_recipe("Third", SAMPLE_CONFIGS)

        names = [s.name for s in store.list_recipes()]
        assert names == ["Third", "Second", "First"]

    def test_duplicate_via_load_and_save(self, store):
        original = store.save_recipe("Original", SAMPLE_CONFIGS)
        loaded = store.load_recipe(original.id)
        dup = loaded.duplicate()
        dup.save(store.folder_path)
        store.scan_folder()

        names = sorted(s.name for s in store.list_recipes())
        assert names == ["Original", "Original (Copy)"]

    def test_replicate_graph_style_config_shape_round_trips(self, store):
        """Recipes built from a live MGraph.save() dict (the
        "save all open graphs" path) use a much larger field set than the
        AI-tool-shaped configs — both must persist and reload unchanged."""
        mgraph_style_config = {
            "grid": True, "plot_style": "scatter", "plot_width": 480,
            "plot_height": 420, "dpi": 100, "df_name": "fit_results",
            "x": "Slot", "y": ["fwhm_Si"], "z": None, "xmin": None,
            "color_palette": "jet", "legend_visible": True,
        }
        recipe = store.save_recipe("From Graph", [mgraph_style_config])
        loaded = store.load_recipe(recipe.id)
        assert loaded.configs == [mgraph_style_config]
