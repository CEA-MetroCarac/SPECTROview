"""
Tests for spectroview/model/m_plot_template.py and m_plot_template_store.py.

Mirrors the existing MConversationStore test patterns (save/load/list/
rename/duplicate/delete) since MPlotTemplateStore is deliberately modeled
on that class.
"""
import json
import os

import pytest

from spectroview.model.m_plot_template import MPlotTemplate
from spectroview.model.m_plot_template_store import MPlotTemplateStore


SAMPLE_CONFIGS = [
    {"x": "Slot", "y": "fwhm_Si", "plot_style": "point", "df_name": "fit_results", "grid": True},
    {"x": "Slot", "y": "x0_Si", "plot_style": "box", "df_name": "fit_results"},
]


@pytest.fixture
def store(tmp_path):
    return MPlotTemplateStore(str(tmp_path / "templates"))


class TestMPlotTemplateSaveLoad:
    def test_save_creates_a_json_file(self, tmp_path):
        tpl = MPlotTemplate()
        tpl.name = "My Template"
        tpl.configs = SAMPLE_CONFIGS
        tpl.save(str(tmp_path))

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        assert "My_Template" in files[0].name

    def test_empty_template_is_not_saved(self, tmp_path):
        tpl = MPlotTemplate()
        tpl.name = "Empty"
        tpl.save(str(tmp_path))
        assert list(tmp_path.glob("*.json")) == []

    def test_load_roundtrip(self, tmp_path):
        tpl = MPlotTemplate()
        tpl.name = "Roundtrip"
        tpl.configs = SAMPLE_CONFIGS
        tpl.save(str(tmp_path))

        loaded = MPlotTemplate(tpl._filepath)
        assert loaded.id == tpl.id
        assert loaded.name == "Roundtrip"
        assert loaded.configs == SAMPLE_CONFIGS

    def test_graph_count_reflects_configs(self):
        tpl = MPlotTemplate()
        tpl.configs = SAMPLE_CONFIGS
        assert tpl.graph_count == 2

    def test_duplicate_gets_new_id_and_copied_configs(self):
        tpl = MPlotTemplate()
        tpl.name = "Original"
        tpl.configs = SAMPLE_CONFIGS
        dup = tpl.duplicate()

        assert dup.id != tpl.id
        assert dup.name == "Original (Copy)"
        assert dup.configs == tpl.configs
        dup.configs[0]["x"] = "Changed"
        assert tpl.configs[0]["x"] == "Slot"  # deep copy, not shared

    def test_rename(self):
        tpl = MPlotTemplate()
        tpl.rename("New Name")
        assert tpl.name == "New Name"


class TestMPlotTemplateStore:
    def test_scan_folder_indexes_saved_templates(self, tmp_path):
        tpl = MPlotTemplate()
        tpl.name = "Indexed"
        tpl.configs = SAMPLE_CONFIGS
        tpl.save(str(tmp_path))

        store = MPlotTemplateStore(str(tmp_path))
        summaries = store.list_templates()
        assert len(summaries) == 1
        assert summaries[0].name == "Indexed"
        assert summaries[0].graph_count == 2

    def test_save_template_creates_and_indexes(self, store):
        tpl = store.save_template("Quick Save", SAMPLE_CONFIGS)
        assert tpl is not None
        assert len(store.list_templates()) == 1
        assert store.get_summary(tpl.id).name == "Quick Save"

    def test_save_template_with_empty_configs_returns_none(self, store):
        result = store.save_template("Nothing", [])
        assert result is None
        assert store.list_templates() == []

    def test_load_template_returns_full_configs(self, store):
        saved = store.save_template("Full", SAMPLE_CONFIGS)
        loaded = store.load_template(saved.id)
        assert loaded is not None
        assert loaded.configs == SAMPLE_CONFIGS

    def test_get_summary_unknown_id_returns_none(self, store):
        assert store.get_summary("nonexistent") is None

    def test_delete_template_removes_file_and_index_entry(self, store):
        tpl = store.save_template("ToDelete", SAMPLE_CONFIGS)
        filepath = tpl._filepath
        assert os.path.exists(filepath)

        result = store.delete_template(tpl.id)
        assert result is True
        assert not os.path.exists(filepath)
        assert store.get_summary(tpl.id) is None

    def test_delete_unknown_template_returns_false(self, store):
        assert store.delete_template("nonexistent") is False

    def test_list_templates_sorted_newest_first(self, store, monkeypatch):
        import spectroview.model.m_plot_template as mpt

        times = iter(["2026-01-01T00:00:00", "2026-01-02T00:00:00", "2026-01-03T00:00:00"])

        class _FixedDatetime:
            @staticmethod
            def now():
                class _D:
                    def isoformat(self_inner):
                        return next(times)
                return _D()

        monkeypatch.setattr(mpt, "datetime", _FixedDatetime)

        store.save_template("First", SAMPLE_CONFIGS)
        store.save_template("Second", SAMPLE_CONFIGS)
        store.save_template("Third", SAMPLE_CONFIGS)

        names = [s.name for s in store.list_templates()]
        assert names == ["Third", "Second", "First"]

    def test_duplicate_via_load_and_save(self, store):
        original = store.save_template("Original", SAMPLE_CONFIGS)
        loaded = store.load_template(original.id)
        dup = loaded.duplicate()
        dup.save(store.folder_path)
        store.scan_folder()

        names = sorted(s.name for s in store.list_templates())
        assert names == ["Original", "Original (Copy)"]

    def test_replicate_graph_style_config_shape_round_trips(self, store):
        """Templates built from a live MGraph.save() dict (the
        "save all open graphs" path) use a much larger field set than the
        AI-tool-shaped configs — both must persist and reload unchanged."""
        mgraph_style_config = {
            "grid": True, "plot_style": "scatter", "plot_width": 480,
            "plot_height": 420, "dpi": 100, "df_name": "fit_results",
            "x": "Slot", "y": ["fwhm_Si"], "z": None, "xmin": None,
            "color_palette": "jet", "legend_visible": True,
        }
        tpl = store.save_template("From Graph", [mgraph_style_config])
        loaded = store.load_template(tpl.id)
        assert loaded.configs == [mgraph_style_config]
