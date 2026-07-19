"""Tests for model/m_style_template_store.py -- mirrors
test_m_plot_recipe.py's patterns for its sibling store (save/load/list/
rename/delete), scoped down since a style template is always exactly one
style dict rather than a list of full graph configs.
"""
import pytest

from spectroview.model.m_style_template_store import MStyleTemplateStore

SAMPLE_STYLE = {"color_palette": "viridis", "grid": True, "title_fontsize": 16}


@pytest.fixture
def store(tmp_path):
    return MStyleTemplateStore(str(tmp_path / "styles"))


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, store):
        tpl_id = store.save_template("My Style", SAMPLE_STYLE)
        assert tpl_id is not None
        loaded = store.load_style(tpl_id)
        assert loaded == SAMPLE_STYLE

    def test_save_with_empty_style_returns_none(self, store):
        assert store.save_template("Empty", {}) is None

    def test_save_without_configured_folder_returns_none(self):
        store = MStyleTemplateStore("")
        assert store.save_template("X", SAMPLE_STYLE) is None

    def test_load_unknown_id_returns_none(self, store):
        assert store.load_style("not-a-real-id") is None


class TestIndexAndList:
    def test_save_template_indexes_it(self, store):
        tpl_id = store.save_template("Style A", SAMPLE_STYLE)
        summaries = store.list_templates()
        assert len(summaries) == 1
        assert summaries[0].id == tpl_id
        assert summaries[0].name == "Style A"

    def test_get_summary_unknown_id_returns_none(self, store):
        assert store.get_summary("nope") is None

    def test_list_templates_sorted_newest_first(self, store, monkeypatch):
        import spectroview.model.m_style_template_store as mod

        times = iter(["2026-01-01T00:00:00", "2026-01-02T00:00:00"])

        class _FixedDatetime:
            @staticmethod
            def now():
                class _D:
                    def isoformat(self_inner):
                        return next(times)
                return _D()

        monkeypatch.setattr(mod, "datetime", _FixedDatetime)

        store.save_template("Older", SAMPLE_STYLE)
        store.save_template("Newer", SAMPLE_STYLE)
        names = [s.name for s in store.list_templates()]
        assert names == ["Newer", "Older"]

    def test_scan_folder_rebuilds_index_from_disk(self, tmp_path):
        store1 = MStyleTemplateStore(str(tmp_path / "styles"))
        tpl_id = store1.save_template("Persisted", SAMPLE_STYLE)

        store2 = MStyleTemplateStore(str(tmp_path / "styles"))  # fresh instance, same folder
        assert store2.get_summary(tpl_id) is not None
        assert store2.load_style(tpl_id) == SAMPLE_STYLE


class TestDelete:
    def test_delete_removes_file_and_index_entry(self, store, tmp_path):
        tpl_id = store.save_template("ToDelete", SAMPLE_STYLE)
        summary = store.get_summary(tpl_id)
        import os
        assert os.path.exists(summary.filepath)

        assert store.delete_template(tpl_id) is True
        assert store.get_summary(tpl_id) is None
        assert not os.path.exists(summary.filepath)

    def test_delete_unknown_id_returns_false(self, store):
        assert store.delete_template("nope") is False
