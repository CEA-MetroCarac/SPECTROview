"""Tests for model/graph_style.py -- the style/data field partition backing
Phase 5's per-graph style templates, copy/paste style, and reset-to-default.
"""
import dataclasses

import pytest

from spectroview.model.m_graph import MGraph
from spectroview.model.graph_style import (
    STYLE_FIELD_NAMES, _NON_STYLE_FIELDS, extract_style, default_style, apply_style_dict,
    RESTYLE_SAFE_FIELDS, can_restyle_without_replot,
)


class TestPartition:
    def test_covers_every_mgraph_field_exactly_once(self):
        all_fields = {f.name for f in dataclasses.fields(MGraph)}
        assert set(STYLE_FIELD_NAMES) | _NON_STYLE_FIELDS == all_fields
        assert set(STYLE_FIELD_NAMES) & _NON_STYLE_FIELDS == set()

    def test_data_identity_fields_are_excluded(self):
        for field_name in ('graph_id', 'df_name', 'x', 'y', 'z', 'filters', 'plot_style'):
            assert field_name not in STYLE_FIELD_NAMES

    def test_data_range_fields_are_excluded(self):
        for field_name in ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax',
                            'annotations', 'axis_breaks',
                            'inset_xmin', 'inset_xmax', 'inset_ymin', 'inset_ymax'):
            assert field_name not in STYLE_FIELD_NAMES

    def test_visual_fields_are_included(self):
        for field_name in ('color_palette', 'grid', 'title_fontsize', 'figure_theme',
                            'legend_ncol', 'scatter_size', 'colormap_norm', 'inset_bounds'):
            assert field_name in STYLE_FIELD_NAMES


class TestExtractStyle:
    def test_pulls_only_style_fields(self):
        g = MGraph(graph_id=7, x='foo', y=['bar'], color_palette='viridis', grid=True)
        style = extract_style(g.save())
        assert 'x' not in style
        assert 'graph_id' not in style
        assert style['color_palette'] == 'viridis'
        assert style['grid'] is True

    def test_deep_copies_mutable_values(self):
        g = MGraph(graph_id=1)
        g.spines_visible = {'top': False, 'right': True, 'bottom': True, 'left': True}
        style = extract_style(g.save())
        style['spines_visible']['top'] = True
        assert g.spines_visible['top'] is False  # source untouched

    def test_missing_keys_in_source_are_skipped_not_defaulted(self):
        # A partial source dict (e.g. an old/hand-built config) shouldn't
        # raise KeyError, and shouldn't silently invent missing fields.
        style = extract_style({'color_palette': 'plasma'})
        assert style == {'color_palette': 'plasma'}


class TestDefaultStyle:
    def test_matches_a_fresh_mgraph(self):
        ds = default_style()
        fresh = MGraph()
        for key, value in ds.items():
            assert value == getattr(fresh, key)

    def test_excludes_data_fields(self):
        ds = default_style()
        assert 'x' not in ds and 'df_name' not in ds


class TestApplyStyleDict:
    def test_writes_style_fields_onto_target(self):
        class FakeGraph:
            pass
        target = FakeGraph()
        target.legend_properties = [{'label': 'stale'}]

        applied = apply_style_dict(target, {'color_palette': 'magma', 'grid': True})
        assert target.color_palette == 'magma'
        assert target.grid is True
        assert applied['color_palette'] == 'magma'

    def test_resets_legend_properties(self):
        class FakeGraph:
            pass
        target = FakeGraph()
        target.legend_properties = [{'label': 'stale'}]
        applied = apply_style_dict(target, {})
        assert target.legend_properties == []
        assert applied['legend_properties'] == []

    def test_ignores_keys_outside_style_whitelist(self):
        """Defensive: a style dict is only ever built via extract_style()/
        default_style() in practice, but apply_style_dict() must not let a
        stray data-field key (e.g. from a hand-edited JSON template file)
        leak through and repoint a graph's data."""
        class FakeGraph:
            pass
        target = FakeGraph()
        target.x = 'original_x'
        target.legend_properties = []
        apply_style_dict(target, {'x': 'hacked', 'color_palette': 'cool'})
        assert target.x == 'original_x'
        assert target.color_palette == 'cool'

    def test_deep_copies_values_onto_target(self):
        class FakeGraph:
            pass
        target = FakeGraph()
        target.legend_properties = []
        source_spines = {'top': False, 'right': True, 'bottom': True, 'left': True}
        apply_style_dict(target, {'spines_visible': source_spines})
        target.spines_visible['top'] = True
        assert source_spines['top'] is False  # source dict untouched


class TestRestyleSafeFields:
    """Phase 5E: RESTYLE_SAFE_FIELDS is a NARROWER cut than STYLE_FIELD_NAMES
    above -- both classify the same MGraph fields, but for different
    questions ("is this a portable look?" vs "can this be repainted onto
    already-plotted artists without a full replot?"). A field can be a
    style field without being restyle-safe (e.g. scatter_size IS a style
    field for templates/copy-paste, but changing it needs the actual
    scatter artist recreated, not just the axes chrome restyled)."""

    def test_restyle_safe_fields_are_a_subset_of_all_mgraph_fields(self):
        import dataclasses
        all_fields = {f.name for f in dataclasses.fields(MGraph)}
        assert RESTYLE_SAFE_FIELDS <= all_fields

    def test_restyle_safe_and_style_fields_are_independent_classifications(self):
        # NOT a subset relationship either way: plot_title/xmin/legend_bbox
        # etc. are restyle-safe (repainting them needs no data) but are
        # excluded from STYLE_FIELD_NAMES for a different reason (5A: not a
        # portable "look" across graphs with different data/ranges), while
        # scatter_size/color_palette are style fields (5A: a portable look)
        # that are NOT restyle-safe (changing them needs the actual series
        # artist recreated). Guard both directions so this doesn't silently
        # collapse into one axis if either set is edited later.
        assert RESTYLE_SAFE_FIELDS - set(STYLE_FIELD_NAMES)  # e.g. plot_title, xmin
        assert set(STYLE_FIELD_NAMES) - RESTYLE_SAFE_FIELDS  # e.g. scatter_size

    def test_restyle_safe_fields_exclude_true_data_identity_fields(self):
        # The one thing that must never happen regardless of which "style"
        # axis a field sits on: an identity/data field must never be
        # restyle-safe (repainting must never be mistaken for a way to
        # rebind what data a graph shows).
        identity_fields = {
            'x', 'y', 'z', 'y2', 'y3', 'x2', 'df_name', 'filters', 'plot_style',
            'annotations', 'axis_breaks', 'legend_properties',
        }
        assert RESTYLE_SAFE_FIELDS.isdisjoint(identity_fields)

    def test_series_appearance_fields_are_not_restyle_safe(self):
        for field_name in ('color_palette', 'scatter_size', 'scatter_edgecolor',
                            'legend_properties', 'figure_theme', 'colormap_norm'):
            assert field_name not in RESTYLE_SAFE_FIELDS

    def test_secondary_axes_fields_are_not_restyle_safe(self):
        for field_name in ('y2min', 'y2max', 'y2color', 'y2label'):
            assert field_name not in RESTYLE_SAFE_FIELDS

    def test_data_identity_fields_are_not_restyle_safe(self):
        for field_name in ('x', 'y', 'z', 'plot_style', 'df_name', 'filters'):
            assert field_name not in RESTYLE_SAFE_FIELDS

    def test_cosmetic_fields_are_restyle_safe(self):
        for field_name in ('title_fontsize', 'grid', 'xmin', 'xmax',
                            'x_inverted', 'legend_ncol', 'tick_direction',
                            'figure_facecolor', 'spines_visible', 'x_rot'):
            assert field_name in RESTYLE_SAFE_FIELDS


class TestCanRestyleWithoutReplot:
    def test_true_for_a_purely_cosmetic_change(self):
        assert can_restyle_without_replot({'title_fontsize', 'grid'}) is True

    def test_false_for_no_changes(self):
        assert can_restyle_without_replot(set()) is False

    def test_false_when_any_field_is_not_restyle_safe(self):
        assert can_restyle_without_replot({'title_fontsize', 'scatter_size'}) is False

    def test_false_for_a_purely_data_relevant_change(self):
        assert can_restyle_without_replot({'color_palette'}) is False
