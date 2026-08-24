"""Shared Graph command-layer coverage (GUI, AI, and MCP use this module)."""
import dataclasses

import pandas as pd
import pytest

from spectroview.model.graph_control import (
    GRAPH_PROPERTY_SET,
    GraphPatch,
    GraphValidationError,
    apply_graph_patch,
    normalize_graph_patch,
)
from spectroview.model.m_graph import MGraph


def test_patch_schema_covers_every_mutable_mgraph_field():
    expected = {field.name for field in dataclasses.fields(MGraph)} - {"graph_id"}
    assert GRAPH_PROPERTY_SET == expected
    assert set(GraphPatch.model_fields) == expected


def test_multi_property_patch_preserves_unrelated_state():
    graph = MGraph(graph_id=7)
    graph.x = "frequency"
    graph.y = ["intensity"]
    graph.ylabel = "Keep me"
    graph.legend_properties = [
        {"label": "A", "color": "red", "marker": "o", "linewidth": 1.0},
        {"label": "B", "color": "blue", "marker": "s", "linewidth": 1.5},
    ]
    styled = [dict(item, linewidth=3.0) for item in graph.legend_properties]

    applied = apply_graph_patch(graph, {
        "xlogscale": True,
        "title_fontsize": 16,
        "legend_loc": "upper left",
        "legend_properties": styled,
    })

    assert set(applied) == {
        "xlogscale", "title_fontsize", "legend_loc", "legend_properties",
    }
    assert graph.ylabel == "Keep me"
    assert graph.x == "frequency"
    assert [item["linewidth"] for item in graph.legend_properties] == [3.0, 3.0]
    assert graph.legend_properties[0]["color"] == "red"


def test_explicit_null_clears_nullable_property_but_omission_preserves_it():
    graph = MGraph()
    graph.plot_title = "Old"
    graph.xmin = 2.0
    apply_graph_patch(graph, {"plot_title": None})
    assert graph.plot_title is None
    assert graph.xmin == 2.0


@pytest.mark.parametrize("patch, fragment", [
    ({"unknown_knob": 1}, "Unknown"),
    ({"xmin": 5, "xmax": 2}, "xmin"),
    ({"legend_alpha": 2}, "legend_alpha"),
    ({"axis_breaks": {
        "x": {"start": 1, "end": 2},
        "y": {"start": 3, "end": 4},
    }}, "Only one"),
    ({"inset_bounds": [0.9, 0.9, 0.5, 0.5]}, "fit"),
    ({"legend_properties": [{"label": "A", "line_width": 2}]}, "line_width"),
    ({"annotations": [{"type": "arrow", "x1": 0, "y1": 0}]}, "missing"),
])
def test_invalid_patch_is_rejected_without_mutating_graph(patch, fragment):
    graph = MGraph()
    before = graph.save()
    with pytest.raises(GraphValidationError, match=fragment):
        apply_graph_patch(graph, patch)
    assert graph.save() == before


def test_column_references_validate_against_target_dataframe():
    df = pd.DataFrame({"x": [1], "y": [2], "group": ["A"]})
    normalized = normalize_graph_patch(
        {"x": "x", "y": "y", "z": "group"}, dataframe=df,
    )
    assert normalized["y"] == ["y"]

    with pytest.raises(GraphValidationError, match="missing"):
        normalize_graph_patch({"x": "missing"}, dataframe=df)


def test_filters_are_normalized_once_at_shared_boundary():
    assert normalize_graph_patch({"filters": ["x > 1"]}) == {
        "filters": [{"expression": "x > 1", "state": True}],
    }


def test_series_identity_change_resets_only_derived_series_styles():
    graph = MGraph()
    graph.x = "x"
    graph.y = ["a"]
    graph.legend_properties = [{"label": "a", "color": "red"}]
    patch = normalize_graph_patch({"y": ["b"]}, current=graph)
    assert patch == {"y": ["b"], "legend_properties": []}
