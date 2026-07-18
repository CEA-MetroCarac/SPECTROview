"""Generic snapshot/diff helper for committing a VGraph widget's current
state back to its MGraph model.

Several call sites in v_workspace_graphs.py (the toolbar's grid checkbox,
save_workspace) need to sync "whatever changed on the widget" back to the
ViewModel. Historically each site hand-picked its own field list, which is
exactly how bugs like a missing annotations/filters sync happened: it's easy
to add a new widget-mutable field and forget to add it to every hand-picked
list. This module derives the field list once, from MGraph's own dataclass
schema, so there is nothing left to keep in sync by hand.
"""
import copy
import dataclasses

from spectroview.model.m_graph import MGraph

COMMIT_FIELDS = tuple(f.name for f in dataclasses.fields(MGraph) if f.name != 'graph_id')


def snapshot(gw) -> dict:
    """Deep-copy of every MGraph-schema field currently on a VGraph widget,
    taken before mutation, for later diffing against `diff()`. Deep-copy
    matters because several fields (legend_properties, annotations,
    axis_breaks) can be mutated in place rather than reassigned."""
    return {f: copy.deepcopy(getattr(gw, f, None)) for f in COMMIT_FIELDS}


def diff(gw, before: dict) -> dict:
    """Patch dict of exactly the fields that changed on `gw` since `before`
    was taken."""
    return {
        f: getattr(gw, f, None) for f in COMMIT_FIELDS
        if getattr(gw, f, None) != before.get(f)
    }
