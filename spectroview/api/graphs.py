"""Programmatic graph plotting mimicking the Graphs workspace.

Delegates to the same PlotRenderer the app's own VGraph widget uses (see
spectroview.view.components.v_plot_renderer) so scripted/notebook plots
genuinely match what the Graphs workspace draws -- same color/marker
cycling, same 95% CI computation, same box/point/trendline/wafer/2Dmap
styling -- rather than approximating it with a different library.

PlotRenderer has no Qt dependency (it only needs an object exposing the
right config attributes plus a matplotlib Axes), so a plain MGraph instance
stands in for the widget it normally reads configuration from; no
QApplication is required to use this module.

All 9 plot styles the Graphs workspace supports are covered here: point,
scatter, box, line, bar, trendline, histogram, wafer, 2Dmap. Plot recipes
(named, reusable sets of plot configs) are also exposed via
`list_plot_recipes`/`load_plot_recipe`/`save_plot_recipe`/
`delete_plot_recipe`.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

from spectroview import DEFAULT_COLORS, DEFAULT_MARKERS
from spectroview.api.exceptions import TemplateError
from spectroview.model.m_graph import MGraph
from spectroview.model.m_plot_recipe_store import MPlotRecipeStore
from spectroview.view.components.v_plot_renderer import PlotRenderer


def _make_renderer(df: pd.DataFrame, x: str, y: Optional[str], hue: Optional[str],
                    ax: plt.Axes, plot_style: str):
    """Build a PlotRenderer bound to a throwaway (non-Qt) MGraph config, and
    compute the (colors, markers, default_color) triplet the same way
    VGraph._plot_primary_axis does before delegating to PlotRenderer."""
    cfg = MGraph()
    cfg.x, cfg.y, cfg.z, cfg.ax, cfg.plot_style = x, [y], hue, ax, plot_style
    renderer = PlotRenderer(cfg)

    n_categories = df[hue].nunique() if hue and hue in df.columns else 0
    colors = DEFAULT_COLORS.copy()
    markers = DEFAULT_MARKERS.copy()
    if n_categories > 0:
        while len(colors) < n_categories:
            idx = len(colors)
            colors.append(DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
            markers.append(DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)])
        colors, markers = colors[:n_categories], markers[:n_categories]

    default_color = colors[0] if colors else 'steelblue'
    return renderer, colors, markers, default_color


def _finalize(ax: plt.Axes, title: Optional[str]) -> None:
    """Title, legend (only if there are labeled artists), and grid --
    applied the same way for every categorical/xy plot style in this module."""
    if title:
        ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.7)


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
                  title: Optional[str] = None, ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled scatter plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    renderer, colors, _markers, c = _make_renderer(df, x, y, hue, ax, 'scatter')
    renderer._plot_scatter(df, y, colors, c)
    _finalize(ax, title)
    return ax


def plot_point(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
               join: bool = False, dodge: bool = True, title: Optional[str] = None,
               ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled statistical point plot with error bars."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    renderer, colors, markers, c = _make_renderer(df, x, y, hue, ax, 'point')
    renderer.vg.join_for_point_plot = join
    renderer.vg.dodge_point_plot = dodge
    renderer._plot_point(df, y, colors, markers, c)
    _finalize(ax, title)
    return ax


def plot_box(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
             title: Optional[str] = None, ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled box plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    renderer, colors, _markers, c = _make_renderer(df, x, y, hue, ax, 'box')
    renderer._plot_box(df, y, colors, c)
    _finalize(ax, title)
    return ax


def plot_trendline(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
                    order: int = 1, title: Optional[str] = None, ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled trendline (polynomial regression) plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    renderer, colors, _markers, c = _make_renderer(df, x, y, hue, ax, 'trendline')
    renderer.vg.trendline_order = order
    renderer._plot_trendline(df, y, colors, c)
    _finalize(ax, title)
    return ax


def plot_line(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
              title: Optional[str] = None, ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled line plot (mean +/- 95% CI per x, like plot_point but joined by a line)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    renderer, colors, _markers, c = _make_renderer(df, x, y, hue, ax, 'line')
    renderer._plot_line(df, y, colors, c)
    _finalize(ax, title)
    return ax


def plot_bar(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
             show_error_bar: bool = False, title: Optional[str] = None,
             ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled bar plot (mean per x, optionally grouped by hue)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    renderer, colors, _markers, c = _make_renderer(df, x, y, hue, ax, 'bar')
    renderer.vg.show_bar_plot_error_bar = show_error_bar
    renderer._plot_bar(df, y, colors, c)
    _finalize(ax, title)
    return ax


def plot_histogram(df: pd.DataFrame, x: str, hue: Optional[str] = None, bins: int = 20,
                    kde: bool = False, title: Optional[str] = None, ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled histogram of column `x`, optionally split by `hue`."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    renderer, colors, _markers, _c = _make_renderer(df, x, None, hue, ax, 'histogram')
    renderer.vg.hist_bins = bins
    renderer.vg.hist_kde = kde
    renderer._plot_histogram(df, colors)
    _finalize(ax, title)
    return ax


def plot_2dmap(df: pd.DataFrame, x: str, y: str, z: str, title: Optional[str] = None,
               ax: Optional[plt.Axes] = None, cmap: str = 'jet',
               vmin: Optional[float] = None, vmax: Optional[float] = None):
    """Create a SPECTROview-styled 2D map heatmap from a tidy DataFrame with
    one row per (x, y) sample and a value column `z`."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    cfg = MGraph()
    cfg.x, cfg.z, cfg.ax, cfg.plot_style = x, z, ax, '2Dmap'
    cfg.color_palette, cfg.zmin, cfg.zmax = cmap, vmin, vmax
    PlotRenderer(cfg)._plot_2dmap(df, y)
    if title:
        ax.set_title(title)
    return ax


def plot_wafer(df: pd.DataFrame, x: str, y: str, z: str, title: Optional[str] = None,
               ax: Optional[plt.Axes] = None, cmap: str = 'jet',
               vmin: Optional[float] = None, vmax: Optional[float] = None,
               wafer_size: float = 300.0, show_stats: bool = True):
    """Create a SPECTROview-styled wafer map (interpolated heatmap over a
    circular wafer outline) from a tidy DataFrame with one row per (x, y) sample."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    cfg = MGraph()
    cfg.x, cfg.y, cfg.z, cfg.ax, cfg.plot_style = x, [y], z, ax, 'wafer'
    cfg.color_palette, cfg.zmin, cfg.zmax = cmap, vmin, vmax
    cfg.wafer_size, cfg.wafer_stats = wafer_size, show_stats
    PlotRenderer(cfg)._plot_wafer(df)
    if title:
        ax.set_title(title)
    return ax


# ── Plot recipes (named, reusable sets of plot configs) ────────────────────

def list_plot_recipes(folder: Union[str, Path]) -> List[Dict[str, Any]]:
    """List plot recipes saved in `folder`.

    Returns:
        List of {'id', 'name', 'created_at', 'graph_count'} summaries.
    """
    store = MPlotRecipeStore(str(folder))
    return [
        {"id": s.id, "name": s.name, "created_at": s.created_at, "graph_count": s.graph_count}
        for s in store.list_recipes()
    ]


def load_plot_recipe(folder: Union[str, Path], recipe_id: str) -> List[Dict[str, Any]]:
    """Load a plot recipe's list of plot-config dicts (each shaped like `MGraph.save()`).

    Raises:
        TemplateError: no recipe with that id exists in `folder`.
    """
    store = MPlotRecipeStore(str(folder))
    recipe = store.load_recipe(recipe_id)
    if recipe is None:
        raise TemplateError(f"No plot recipe with id '{recipe_id}' in {folder}.")
    return recipe.configs


def save_plot_recipe(folder: Union[str, Path], name: str, configs: List[Dict[str, Any]]) -> str:
    """Save a list of plot-config dicts as a named, reusable recipe.

    Args:
        folder: destination folder (created if missing).
        name: display name for the recipe.
        configs: non-empty list of plot-config dicts, each shaped like `MGraph.save()`.

    Returns:
        The new recipe's id.

    Raises:
        TemplateError: `configs` is empty.
    """
    store = MPlotRecipeStore(str(folder))
    recipe = store.save_recipe(name, configs)
    if recipe is None:
        raise TemplateError("configs must be a non-empty list.")
    return recipe.id


def delete_plot_recipe(folder: Union[str, Path], recipe_id: str) -> bool:
    """Delete a plot recipe by id. Returns True if a recipe was deleted."""
    store = MPlotRecipeStore(str(folder))
    return store.delete_recipe(recipe_id)
