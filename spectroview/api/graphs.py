"""Programmatic graph plotting mimicking the Graphs workspace.

Delegates to the same PlotRenderer the app's own VGraph widget uses (see
spectroview.view.components.v_plot_renderer) so scripted/notebook plots
genuinely match what the Graphs workspace draws -- same color/marker
cycling, same 95% CI computation, same box/point/trendline styling -- rather
than approximating it with a different library.

PlotRenderer has no Qt dependency (it only needs an object exposing the
right config attributes plus a matplotlib Axes), so a plain MGraph instance
stands in for the widget it normally reads configuration from; no
QApplication is required to use this module.

Previously this module wrapped seaborn, which was never actually able to
"perfectly replicate the native aesthetic" it claimed to (different CI
method, different color cycling, no shared code with VGraph at all) --
and `seaborn` was removed from this project's dependencies at some point
without this module being updated, so it silently raised
`ModuleNotFoundError` on import.
"""
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from spectroview import DEFAULT_COLORS, DEFAULT_MARKERS
from spectroview.model.m_graph import MGraph
from spectroview.view.components.v_plot_renderer import PlotRenderer


def _make_renderer(df: pd.DataFrame, x: str, y: str, hue: Optional[str],
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
    applied the same way for every plot style in this module."""
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
    """Create a SPECTROview-styled trendline (regression) plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    renderer, colors, _markers, c = _make_renderer(df, x, y, hue, ax, 'trendline')
    renderer.vg.trendline_order = order
    renderer._plot_trendline(df, y, colors, c)
    _finalize(ax, title)
    return ax
