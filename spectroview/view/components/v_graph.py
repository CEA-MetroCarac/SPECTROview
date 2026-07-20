"""Matplotlib graph visualization widget for MVVM pattern."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QDialog, QToolButton, QMenu, QFrame,
    QApplication,
)
from PySide6.QtCore import QObject, QEvent, QSize, Signal, QTimer, Qt
from PySide6.QtGui import QIcon, QAction

from spectroview import (
    DEFAULT_COLORS, DEFAULT_MARKERS, ICON_DIR,
    PLOT_POLICY_LIGHT, PLOT_POLICY_DARK, PLOT_POLICY_SOFT_DARK,
)

# figure_theme value -> mplstyle file. "light" (the default) reproduces the
# app's only-ever-existing hardcoded behavior; dark/soft_dark reuse the same
# theme files the Maps/Spectra viewers already switch between.
_THEME_STYLE_MAP = {
    "light": PLOT_POLICY_LIGHT,
    "dark": PLOT_POLICY_DARK,
    "soft_dark": PLOT_POLICY_SOFT_DARK,
}
from spectroview.view.components.customize_graph.customize_graph_dialog import (
    EditLineDialog, EditTextDialog, EditArrowDialog, EditSpanDialog,
    EditBoxDialog, EditCalloutDialog,
)
from spectroview.model.m_graph import MGraph
from spectroview.viewmodel.utils import show_alert, copy_fig_to_clb
from spectroview.view.components.v_plot_renderer import PlotRenderer


class VGraph(QWidget):
    """Graph widget rendering plots based on MGraph model properties."""
    properties_changed = Signal(int, dict)
    annotation_position_changed = Signal(int, str, float, float)
    replicate_requested = Signal(int)
    customize_requested = Signal(int)
    export_requested = Signal(int)
    export_all_requested = Signal()
    style_action_requested = Signal(int, str)
    notify = Signal(str)
    
    def __init__(self, graph_id=None):
        super().__init__()
        self.graph_id = graph_id

        # Model fields: seed every MGraph field with its dataclass default.
        # Single source of truth -- a hand-written copy here once silently
        # drifted from the model (minor_ticks_* were missing entirely).
        for key, value in vars(MGraph()).items():
            if key != 'graph_id':
                setattr(self, key, value)

        # Runtime-only state (not part of the model schema, never persisted)
        self.df = None  # DataFrame kept for replotting
        self.trendline_equations = []  # {label, equation, r2} per hue group
        self.ax_break_secondary = None  # second panel of a broken axis, if active
        self._current_break_mode = None  # None/'x'/'y' -- current Axes layout
        self._subtitle_artist = None  # Text artist _set_figure_style() draws the subtitle onto
        self.inset_ax = None  # the rendered inset Axes, if any

        # Matplotlib objects
        self.figure = None
        self.ax = None
        self.ax2 = None
        self.ax3 = None
        self.ax_x2 = None
        self.canvas = None

        # Layout setup
        self.graph_layout = QVBoxLayout()
        self.setLayout(self.graph_layout)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_layout.setSpacing(0)

    def clear_layout(self, layout):
        """Clears all widgets and layouts from the specified layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())
    
    def create_plot_widget(self, dpi):
        """Create the matplotlib figure/canvas and this graph's (hidden)
        toolbar_container. Must be called before plot()."""
        self.dpi = dpi or 100

        self.clear_layout(self.graph_layout)
        old_toolbar_container = getattr(self, 'toolbar_container', None)

        # OO Figure API (not plt.figure()): keeps this widget's figures out
        # of pyplot's global manager, so plt.close('all') elsewhere can't
        # silently close them.
        with plt.style.context(_THEME_STYLE_MAP.get(self.figure_theme, PLOT_POLICY_LIGHT)):
            self.figure = Figure(layout="compressed", dpi=self.dpi)
            self.ax = self.figure.add_subplot(111)
            
            # Reset broken axis state since this is a completely new figure
            self._current_break_mode = None
            self.ax_break_secondary = None
            
        self.canvas = FigureCanvas(self.figure)
        
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        original_set_style = self.toolbar.setStyleSheet
        def custom_set_style(css):
            original_set_style("QToolBar { background: transparent; border: none; }")
        self.toolbar.setStyleSheet = custom_set_style
        self.toolbar.setStyleSheet("")

        class ToolbarEventFilter(QObject):
            def __init__(self, toolbar):
                # Parented to `toolbar` so Qt cancels the pending single-shot
                # timer on teardown -- unparented, it could fire 100ms later
                # against an already-deleted toolbar (real segfault).
                super().__init__(toolbar)
                self.toolbar = toolbar
                self._timer = QTimer(self)
                self._timer.setSingleShot(True)
                self._timer.setInterval(100)
                self._timer.timeout.connect(self._update_icons)

            def eventFilter(self, obj, event):
                if event.type() == QEvent.PaletteChange:
                    self._timer.start()
                return False

            def _update_icons(self):
                try:
                    action_dict = {action.text(): action for action in self.toolbar.actions() if action.text()}
                except RuntimeError:
                    return  # toolbar's C++ object already deleted
                for text, tooltip_text, image_file, name_of_method in self.toolbar.toolitems:
                    if text in action_dict and image_file is not None:
                        try:
                            icon = self.toolbar._icon(image_file + '.png')
                            action_dict[text].setIcon(icon)
                        except Exception:
                            pass
                
        self.toolbar_filter = ToolbarEventFilter(self.toolbar)
        self.toolbar.installEventFilter(self.toolbar_filter)

        self.toolbar.setIconSize(QSize(30, 30))  # Set larger icon size
        for action in self.toolbar.actions():
            if action.text() in ['Save', 'Back', 'Forward', 'Subplots', 'Customize']:
                action.setVisible(False)
        
        # Create Replicate button
        self.btn_replicate = QPushButton()
        self.btn_replicate.setIcon(QIcon(f"{ICON_DIR}/duplicate.png"))
        self.btn_replicate.setIconSize(QSize(26, 26))
        self.btn_replicate.setFixedSize(30, 30)
        self.btn_replicate.setToolTip("Duplicate selected graph")
        self.btn_replicate.clicked.connect(lambda *args: self.replicate_requested.emit(self.graph_id))

        # Create Customize button -- colorful icon, same regardless of theme.
        self.btn_customize = QPushButton()
        self.btn_customize.setIcon(QIcon(f"{ICON_DIR}/customize2.png"))
        self.btn_customize.setIconSize(QSize(26, 26))
        self.btn_customize.setFixedSize(30, 30)
        self.btn_customize.setToolTip("Customize graph (Ctrl + E)")
        self.btn_customize.clicked.connect(lambda: self.customize_requested.emit(self.graph_id))

        # Create Copy button -- colorful icon, same regardless of theme.
        self.btn_copy_figure = QPushButton()
        self.btn_copy_figure.setIcon(QIcon(f"{ICON_DIR}/copy3.png"))
        self.btn_copy_figure.setIconSize(QSize(26, 26))
        self.btn_copy_figure.setFixedSize(30, 30)
        self.btn_copy_figure.setToolTip("Copy selected figure to clipboard (Ctrl +C)")
        self.btn_copy_figure.clicked.connect(self.copy_to_clipboard)

        # Create Export button -- colorful icon, same regardless of theme.
        # Click exports just this graph; Ctrl+Click exports every open graph.
        self.btn_export = QPushButton()
        self.btn_export.setIcon(QIcon(f"{ICON_DIR}/export.png"))
        self.btn_export.setIconSize(QSize(26, 26))
        self.btn_export.setFixedSize(30, 30)
        self.btn_export.setToolTip("Export selected graph to file. Hold Ctrl + Click to export all open graphs")
        self.btn_export.clicked.connect(self._on_export_clicked)

        # Style menu: every action just emits style_action_requested,
        # handled centrally by VWorkspaceGraphs.
        self.btn_style_menu = QToolButton()
        self.btn_style_menu.setFixedSize(30, 30)
        self.btn_style_menu.setIcon(QIcon(f"{ICON_DIR}/styling.png"))
        self.btn_style_menu.setIconSize(QSize(26, 26))
        self.btn_style_menu.setToolTip("Graph style: save/apply, copy/paste, reset, set default")
        self.btn_style_menu.setPopupMode(QToolButton.InstantPopup)
        style_menu = QMenu(self)
        for action_text, action_name in [
            ("Save Style...", "save_template"),
            ("Apply Style...", "apply_template"),
            ("Copy Style", "copy"),
            ("Paste Style", "paste"),
            ("Reset to Default", "reset"),
            ("Set as Default Style", "set_default"),
        ]:
            action = QAction(action_text, self)
            action.triggered.connect(
                lambda checked=False, name=action_name: self.style_action_requested.emit(self.graph_id, name)
            )
            style_menu.addAction(action)
        self.btn_style_menu.setMenu(style_menu)

        # Create toolbar layout with customize and copy buttons
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        toolbar_layout.addWidget(self.toolbar)

        # Vertical separator between the matplotlib nav buttons and our own
        # action buttons, which then get pushed flush against the right
        # edge of the (now workspace-wide, see VWorkspaceGraphs) toolbar.
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        toolbar_layout.addWidget(separator)

        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_replicate)
        toolbar_layout.addWidget(self.btn_customize)
        toolbar_layout.addWidget(self.btn_copy_figure)
        toolbar_layout.addWidget(self.btn_export)
        toolbar_layout.addWidget(self.btn_style_menu)
        
        # Create container widget for toolbar layout -- kept parentless and
        # hidden here; VWorkspaceGraphs.sync_active_graph_toolbar() reparents
        # it into the shared toolbar slot when this graph is the active one.
        toolbar_container = QWidget()
        toolbar_container.setLayout(toolbar_layout)
        toolbar_container.setFixedHeight(35)
        toolbar_container.hide()
        self.toolbar_container = toolbar_container
        if old_toolbar_container is not None:
            old_toolbar_container.deleteLater()

        self.graph_layout.addWidget(self.canvas)

        # Connect pick event for legend customization
        self.canvas.mpl_connect('pick_event', self._on_legend_pick)
        
        # Connect annotation drag events
        self.canvas.mpl_connect('motion_notify_event', self._on_annotation_drag)
        self.canvas.mpl_connect('button_release_event', self._on_annotation_release)
        self.canvas.mpl_connect('button_press_event', self._on_annotation_click)
        
        self.canvas.draw_idle()
    
    def copy_to_clipboard(self):
        """Copy the current figure to clipboard."""
        copy_fig_to_clb(self.canvas)

    def _on_export_clicked(self):
        """Export button: plain click exports just this graph; Ctrl+Click
        exports every open graph instead (mirrors VSpectraViewer's copy
        button pattern)."""
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            self.export_all_requested.emit()
        else:
            self.export_requested.emit(self.graph_id)

    def plot(self, df=None):
        """Wrapper to render plot using local style context."""
        with plt.style.context(_THEME_STYLE_MAP.get(self.figure_theme, PLOT_POLICY_LIGHT)):
            self._plot_internal(df)

    def _plot_internal(self, df):
        """Renders plot based on DataFrame and current properties."""
        self.df = df

        # Ensure scatter_edgecolor is always a valid string color, defaulting to 'black'
        edge_c = getattr(self, 'scatter_edgecolor', 'black')
        if not edge_c or not isinstance(edge_c, str) or edge_c.strip() in ("", "None", "none", "null"):
            self.scatter_edgecolor = 'black'

        self._setup_broken_axes()  # no-op unless the break mode changed since the last render

        if self._current_break_mode:
            self._plot_internal_with_break(df)
        else:
            self._plot_internal_normal(df)

        self.get_legend_properties()
        self.canvas.draw_idle()

    def _plot_internal_normal(self, df):
        """The original single-axes render pipeline -- completely unchanged
        by the broken-axis rewrite (see _plot_internal_with_break for that
        path); this is what runs for the overwhelming majority of graphs
        (no axis break active)."""
        self.ax.clear()
        if self.ax2:
            self.ax2.clear()
        if self.ax3:
            self.ax3.clear()
        if self.ax_x2:
            self.ax_x2.clear()

        if df is not None and self.df_name is not None and self.x is not None and self.y is not None:
            self._plot_primary_axis(df)
            self._plot_secondary_axis(df)
            self._plot_tertiary_axis(df)
            self._plot_secondary_x_axis(df)
        else:
            self.ax.plot([], [])

        self._set_limits()
        self._set_axis_scale(df)
        self._set_axis_direction()  # must run after _set_limits: it flips whatever the current limits are
        self._set_labels()
        self._set_figure_style()  # after _set_labels: the subtitle is positioned relative to the title
        self._set_grid()
        self._set_rotation()
        self._set_legend()
        self._render_annotations()  # Render annotations after all plot elements
        self._render_inset(df)  # Inset drawn last: needs self.ax's final view/transform

    def restyle(self) -> bool:
        """Fast preview path for a pure cosmetic change: re-run label/grid/
        limit/scale/legend styling on the existing artists without a full
        replot. Callers must check can_restyle_without_replot() first.
        Returns False (caller must fall back to plot()) when a broken axis
        is active, since that needs both panels kept in sync."""
        if self._current_break_mode is not None:
            return False

        with plt.style.context(_THEME_STYLE_MAP.get(self.figure_theme, PLOT_POLICY_LIGHT)):
            self._set_limits()
            self._set_axis_scale(self.df)
            self._set_axis_direction()
            self._set_labels()
            self._set_figure_style()
            self._set_grid()
            self._set_rotation()
            self._set_legend()

        self.canvas.draw_idle()
        return True

    def _setup_broken_axes(self):
        """Ensure self.ax (and self.ax_break_secondary) match the current
        axis_breaks state. Only rebuilds the layout when the break mode
        changed since the last render, so ordinary replots keep Axes
        identity/toolbar history stable. If both X and Y breaks are somehow
        set, X wins (simultaneous X+Y isn't supported)."""
        x_break = self.axis_breaks.get('x')
        y_break = self.axis_breaks.get('y') if not x_break else None
        new_mode = 'x' if x_break else ('y' if y_break else None)

        if new_mode == self._current_break_mode:
            return

        if self.ax is not None:
            try:
                self.ax.remove()
            except Exception:
                pass
        if self.ax_break_secondary is not None:
            try:
                self.ax_break_secondary.remove()
            except Exception:
                pass
            self.ax_break_secondary = None

        if new_mode is None:
            self.ax = self.figure.add_subplot(111)
        elif new_mode == 'x':
            gs = self.figure.add_gridspec(1, 2, wspace=0.005)
            self.ax = self.figure.add_subplot(gs[0, 0])
            self.ax_break_secondary = self.figure.add_subplot(gs[0, 1], sharey=self.ax)
        else:
            gs = self.figure.add_gridspec(2, 1, hspace=0.005)
            # self.ax stays the "primary" (bottom) panel so every existing
            # self.ax-based call site keeps working unchanged.
            self.ax_break_secondary = self.figure.add_subplot(gs[0, 0])
            self.ax = self.figure.add_subplot(gs[1, 0], sharex=self.ax_break_secondary)

        self._current_break_mode = new_mode

        engine = self.figure.get_layout_engine()
        if engine is not None:
            if new_mode is not None:
                # Tighten the layout engine's own padding so the small gridspec
                # gap isn't overridden and forced wider by the layout manager
                engine.set(w_pad=0.0, h_pad=0.0, wspace=0.005, hspace=0.005)
            else:
                # Restore standard compressed layout defaults
                engine.set(w_pad=0.04167, h_pad=0.04167, wspace=0.02, hspace=0.02)

    def _plot_internal_with_break(self, df):
        """Two-panel broken-axis render path (see _setup_broken_axes()).
        Mirrors the normal single-axes pipeline (same per-step methods on
        both panels via the self.ax swap in _render_series_on()), but uses
        per-panel range splitting instead of _set_limits(), and skips
        secondary/twin axes, zoom inset, and per-panel title/legend
        duplication (documented scope limits)."""
        primary = self.ax
        secondary = self.ax_break_secondary
        is_x_break = (self._current_break_mode == 'x')
        break_range = self.axis_breaks['x' if is_x_break else 'y']

        primary.clear()
        secondary.clear()

        # Twin axes aren't supported alongside a broken axis -- drop any
        # stale ones left over from a previous non-break render.
        for attr in ('ax2', 'ax3', 'ax_x2'):
            old = getattr(self, attr, None)
            if old is not None:
                try:
                    old.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        has_data = df is not None and self.df_name is not None and self.x is not None and self.y is not None

        # Primary panel: the exact normal single-axes pipeline.
        if has_data:
            self._plot_primary_axis(df)
        else:
            primary.plot([], [])
        self._set_axis_scale(df)
        self._set_labels()
        if not is_x_break:
            # Y-break: primary (self.ax) is the bottom panel, so
            # _set_labels()'s title lands in the panel gap -- move it up.
            title_text = primary.get_title()
            title_fontsize = primary.title.get_fontsize()
            if title_text:
                primary.set_title('')
                secondary.set_title(title_text, fontsize=title_fontsize)
                
            # Center the ylabel across both panels (y=1.0025 is the gap between bottom and top)
            primary.yaxis.label.set_y(1.0025)
            primary.yaxis.label.set_va('center')
        self._set_figure_style()
        
        if is_x_break:
            # Center title, subtitle, and xlabel across both panels 
            # (x=1.0025 is the gap between the left and right panels)
            primary.title.set_x(1.0025)
            primary.title.set_ha('center')
            
            primary.xaxis.label.set_x(1.0025)
            primary.xaxis.label.set_ha('center')
            
            if getattr(self, '_subtitle_artist', None):
                self._subtitle_artist.set_x(1.0025)
                self._subtitle_artist.set_ha('center')
        self._set_grid()
        self._set_legend()

        # Secondary panel: same series, unclipped (clipping to its half of
        # the range happens in _apply_break_split below). Title/subtitle/
        # legend stay primary-only -- this is a continuation, not a second axis.
        if has_data:
            self._render_series_on(secondary)
        else:
            secondary.plot([], [])

        original_ax = self.ax
        self.ax = secondary
        try:
            self._set_axis_scale(df)
            self._set_grid()
        finally:
            self.ax = original_ax
        secondary.set_facecolor(primary.get_facecolor())
        for side, visible in (self.spines_visible or {}).items():
            if side in secondary.spines:
                secondary.spines[side].set_visible(visible)
        if self.figure_margins:
            x_margin, y_margin = self.figure_margins
            secondary.margins(x=x_margin, y=y_margin)

        self._apply_break_split(primary, secondary, is_x_break, break_range)
        
        self._set_rotation()
        original_ax = self.ax
        self.ax = secondary
        try:
            self._set_rotation()
        finally:
            self.ax = original_ax

        self._render_annotations()  # primary panel's copy (secondary's was drawn by _render_series_on above)

    def _apply_break_split(self, primary, secondary, is_x_break, break_range):
        """Clip each panel to its half of the broken range, hide the facing
        spines, and draw the diagonal 'd' marks on the real spine
        boundaries -- the standard brokenaxes-style technique, replacing
        the old post-hoc artist mutation that didn't work for bar/box/
        wafer/2Dmap."""
        start, end = break_range['start'], break_range['end']

        if is_x_break:
            lo, hi = primary.get_xlim()
        else:
            lo, hi = primary.get_ylim()

        clipped_start = min(max(start, lo), hi)
        clipped_end = min(max(end, lo), hi)

        # A break clamped to zero width (e.g. stale range after switching
        # columns) would hand matplotlib a singular xlim/ylim -- treat it as
        # "no break" instead: both panels show the full range, unsplit.
        if clipped_start <= lo or clipped_end >= hi or clipped_start >= clipped_end:
            if is_x_break:
                primary.set_xlim(lo, hi)
                secondary.set_xlim(lo, hi)
            else:
                primary.set_ylim(lo, hi)
                secondary.set_ylim(lo, hi)
            return

        if is_x_break:
            primary.set_xlim(lo, clipped_start)
            secondary.set_xlim(clipped_end, hi)
            primary.spines['right'].set_visible(False)
            secondary.spines['left'].set_visible(False)
            secondary.tick_params(axis='y', which='both', left=False, labelleft=False)
        else:
            primary.set_ylim(lo, clipped_start)
            secondary.set_ylim(clipped_end, hi)
            primary.spines['top'].set_visible(False)
            secondary.spines['bottom'].set_visible(False)
            secondary.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        self._draw_break_marks(primary, secondary, is_x_break)

    def _draw_break_marks(self, primary, secondary, is_x_break):
        """Small diagonal marks on the two facing spines signaling "the
        axis is broken here" -- the standard matplotlib broken-axis
        technique (matplotlib's own gallery example: broken_axis.html),
        now drawn on real spine boundaries instead of mutated data, so it
        renders identically regardless of plot style."""
        kwargs = dict(
            marker=[(-1, -1), (1, 1)], markersize=5, linestyle="none",
            color='k', mec='k', mew=1, clip_on=False,
        )
        dashed_kwargs = dict(
            color='k', linestyle='--', linewidth=0.5, alpha=0.8, clip_on=False
        )
        if is_x_break:
            primary.plot([1, 1], [0, 1], transform=primary.transAxes, **dashed_kwargs)
            secondary.plot([0, 0], [0, 1], transform=secondary.transAxes, **dashed_kwargs)
            primary.plot([1], [0], transform=primary.transAxes, **kwargs)
            primary.plot([1], [1], transform=primary.transAxes, **kwargs)
            secondary.plot([0], [0], transform=secondary.transAxes, **kwargs)
            secondary.plot([0], [1], transform=secondary.transAxes, **kwargs)
        else:
            primary.plot([0, 1], [1, 1], transform=primary.transAxes, **dashed_kwargs)
            secondary.plot([0, 1], [0, 0], transform=secondary.transAxes, **dashed_kwargs)
            primary.plot([0], [1], transform=primary.transAxes, **kwargs)
            primary.plot([1], [1], transform=primary.transAxes, **kwargs)
            secondary.plot([0], [0], transform=secondary.transAxes, **kwargs)
            secondary.plot([1], [0], transform=secondary.transAxes, **kwargs)

    def _render_series_on(self, target_ax):
        """Redraw the primary series + annotations onto `target_ax` instead
        of `self.ax`: temporarily repoints self.ax and self.figure there,
        replays the render calls, then restores both. Does not replay
        limits/legend/twin-axes/labels -- callers set target_ax's own
        limits afterward. Shared by the inset renderer, broken-axis
        two-panel renderer, and the multi-panel composer."""
        original_ax = self.ax
        original_figure = self.figure
        self.ax = target_ax
        self.figure = target_ax.figure
        try:
            if self.df is not None and self.df_name is not None and self.x is not None and self.y is not None:
                self._plot_primary_axis(self.df)
            self._render_annotations()
        finally:
            self.ax = original_ax
            self.figure = original_figure

    def _render_inset(self, df):
        """Draw the optional zoom-inset axes: the same primary series +
        annotations as the main plot, at its own x/y limits. Skipped when a
        broken axis is active (out of scope -- which panel would it attach to?)."""
        if self.inset_ax is not None:
            try:
                self.inset_ax.remove()
            except Exception:
                pass
            self.inset_ax = None

        if not getattr(self, 'inset_enabled', False):
            return
        if self.axis_breaks.get('x') or self.axis_breaks.get('y'):
            return

        try:
            inset_ax = self.ax.inset_axes(self.inset_bounds)
            self.inset_ax = inset_ax
            self._render_series_on(inset_ax)

            if self.inset_xmin is not None or self.inset_xmax is not None:
                xlo, xhi = inset_ax.get_xlim()
                inset_ax.set_xlim(
                    self.inset_xmin if self.inset_xmin is not None else xlo,
                    self.inset_xmax if self.inset_xmax is not None else xhi,
                )
            if self.inset_ymin is not None or self.inset_ymax is not None:
                ylo, yhi = inset_ax.get_ylim()
                inset_ax.set_ylim(
                    self.inset_ymin if self.inset_ymin is not None else ylo,
                    self.inset_ymax if self.inset_ymax is not None else yhi,
                )

            if self.inset_show_zoom_indicator:
                self.ax.indicate_inset_zoom(inset_ax, edgecolor="gray")
        except Exception as e:
            print(f"[WARNING] Failed to render inset axes: {e}")

    def get_legend_properties(self):
        """Retrieves properties of each legend within legend box."""
        if hasattr(self, 'legend_properties') and self.legend_properties:
            return self.legend_properties
        
        import matplotlib.colors as mcolors
        legend_properties = []
        if self.ax:
            legend = self.ax.get_legend()
            if legend:
                legend_texts = legend.get_texts()
                # legend_handles holds the actual drawn artists — read their true colors
                legend_handles = getattr(legend, 'legend_handles', [])
                for idx, text in enumerate(legend_texts):
                    label = text.get_text()
                    color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                    marker = DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)] if DEFAULT_MARKERS else 'o'

                    # Read true color from the matplotlib handle
                    if idx < len(legend_handles):
                        handle = legend_handles[idx]
                        try:
                            c = None
                            if hasattr(handle, 'get_color'):
                                c = handle.get_color()
                            elif hasattr(handle, 'get_facecolor'):
                                fc = handle.get_facecolor()
                                if hasattr(fc, '__len__') and len(fc) > 0:
                                    c = fc[0]
                                    
                            if c is not None:
                                hex_c = None
                                if isinstance(c, str) and c not in ('none', '', 'None'):
                                    if c.startswith('#'):
                                        hex_c = mcolors.to_hex(c)
                                    else:
                                        color = c
                                elif not isinstance(c, str):
                                    hex_c = mcolors.to_hex(c)
                                    
                                if hex_c is not None:
                                    matched = False
                                    for dc in DEFAULT_COLORS:
                                        if mcolors.to_hex(dc).lower() == hex_c.lower():
                                            color = dc
                                            matched = True
                                            break
                                    if not matched:
                                        color = hex_c
                        except Exception:
                            pass
                        try:
                            if hasattr(handle, 'get_marker'):
                                m = handle.get_marker()
                                if m and str(m) not in ('None', 'none', ''):
                                    marker = str(m)
                        except Exception:
                            pass

                    rgba_color = list(mcolors.to_rgba(color))
                    legend_properties.append({
                        'label': label,
                        'marker': marker,
                        'color': color,
                        'rgba': rgba_color
                    })
            elif self.plot_style not in ['2Dmap', 'wafer']:
                color = DEFAULT_COLORS[0] if DEFAULT_COLORS else 'steelblue'
                rgba_color = list(mcolors.to_rgba(color))
                legend_properties.append({
                    'label': 'All data',
                    'marker': 'o',
                    'color': color,
                    'rgba': rgba_color
                })
        
        self.legend_properties = legend_properties
        return self.legend_properties
    
    
    def _on_legend_pick(self, event):
        """Handle pick event — legend double-click.

        Annotation drag-candidate detection used to also happen here, but
        moved to _on_annotation_click's manual hit-test: matplotlib's
        pick_event only reaches an artist when event.inaxes equals that
        artist's own Axes (see _ax_data_coords), which silently never
        happens for annotations on self.ax whenever a secondary Y-axis is
        also present -- it fully overlaps self.ax and has higher z-order.
        """
        artist = event.artist
        if hasattr(artist, '_annotation_data'):
            return

        # Check if legend was clicked
        if artist.get_label() == '_legend_':
            return
        
        # Check if it's a double-click on the legend
        legend = self.ax.get_legend()
        if legend and artist == legend and event.mouseevent.dblclick:
            # Emit signal to let workspace handle the dialog
            self.customize_requested.emit(self.graph_id)
    
    def _plot_primary_axis(self, df):
        """Plots data on the primary axis based on the current plot style."""
        self.renderer = PlotRenderer(self)
        # Determine number of hue categories
        n_categories = df[self.z].nunique() if self.z and self.z in df.columns else 0
        
        # Reset legend_properties if number of categories changed
        if self.legend_properties and n_categories > 0 and len(self.legend_properties) != n_categories:
            self.legend_properties = []
        
        if not self.legend_properties:
            markers = DEFAULT_MARKERS.copy()
            colors = DEFAULT_COLORS.copy()
        else:
            markers = [str(prop['marker']) for prop in self.legend_properties]
            colors = [str(prop['color']) for prop in self.legend_properties]
        
        # Extend or trim colors/markers to match actual number of hue categories
        if n_categories > 0:
            # Extend by cycling through DEFAULT_COLORS/DEFAULT_MARKERS if needed
            while len(colors) < n_categories:
                idx = len(colors)
                colors.append(DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
                markers.append(DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)])
            # Trim if too many
            colors = colors[:n_categories]
            markers = markers[:n_categories]
        
        for y in self.y:
            c = colors[0] if colors else 'steelblue'
            
            if self.plot_style == 'point':
                self.renderer._plot_point(df, y, colors, markers, c)
            elif self.plot_style == 'scatter':
                self.renderer._plot_scatter(df, y, colors, c)
            elif self.plot_style == 'box':
                self.renderer._plot_box(df, y, colors, c)
            elif self.plot_style == 'line':
                self.renderer._plot_line(df, y, colors, c)
            elif self.plot_style == 'bar':
                self.renderer._plot_bar(df, y, colors, c)
            elif self.plot_style == 'trendline':
                self.renderer._plot_trendline(df, y, colors, c)
            elif self.plot_style == 'histogram':
                self.renderer._plot_histogram(df, colors)
            elif self.plot_style == 'wafer':
                self.renderer._plot_wafer(df)
            elif self.plot_style == '2Dmap':
                self.renderer._plot_2dmap(df, y)
            else:
                show_alert("Unsupported plot style")

    
    def _set_legend(self):
        """Sets up and displays the legend for the plot."""
        handles, labels = self.ax.get_legend_handles_labels()
        
        if self.ax2:
            handles2, labels2 = self.ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
            if self.ax2.get_legend():
                self.ax2.get_legend().remove()
        
        if self.ax3:
            handles3, labels3 = self.ax3.get_legend_handles_labels()
            handles += handles3
            labels += labels3
            if self.ax3.get_legend():
                self.ax3.get_legend().remove()
        
        if self.ax_x2:
            handles_x2, labels_x2 = self.ax_x2.get_legend_handles_labels()
            handles += handles_x2
            labels += labels_x2
            if self.ax_x2.get_legend():
                self.ax_x2.get_legend().remove()
        
        if handles:
                
            if self.legend_visible:
                unique_labels = []
                unique_handles = []
                for h, l in zip(handles, labels):
                    if l not in unique_labels:
                        unique_labels.append(l)
                        unique_handles.append(h)
                        
                # Apply custom labels from legend_properties
                if getattr(self, 'legend_properties', []):
                    custom_labels = []
                    for i, l in enumerate(unique_labels):
                        if i < len(self.legend_properties):
                            custom_labels.append(self.legend_properties[i].get('label', l))
                        else:
                            custom_labels.append(l)
                    unique_labels = custom_labels
                        
                # Shared legend-box styling (ncol/frame/title/fontsize/alpha)
                # applies whether the legend sits inside or outside the axes;
                # only its *position* differs between the two branches below.
                style_kwargs = {
                    'ncol': getattr(self, 'legend_ncol', 1),
                    'frameon': getattr(self, 'legend_frame', True),
                    'framealpha': getattr(self, 'legend_alpha', 0.7),
                }
                if getattr(self, 'legend_title', None):
                    style_kwargs['title'] = self.legend_title
                style_kwargs['fontsize'] = getattr(self, 'legend_fontsize', None) or 10

                if getattr(self, 'legend_outside', False):
                    legend = self.ax.legend(
                        unique_handles, unique_labels,
                        loc='center left', bbox_to_anchor=(1.02, 0.5),
                        **style_kwargs
                    )
                else:
                    legend = self.ax.legend(
                        unique_handles, unique_labels,
                        loc=getattr(self, 'legend_loc', 'best'),
                        **style_kwargs
                    )
                    if getattr(self, 'legend_bbox', None) is not None:
                        legend._loc = tuple(self.legend_bbox)
                        
                legend.set_picker(True)
                legend.set_draggable(True)
            else:
                if self.ax.get_legend():
                    self.ax.get_legend().remove()
    
    def _save_legend_position(self):
        """Save current legend position in axes coordinates."""
        legend = self.ax.get_legend()
        if legend is not None:
            loc = legend._loc
            # After dragging, _loc becomes a tuple (x, y) in axes coords
            if isinstance(loc, tuple):
                self.legend_bbox = [float(loc[0]), float(loc[1])]
    
    def _set_grid(self):
        """Add grid for the plot (supports linear & log scale automatically)."""
        is_wafer = (self.plot_style == 'wafer')
        self.ax.set_axisbelow(True)

        # tick_direction only overrides matplotlib's default when set.
        extra_ticks = {'labelsize': getattr(self, 'tick_label_fontsize', None) or 9}
        if getattr(self, 'tick_direction', None):
            extra_ticks['direction'] = self.tick_direction

        self.ax.tick_params(
            which='major',
            bottom=False if is_wafer else True,
            left=True,
            top=False if is_wafer else getattr(self, 'minor_ticks_top', False),
            right=getattr(self, 'minor_ticks_right', False),
            **extra_ticks
        )

        if any([getattr(self, 'minor_ticks_bottom', True),
                getattr(self, 'minor_ticks_left', True),
                getattr(self, 'minor_ticks_top', False),
                getattr(self, 'minor_ticks_right', False)]):
            self.ax.minorticks_on()
            self.ax.tick_params(
                which='minor',
                bottom=False if is_wafer else getattr(self, 'minor_ticks_bottom', True),
                top=False if is_wafer else getattr(self, 'minor_ticks_top', False),
                left=getattr(self, 'minor_ticks_left', True),
                right=getattr(self, 'minor_ticks_right', False),
                **({'direction': self.tick_direction} if getattr(self, 'tick_direction', None) else {})
            )
        else:
            self.ax.minorticks_off()

        if not self.grid:
            self.ax.grid(False)
            return
        
        self.ax.grid(True, which='major', linestyle='--')
        
        if self.xlogscale or self.ylogscale:
            self.ax.grid(True, which='minor', alpha=0.15, linestyle='--')
    
    def _set_rotation(self):
        """Set rotation of the ticklabels of the x axis."""
        if self.x_rot != 0:
            rotation, ha, rotation_mode = self.x_rot, "right", "anchor"
        else:
            # Reset to default when rotation is 0
            rotation, ha, rotation_mode = 0, "center", None

        for label in self.ax.get_xticklabels():
            label.set_rotation(rotation)
            label.set_ha(ha)
            label.set_rotation_mode(rotation_mode)
    
    
    def _apply_limit_pair(self, setter, vmin, vmax, axis_name: str) -> None:
        """Apply a (min, max) limit pair via *setter* (e.g. ax.set_xlim),
        skipping degenerate equal bounds instead of handing matplotlib a
        zero-width range (which triggers a "singular transformation"
        UserWarning and silently auto-expands anyway). Uses explicit
        `is not None` checks (not truthy checks) so a limit of exactly 0.0
        -- a common, legitimate axis bound -- is not silently ignored.
        """
        if vmin is None or vmax is None:
            return
        vmin, vmax = float(vmin), float(vmax)
        if vmin == vmax:
            self.notify.emit(f"Skipping {axis_name} limits: min == max ({vmin}).")
            return
        setter(vmin, vmax)

    def _set_limits(self):
        """Set the limits of axis."""
        self._apply_limit_pair(self.ax.set_xlim, self.xmin, self.xmax, "x-axis")
        self._apply_limit_pair(self.ax.set_ylim, self.ymin, self.ymax, "y-axis")
        if self.ax2:
            self._apply_limit_pair(self.ax2.set_ylim, self.y2min, self.y2max, "y2-axis")
        if self.ax3:
            self._apply_limit_pair(self.ax3.set_ylim, self.y3min, self.y3max, "y3-axis")
        if self.ax_x2:
            self._apply_limit_pair(self.ax_x2.set_xlim, self.x2min, self.x2max, "x2-axis")
    
    def _set_axis_scale(self, df):
        """Apply log/symlog scale only if the corresponding axis column is numeric."""
        if df is None:
            return

        if self.xlogscale:
            x_data = df[self.x]
            if np.issubdtype(x_data.dtype, np.number):
                self.ax.set_xscale(self.xscale_mode if self.xscale_mode == 'symlog' else 'log')
            else:
                self.notify.emit(f"Skipping x-logscale because '{self.x}' is categorical.")

        if self.ylogscale and len(self.y) > 0:
            y_data = df[self.y[0]]
            if np.issubdtype(y_data.dtype, np.number):
                self.ax.set_yscale(self.yscale_mode if self.yscale_mode == 'symlog' else 'log')
            else:
                self.notify.emit(f"Skipping y-logscale because '{self.y[0]}' is categorical.")

        if self.ax2 and self.y2 and self.y2logscale:
            y2_data = df[self.y2]
            if np.issubdtype(y2_data.dtype, np.number):
                self.ax2.set_yscale('log')

        if self.ax3 and self.y3 and self.y3logscale:
            y3_data = df[self.y3]
            if np.issubdtype(y3_data.dtype, np.number):
                self.ax3.set_yscale('log')

        if self.ax_x2 and self.x2 and self.x2logscale:
            x2_data = df[self.x2]
            if np.issubdtype(x2_data.dtype, np.number):
                self.ax_x2.set_xscale('log')

        if self.tick_label_format:
            formatter = FormatStrFormatter(self.tick_label_format)
            self.ax.xaxis.set_major_formatter(formatter)
            self.ax.yaxis.set_major_formatter(formatter)

    def _set_axis_direction(self):
        """Invert X/Y axes if requested. Must run after _set_limits(), since
        it flips whatever the current limits happen to be.

        invert_xaxis()/invert_yaxis() *toggle* rather than set, so compare
        against xaxis_inverted()/yaxis_inverted() first -- restyle() calls
        this repeatedly without an ax.clear() in between, where an
        unconditional toggle would flip-flop the axis on every other call."""
        if self.x_inverted != self.ax.xaxis_inverted():
            self.ax.invert_xaxis()
        if self.y_inverted != self.ax.yaxis_inverted():
            self.ax.invert_yaxis()

    def _format_axis_label(self, col_name) -> str:
        """Format axis label based on specific parameter rules.

        E.g. "x0_Si" -> "Si peak position (cm$^{-1}$)". Column names of the
        form "<param>_<peaklabel>" (fit-result columns) are turned into a
        friendly, unit-annotated label instead of shown raw.
        """
        if not col_name or not isinstance(col_name, str):
            return str(col_name) if col_name is not None else ""

        parts = col_name.split('_', 1)
        if len(parts) == 2:
            param, peaklabel = parts
            param = param.lower()
            if param == "x0":
                return f"{peaklabel} peak position (cm$^{{-1}}$)"
            elif param == "fwhm":
                return f"{peaklabel} peak width (cm$^{{-1}}$)"
            elif param == "ampli":
                return f"{peaklabel} peak intensity (a.u.)"
            elif param == "area":
                return f"{peaklabel} peak area (a.u.)"

        return col_name

    def _get_y_label_default(self, y_col):
        if isinstance(y_col, list) and len(y_col) > 0:
            return self._format_axis_label(y_col[0])
        return self._format_axis_label(y_col)

    def _set_labels(self):
        """Set titles and labels for axis and plot."""
        if self.plot_style == 'wafer':
            if self.plot_title:
                self.ax.set_title(self.plot_title)
            else:
                self.ax.set_title(self._format_axis_label(self.z) if self.z else "")
            self.ax.tick_params(axis='x', labelbottom=False)
        elif self.plot_style == '2Dmap':
            if self.plot_title:
                self.ax.set_title(self.plot_title)
            else:
                self.ax.set_title(self._format_axis_label(self.z) if self.z else "")
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            else:
                self.ax.set_xlabel(self._format_axis_label(self.x))
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            else:
                self.ax.set_ylabel(self._get_y_label_default(self.y))
        else:
            self.ax.set_title(self.plot_title)
            if self.xlabel:
                self.ax.set_xlabel(self.xlabel)
            else:
                self.ax.set_xlabel(self._format_axis_label(self.x))
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            else:
                if self.plot_style == 'histogram':
                    self.ax.set_ylabel("Frequency")
                else:
                    self.ax.set_ylabel(self._get_y_label_default(self.y))

        # Applied once here rather than threading fontsize= into every
        # set_title/set_xlabel/set_ylabel call above.
        self.ax.title.set_fontsize(self.title_fontsize or 12)
        axis_label_fontsize = self.axis_label_fontsize or 12
        self.ax.xaxis.label.set_fontsize(axis_label_fontsize)
        self.ax.yaxis.label.set_fontsize(axis_label_fontsize)

    def _set_figure_style(self):
        """Figure/axes-wide styling: background color, subtitle, spine
        visibility, margins. `ax.clear()` doesn't retroactively repaint an
        Axes built under a different theme, so the current theme's
        facecolor is re-applied here explicitly, with figure_facecolor (an
        explicit user override) layered on top."""
        self.ax.set_facecolor(plt.rcParams['axes.facecolor'])
        self.figure.set_facecolor(plt.rcParams['figure.facecolor'])
        if self.figure_facecolor:
            self.ax.set_facecolor(self.figure_facecolor)
            self.figure.set_facecolor(self.figure_facecolor)

        # Same gap as facecolor above, for text/spine color -- without this,
        # switching to Dark/Soft Dark left labels/title/ticks/spines black.
        self.ax.xaxis.label.set_color(plt.rcParams['axes.labelcolor'])
        self.ax.yaxis.label.set_color(plt.rcParams['axes.labelcolor'])
        self.ax.title.set_color(plt.rcParams['text.color'])
        self.ax.tick_params(axis='x', colors=plt.rcParams['xtick.color'])
        self.ax.tick_params(axis='y', colors=plt.rcParams['ytick.color'])
        for spine in self.ax.spines.values():
            spine.set_edgecolor(plt.rcParams['axes.edgecolor'])

        # Detect a stale artist from a since-replaced Axes (e.g. after
        # _setup_broken_axes() rebuilds the layout) via ax.texts membership.
        if self._subtitle_artist is not None and self._subtitle_artist not in self.ax.texts:
            self._subtitle_artist = None

        if self.plot_subtitle:
            # Sits right at the axes' top edge -- the title itself floats
            # further up above this, via its own default padding, so the
            # two don't collide without needing precise pad arithmetic.
            fontsize = self.subtitle_fontsize if self.subtitle_fontsize is not None else 10
            if self._subtitle_artist is not None:
                # Update the existing artist in place -- calling ax.text()
                # again on every restyle() tick (Phase 5E's no-clear fast
                # path) would otherwise stack a new copy on top each time.
                self._subtitle_artist.set_text(self.plot_subtitle)
                self._subtitle_artist.set_fontsize(fontsize)
            else:
                self._subtitle_artist = self.ax.text(
                    0.5, 1.0, self.plot_subtitle,
                    transform=self.ax.transAxes, ha='center', va='bottom',
                    fontsize=fontsize,
                )
        elif self._subtitle_artist is not None:
            self._subtitle_artist.remove()
            self._subtitle_artist = None

        spines = self.spines_visible or {}
        for side, visible in spines.items():
            if side in self.ax.spines:
                self.ax.spines[side].set_visible(visible)

        if self.figure_margins:
            x_margin, y_margin = self.figure_margins
            self.ax.margins(x=x_margin, y=y_margin)

    def _draw_twin_series(self, twin_ax, x_vals, y_vals, *, color, marker, series_label):
        """Draw one series onto a twin axis, dispatching by the primary
        plot_style (point/line/scatter share this look across y2/y3/x2).

        Returns True if drawn, False if plot_style isn't supported on a twin
        axis — the caller is responsible for removing the (now-empty) axis
        in that case, matching each twin axis's prior standalone behavior.
        """
        if self.plot_style == 'line':
            twin_ax.plot(x_vals, y_vals, color=color, label=series_label)
        elif self.plot_style == 'point':
            twin_ax.errorbar(
                x_vals, y_vals,
                fmt=marker, color=color, markeredgecolor='black',
                markeredgewidth=1, capsize=3,
                linestyle='-' if self.join_for_point_plot else 'none',
                label=series_label
            )
        elif self.plot_style == 'scatter':
            twin_ax.scatter(
                x_vals, y_vals,
                s=self.scatter_size, edgecolors=self.scatter_edgecolor,
                color=color, label=series_label
            )
        else:
            return False
        return True

    def _plot_secondary_axis(self, df):
        """Plot data on secondary y-axis (y2, twinx, red)."""
        if self.ax2:
            self.ax2.remove()
            self.ax2 = None

        if not self.y2:
            return

        self.ax2 = self.ax.twinx()

        # For simplicity with secondary axes, just use numeric spacing if x is numeric,
        # otherwise just plot directly (matplotlib will handle it if X is strings)
        x_vals = df[self.x]
        if getattr(self, 'x_as_numeric', False):
            x_vals = pd.to_numeric(x_vals, errors='coerce')

        if self._draw_twin_series(self.ax2, x_vals, df[self.y2], color=self.y2color, marker=self.y2marker, series_label=self.y2):
            self.ax2.set_ylabel(self.y2label or self._format_axis_label(self.y2), color=self.y2color)
            self.ax2.tick_params(axis='y', colors=self.y2color)
        else:
            self.ax2.remove()
            self.ax2 = None

    def _plot_tertiary_axis(self, df):
        """Plot data on tertiary y-axis (y3, twinx offset outward, green)."""
        if self.ax3:
            self.ax3.remove()
            self.ax3 = None

        if not self.y3:
            return

        self.ax3 = self.ax.twinx()
        self.ax3.spines["right"].set_position(("outward", 100))

        x_vals = df[self.x]
        if getattr(self, 'x_as_numeric', False):
            x_vals = pd.to_numeric(x_vals, errors='coerce')

        if self._draw_twin_series(self.ax3, x_vals, df[self.y3], color=self.y3color, marker=self.y3marker, series_label=self.y3):
            self.ax3.set_ylabel(self.y3label or self._format_axis_label(self.y3), color=self.y3color)
            self.ax3.tick_params(axis='y', colors=self.y3color)
        else:
            self.ax3.remove()
            self.ax3 = None

    def _plot_secondary_x_axis(self, df):
        """Plot data on secondary x-axis (x2, twiny, purple)."""
        if self.ax_x2:
            self.ax_x2.remove()
            self.ax_x2 = None

        if not (self.x2 and self.x2 in df.columns):
            return

        self.ax_x2 = self.ax.twiny()

        x_vals = df[self.x2]
        if getattr(self, 'x2logscale', False):
            x_vals = pd.to_numeric(x_vals, errors='coerce')

        if self._draw_twin_series(self.ax_x2, x_vals, df[self.y[0]], color=self.x2color, marker=self.x2marker, series_label=self.y[0]):
            self.ax_x2.set_xlabel(
                self.x2label or self._format_axis_label(self.x2), color=self.x2color
            )
            self.ax_x2.tick_params(axis='x', colors=self.x2color)
        else:
            self.ax_x2.remove()
            self.ax_x2 = None
    
    # ═══════════════════════════════════════════════════════════════════
    # Annotation Rendering
    # ═══════════════════════════════════════════════════════════════════
    
    def _render_annotations(self):
        """Render all annotations (lines and text) on the plot."""
        if not self.annotations:
            return
        
        for ann in self.annotations:
            ann_type = ann.get('type')
            try:
                if ann_type == 'vline':
                    self._render_vline(ann)
                elif ann_type == 'hline':
                    self._render_hline(ann)
                elif ann_type == 'text':
                    self._render_text(ann)
                elif ann_type == 'arrow':
                    self._render_arrow(ann)
                elif ann_type in ('vspan', 'hspan'):
                    self._render_span(ann)
                elif ann_type == 'box':
                    self._render_box(ann)
                elif ann_type == 'callout':
                    self._render_callout(ann)
            except Exception as e:
                print(f"[WARNING] Failed to render annotation {ann.get('id')}: {e}")
    
    def _render_vline(self, ann: dict):
        """Render vertical line annotation."""
        x_pos = ann.get('x', 0)
        color = ann.get('color', 'red')
        linestyle = ann.get('linestyle', '--')
        linewidth = ann.get('linewidth', 1.5)
        label = ann.get('label', None)
        
        line = self.ax.axvline(
            x=x_pos,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=100,  # Render on top of other elements
            picker=5  # Enable picking with 5pt tolerance
        )
        
        # Attach metadata for drag handling
        line._annotation_data = ann
        line._is_dragging = False
        return line
    
    def _render_hline(self, ann: dict):
        """Render horizontal line annotation."""
        y_pos = ann.get('y', 0)
        color = ann.get('color', 'blue')
        linestyle = ann.get('linestyle', '--')
        linewidth = ann.get('linewidth', 1.5)
        label = ann.get('label', None)
        
        line = self.ax.axhline(
            y=y_pos,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=100,  # Render on top of other elements
            picker=5  # Enable picking with 5pt tolerance
        )
        
        # Attach metadata for drag handling
        line._annotation_data = ann
        line._is_dragging = False
        return line
    
    def _render_text(self, ann: dict):
        """Render text annotation."""
        x_pos = ann.get('x', 0)
        y_pos = ann.get('y', 0)
        text = ann.get('text', '')
        fontsize = ann.get('fontsize', 12)
        color = ann.get('color', 'black')
        ha = ann.get('ha', 'center')
        va = ann.get('va', 'center')
        
        # Get bbox from annotation, use default if not specified
        bbox_props = ann.get('bbox')
        
        text_obj = self.ax.text(
            x_pos,
            y_pos,
            text,
            fontsize=fontsize,
            color=color,
            ha=ha,
            va=va,
            bbox=bbox_props,
            zorder=101,  # Render on top of lines
            picker=True  # Enable picking for drag functionality
        )
        
        # Attach metadata for drag handling
        text_obj._annotation_data = ann
        text_obj._is_dragging = False
        return text_obj

    def _render_arrow(self, ann: dict):
        """Render arrow annotation.

        Uses a FancyArrowPatch (added via add_patch) rather than
        ax.annotate('', ...) -- Patch.contains() picks along the whole
        drawn arrow shape, whereas an empty-string Annotation's pick region
        is effectively just its (near zero-size) text anchor point, which
        would make double-click-to-edit and drag only work when clicking
        almost exactly on the arrow tip.
        """
        x1, y1 = ann.get('x1', 0), ann.get('y1', 0)
        x2, y2 = ann.get('x2', 0), ann.get('y2', 0)
        color = ann.get('color', 'black')
        linewidth = ann.get('linewidth', 1.5)
        linestyle = ann.get('linestyle', '-')

        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            arrowstyle='-|>',
            mutation_scale=15,
            zorder=100,
            picker=True,
        )
        self.ax.add_patch(arrow)

        arrow._annotation_data = ann
        arrow._is_dragging = False
        return arrow

    def _render_span(self, ann: dict):
        """Render a vertical or horizontal shaded span (axvspan/axhspan).

        Both return a plain Rectangle patch under the hood (confirmed:
        axvspan's y-extent is a blended-transform 0-1 axes fraction, x is
        data coords, and vice-versa for axhspan) -- that's what makes
        dragging just a set_x()/set_y() call, no custom polygon math.
        """
        ann_type = ann.get('type')
        color = ann.get('color', 'orange')
        alpha = ann.get('alpha', 0.3)

        if ann_type == 'vspan':
            x1, x2 = ann.get('x1', 0), ann.get('x2', 1)
            span = self.ax.axvspan(x1, x2, color=color, alpha=alpha, zorder=50, picker=True)
        else:
            y1, y2 = ann.get('y1', 0), ann.get('y2', 1)
            span = self.ax.axhspan(y1, y2, color=color, alpha=alpha, zorder=50, picker=True)

        span._annotation_data = ann
        span._is_dragging = False
        return span

    def _render_box(self, ann: dict):
        """Render a rectangle/box annotation."""
        x, y = ann.get('x', 0), ann.get('y', 0)
        width = ann.get('width', 1)
        height = ann.get('height', 1)
        facecolor = ann.get('facecolor', 'yellow')
        edgecolor = ann.get('edgecolor', 'black')
        linewidth = ann.get('linewidth', 1.5)
        alpha = ann.get('alpha', 0.3)

        box = patches.Rectangle(
            (x, y), width, height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=99,
            picker=True,
        )
        self.ax.add_patch(box)

        box._annotation_data = ann
        box._is_dragging = False
        return box

    def _render_callout(self, ann: dict):
        """Render a callout: text with an arrow pointing at a data point."""
        x, y = ann.get('x', 0), ann.get('y', 0)
        tx, ty = ann.get('tx', x), ann.get('ty', y)
        text = ann.get('text', '')
        fontsize = ann.get('fontsize', 11)
        color = ann.get('color', 'black')
        arrowcolor = ann.get('arrowcolor', 'black')

        callout = self.ax.annotate(
            text, xy=(x, y), xytext=(tx, ty),
            fontsize=fontsize, color=color,
            arrowprops=dict(arrowstyle='->', color=arrowcolor),
            zorder=101,
            picker=True,
        )

        callout._annotation_data = ann
        callout._is_dragging = False
        return callout

    def _ax_data_coords(self, event):
        """(x, y) of a mouse event in self.ax's data coordinates, computed
        from the raw pixel position -- not event.xdata/ydata.

        event.xdata/ydata come from event.inaxes, which resolves to the
        topmost overlapping Axes: a twinx() secondary axis covers self.ax
        entirely, so relying on them broke annotation picking/dragging on
        any graph with a secondary axis. Transforming through
        self.ax.transData directly sidesteps the ambiguity.
        """
        return self.ax.transData.inverted().transform((event.x, event.y))

    def _on_annotation_click(self, event):
        """Handle a press on the canvas: a single click on an annotation
        starts a potential drag, a double-click opens its edit dialog.

        Hit-testing is done manually against self.ax's own annotation
        artists (self.ax.findobj() + artist.contains(event)) rather than
        relying on matplotlib's pick_event -- Figure.pick() only delivers a
        pick event to an artist when event.inaxes equals that artist's own
        Axes, which (see _ax_data_coords) is unreliable whenever a
        secondary Y-axis overlaps self.ax. artist.contains(event) itself
        only needs the event's pixel position, which is always correct.
        """
        if event.x is None or event.y is None:
            return

        # Cancel any pending/in-progress drag -- a new press always
        # supersedes it, whether or not this press turns out to be on an
        # annotation.
        if hasattr(self, '_drag_candidate'):
            del self._drag_candidate
        if hasattr(self, '_dragged_annotation'):
            self._dragged_annotation._is_dragging = False
            del self._dragged_annotation

        for ann in self.ax.findobj():
            if hasattr(ann, '_annotation_data'):
                contains, _ = ann.contains(event)
                if contains:
                    if event.dblclick:
                        self._edit_annotation_direct(ann._annotation_data)
                    else:
                        # Record as drag candidate -- actual drag starts on
                        # mouse move past a small threshold (see
                        # _on_annotation_drag).
                        self._drag_candidate = ann
                        self._drag_start_x, self._drag_start_y = self._ax_data_coords(event)
                    return
                    
    def _edit_annotation_direct(self, annotation):
        """Open edit dialog for annotation (called from double-click)."""
        
        # Open appropriate edit dialog based on type
        if annotation['type'] in ['vline', 'hline']:
            dialog = EditLineDialog(annotation, None)
            if dialog.exec() == QDialog.Accepted:
                # Update annotation properties
                props = dialog.get_properties()
                annotation.update(props)
                
                # Update label
                if annotation['type'] == 'vline':
                    annotation['label'] = f"V-Line at x={annotation['x']:.2f}"
                else:
                    annotation['label'] = f"H-Line at y={annotation['y']:.2f}"
                
                # Refresh plot
                self.ax.clear()
                if self.df is not None:
                    self.plot(self.df)
        
        elif annotation['type'] == 'text':
            dialog = EditTextDialog(annotation, None)
            if dialog.exec() == QDialog.Accepted:
                # Update annotation properties
                props = dialog.get_properties()
                annotation.update(props)

                # Refresh plot
                self.ax.clear()
                if self.df is not None:
                    self.plot(self.df)

        elif annotation['type'] == 'arrow':
            dialog = EditArrowDialog(annotation, None)
            if dialog.exec() == QDialog.Accepted:
                annotation.update(dialog.get_properties())
                self.ax.clear()
                if self.df is not None:
                    self.plot(self.df)

        elif annotation['type'] in ('vspan', 'hspan'):
            dialog = EditSpanDialog(annotation, None)
            if dialog.exec() == QDialog.Accepted:
                annotation.update(dialog.get_properties())
                self.ax.clear()
                if self.df is not None:
                    self.plot(self.df)

        elif annotation['type'] == 'box':
            dialog = EditBoxDialog(annotation, None)
            if dialog.exec() == QDialog.Accepted:
                annotation.update(dialog.get_properties())
                self.ax.clear()
                if self.df is not None:
                    self.plot(self.df)

        elif annotation['type'] == 'callout':
            dialog = EditCalloutDialog(annotation, None)
            if dialog.exec() == QDialog.Accepted:
                annotation.update(dialog.get_properties())
                self.ax.clear()
                if self.df is not None:
                    self.plot(self.df)

    def _on_annotation_drag(self, event):
        """Handle annotation drag (mouse move while dragging).

        Uses _ax_data_coords(event) throughout instead of event.xdata/
        event.ydata -- see that method's docstring for why those are
        unreliable whenever a secondary Y-axis overlaps self.ax.
        """
        if event.x is None or event.y is None:
            return
        xdata, ydata = self._ax_data_coords(event)

        # Promote drag candidate to actual drag once mouse moves
        if hasattr(self, '_drag_candidate') and not hasattr(self, '_dragged_annotation'):
            dx = abs(xdata - (self._drag_start_x or 0))
            dy = abs(ydata - (self._drag_start_y or 0))
            # Use a small threshold to distinguish click from drag
            x_range = abs(self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            y_range = abs(self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            if dx > x_range * 0.005 or dy > y_range * 0.005:
                self._dragged_annotation = self._drag_candidate
                self._dragged_annotation._is_dragging = True
                # Snapshot for delta-based dragging of multi-point shapes
                # below (vline/hline/text jump straight to the mouse instead).
                self._drag_start_coords = dict(self._dragged_annotation._annotation_data)
                del self._drag_candidate

        if not hasattr(self, '_dragged_annotation'):
            return

        ann = self._dragged_annotation
        if not getattr(ann, '_is_dragging', False):
            return

        ann_data = ann._annotation_data
        ann_type = ann_data['type']

        # Update visual position based on annotation type
        if ann_type == 'vline':
            ann.set_xdata([xdata, xdata])
            ann_data['x'] = xdata
        elif ann_type == 'hline':
            ann.set_ydata([ydata, ydata])
            ann_data['y'] = ydata
        elif ann_type == 'text':
            ann.set_position((xdata, ydata))
            ann_data['x'] = xdata
            ann_data['y'] = ydata
        elif ann_type == 'arrow':
            start = self._drag_start_coords
            dx = xdata - self._drag_start_x
            dy = ydata - self._drag_start_y
            ann_data['x1'] = start['x1'] + dx
            ann_data['y1'] = start['y1'] + dy
            ann_data['x2'] = start['x2'] + dx
            ann_data['y2'] = start['y2'] + dy
            ann.set_positions((ann_data['x1'], ann_data['y1']), (ann_data['x2'], ann_data['y2']))
        elif ann_type == 'vspan':
            start = self._drag_start_coords
            dx = xdata - self._drag_start_x
            ann_data['x1'] = start['x1'] + dx
            ann_data['x2'] = start['x2'] + dx
            ann.set_x(ann_data['x1'])
        elif ann_type == 'hspan':
            start = self._drag_start_coords
            dy = ydata - self._drag_start_y
            ann_data['y1'] = start['y1'] + dy
            ann_data['y2'] = start['y2'] + dy
            ann.set_y(ann_data['y1'])
        elif ann_type == 'box':
            start = self._drag_start_coords
            dx = xdata - self._drag_start_x
            dy = ydata - self._drag_start_y
            ann_data['x'] = start['x'] + dx
            ann_data['y'] = start['y'] + dy
            ann.set_xy((ann_data['x'], ann_data['y']))
        elif ann_type == 'callout':
            # Moves only the text position -- the arrow's pointed-at data
            # point (xy) is left fixed, matching set_position()'s own
            # behavior (it only ever touches xytext, never xy).
            ann.set_position((xdata, ydata))
            ann_data['tx'] = xdata
            ann_data['ty'] = ydata

        self.canvas.draw_idle()
    
    def _on_annotation_release(self, event):
        """Handle mouse release (finish dragging)."""
        # Clean up drag candidate if no drag occurred
        if hasattr(self, '_drag_candidate'):
            del self._drag_candidate
        
        if not hasattr(self, '_dragged_annotation'):
            return
        
        ann = self._dragged_annotation
        ann._is_dragging = False
        
        # Emit signal to update model
        ann_data = ann._annotation_data
        if ann_data['type'] == 'vline':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x'], 0
            )
        elif ann_data['type'] == 'hline':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], 0, ann_data['y']
            )
        elif ann_data['type'] == 'text':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x'], ann_data['y']
            )
        elif ann_data['type'] == 'arrow':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x1'], ann_data['y1']
            )
        elif ann_data['type'] == 'vspan':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x1'], 0
            )
        elif ann_data['type'] == 'hspan':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], 0, ann_data['y1']
            )
        elif ann_data['type'] == 'box':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x'], ann_data['y']
            )
        elif ann_data['type'] == 'callout':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['tx'], ann_data['ty']
            )

        del self._dragged_annotation
        if hasattr(self, '_drag_start_coords'):
            del self._drag_start_coords
        self.canvas.draw_idle()
    
