"""Matplotlib graph visualization widget for MVVM pattern."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QPushButton, QToolButton, QHBoxLayout, QMenu, QFrame,
    QApplication, QLabel,
)
from PySide6.QtCore import QSize, Signal, Qt, QTimer
from PySide6.QtGui import QAction

from spectroview import (
    PLOT_POLICY_LIGHT, PLOT_POLICY_DARK,
)
from spectroview.i18n import bind_text, bind_tooltip, on_language_changed
from spectroview.view.theme import Icons, current_tokens
from spectroview.model.graphs.m_graph import MGraph
from spectroview.viewmodel.common.export_utils import copy_fig_to_clb
from spectroview.view.components.graphs.v_graph_annotations import GraphAnnotationMixin
from spectroview.view.components.graphs.v_graph_axes import GraphAxesStyleMixin
from spectroview.view.components.graphs.v_graph_breaks import GraphBreakInsetMixin
from spectroview.view.components.graphs.v_graph_canvas import (
    _DebouncedFigureCanvas,
    _FastNavigationToolbar,
)
from spectroview.view.components.graphs.v_plot_renderer import PlotRenderer

# figure_theme value -> mplstyle file. "light" (the default) reproduces the
# app's only-ever-existing hardcoded behavior; dark reuses the same
# theme file the Maps/Spectra viewers switch between.
_THEME_STYLE_MAP = {
    "light": PLOT_POLICY_LIGHT,
    "dark": PLOT_POLICY_DARK,
}


class VGraph(GraphBreakInsetMixin, GraphAxesStyleMixin, GraphAnnotationMixin, QWidget):
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
        self.renderer = PlotRenderer(self)

        # Layout setup
        self.graph_layout = QVBoxLayout()
        self.setLayout(self.graph_layout)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_layout.setSpacing(0)
        on_language_changed(self, self._retranslate_graph)

    def _retranslate_graph(self):
        """Re-render the current plot when UI language changes so localized labels update."""
        if self.df is not None and self.canvas is not None and self.ax is not None:
            self.plot(self.df)

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

        # Debounced canvas: avoids a constrained-layout redraw storm on resize.
        self.canvas = _DebouncedFigureCanvas(self.figure)

        # Matplotlib toolbar kept hidden as a background interaction helper for zoom/pan/rescale
        # (fast-pan subclass: layout suspended + throttled redraws while dragging).
        self.toolbar = _FastNavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        # Forward any matplotlib toolbar message events to our built-in coordinate label
        def _on_toolbar_message(s):
            clean_s = " ".join(s.split()) if s else ""
            if hasattr(self, "lbl_coordinates"):
                self.lbl_coordinates.setText(clean_s)
            if hasattr(self.toolbar, "locLabel"):
                self.toolbar.locLabel.setText(s)

        self.toolbar.set_message = _on_toolbar_message

        # Dedicated navigation buttons (Home/Rescale, Pan/Drag, Zoom) matching 30x30 SPECTROview controls
        self.btn_home = QPushButton()
        self.btn_home.setIcon(Icons.rescale())
        self.btn_home.setIconSize(QSize(22, 22))
        self.btn_home.setFixedSize(30, 30)
        bind_tooltip(self.btn_home, "Rescale plot to fit data (Ctrl+R)")
        self.btn_home.clicked.connect(self.toolbar.home)
        self.btn_rescale = self.btn_home  # Alias matching v_spectra_viewer / v_map_viewer

        self.btn_pan = QToolButton()
        self.btn_pan.setCheckable(True)
        self.btn_pan.setIcon(Icons.pan())
        self.btn_pan.setIconSize(QSize(22, 22))
        self.btn_pan.setFixedSize(30, 30)
        bind_tooltip(self.btn_pan, "Pan / Drag axes with left mouse, zoom with right")
        self.btn_pan.clicked.connect(self.toolbar.pan)

        self.btn_zoom = QToolButton()
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setIcon(Icons.zoom())
        self.btn_zoom.setIconSize(QSize(22, 22))
        self.btn_zoom.setFixedSize(30, 30)
        bind_tooltip(self.btn_zoom, "Zoom to rectangle")
        self.btn_zoom.clicked.connect(self.toolbar.zoom)

        # Bidirectional sync with matplotlib toolbar action checked states
        action_dict = {a.text(): a for a in self.toolbar.actions() if a.text()}
        if 'Pan' in action_dict:
            action_dict['Pan'].toggled.connect(self.btn_pan.setChecked)
        if 'Zoom' in action_dict:
            action_dict['Zoom'].toggled.connect(self.btn_zoom.setChecked)

        # Create Replicate button
        self.btn_replicate = QPushButton()
        self.btn_replicate.setIcon(Icons.duplicate())
        self.btn_replicate.setIconSize(QSize(22, 22))
        self.btn_replicate.setFixedSize(30, 30)
        bind_tooltip(self.btn_replicate, "Duplicate selected graph")
        self.btn_replicate.clicked.connect(lambda *args: self.replicate_requested.emit(self.graph_id))

        # Create Export button
        # Click exports just this graph; Ctrl+Click exports every open graph.
        self.btn_export = QPushButton()
        self.btn_export.setIcon(Icons.export())
        self.btn_export.setIconSize(QSize(22, 22))
        self.btn_export.setFixedSize(30, 30)
        bind_tooltip(self.btn_export, "Export selected graph to file. Hold Ctrl + Click to export all open graphs")
        self.btn_export.clicked.connect(self._on_export_clicked)

        # Style menu: every action just emits style_action_requested,
        # handled centrally by VWorkspaceGraphs.
        self.btn_style_menu = QToolButton()
        self.btn_style_menu.setFixedSize(30, 30)
        self.btn_style_menu.setIcon(Icons.styling())
        self.btn_style_menu.setIconSize(QSize(22, 22))
        bind_tooltip(self.btn_style_menu, "Graph style: save/apply, copy/paste, reset, set default")
        self.btn_style_menu.setPopupMode(QToolButton.InstantPopup)
        style_menu = QMenu(self)
        _style_actions = [
            ("Save Style...", "save_template", Icons.save()),
            ("Apply Style...", "apply_template", Icons.check()),
            ("Copy Style", "copy", Icons.copy()),
            ("Paste Style", "paste", Icons.paste()),
            ("Reset to Default", "reset", Icons.undo()),
            ("Set as Default Style", "set_default", Icons.settings()),
        ]
        for action_text, action_name, action_icon in _style_actions:
            action = QAction(action_icon, action_text, self)
            bind_text(action, action_text)
            action.triggered.connect(
                lambda checked=False, name=action_name: self.style_action_requested.emit(self.graph_id, name)
            )
            style_menu.addAction(action)
        self.btn_style_menu.setMenu(style_menu)

        # Create Customize button (Editing icon instead of gear)
        self.btn_customize = QPushButton()
        self.btn_customize.setIcon(Icons.edit())
        self.btn_customize.setIconSize(QSize(22, 22))
        self.btn_customize.setFixedSize(30, 30)
        bind_tooltip(self.btn_customize, "Customize graph (Ctrl + E)")
        self.btn_customize.clicked.connect(lambda: self.customize_requested.emit(self.graph_id))

        # Create Copy button
        self.btn_copy_figure = QPushButton()
        self.btn_copy_figure.setIcon(Icons.copy())
        self.btn_copy_figure.setIconSize(QSize(22, 22))
        self.btn_copy_figure.setFixedSize(30, 30)
        bind_tooltip(self.btn_copy_figure, "Copy selected figure to clipboard (Ctrl + C)")
        self.btn_copy_figure.clicked.connect(self.copy_to_clipboard)

        # Create toolbar layout:
        # Left: Replicate, Export, Style
        # Stretch (center)
        # Right: Home, Pan, Zoom, Separator, Customize, Copy, Coordinates
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        toolbar_layout.addWidget(self.btn_replicate)
        toolbar_layout.addWidget(self.btn_export)
        toolbar_layout.addWidget(self.btn_style_menu)

        toolbar_layout.addStretch()

        # Dedicated 30x30 navigation controls matching left buttons
        toolbar_layout.addWidget(self.btn_home)
        toolbar_layout.addWidget(self.btn_pan)
        toolbar_layout.addWidget(self.btn_zoom)

        # Built-in live coordinates display
        self.lbl_coordinates = QLabel("")
        self.lbl_coordinates.setObjectName("lblCoordinates")
        self.lbl_coordinates.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_coordinates.setMinimumWidth(180)
        tok = current_tokens()
        self.lbl_coordinates.setStyleSheet(f"font-size: 11px; padding: 0px 4px; color: {tok.text_secondary};")
        toolbar_layout.addWidget(self.lbl_coordinates)

        # Vertical separator between navigation/coordinates and edit/copy actions
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        toolbar_layout.addWidget(separator)

        # Very right side: Customize Graph and Copy to clipboard
        toolbar_layout.addWidget(self.btn_customize)
        toolbar_layout.addWidget(self.btn_copy_figure)

        # Toolbar buttons should never steal focus from the MDI subwindow/canvas
        for btn in (
            self.btn_home, self.btn_pan, self.btn_zoom,
            self.btn_replicate, self.btn_export, self.btn_style_menu,
            self.btn_customize, self.btn_copy_figure,
        ):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Create container widget for toolbar layout -- kept parentless and
        # hidden here; VWorkspaceGraphs.sync_active_graph_toolbar() reparents
        # it into the shared toolbar slot when this graph is the active one.
        toolbar_container = QWidget()
        toolbar_container.setLayout(toolbar_layout)
        toolbar_container.setFixedHeight(38)
        toolbar_container.hide()
        self.toolbar_container = toolbar_container
        if old_toolbar_container is not None:
            old_toolbar_container.deleteLater()

        self.graph_layout.addWidget(self.canvas)

        # Connect pick event for legend customization
        self.canvas.mpl_connect('pick_event', self._on_legend_pick)

        # Connect annotation drag events and canvas hover/leave
        self.canvas.mpl_connect('motion_notify_event', self._on_annotation_drag)
        self.canvas.mpl_connect('button_release_event', self._on_annotation_release)
        self.canvas.mpl_connect('button_press_event', self._on_annotation_click)
        self.canvas.mpl_connect('figure_leave_event', self._on_figure_leave)

        self.canvas.draw_idle()

    def copy_to_clipboard(self):
        """Copy the current figure to clipboard."""
        copy_fig_to_clb(self.canvas)
        self._flash_copy_feedback()

    def _flash_copy_feedback(self) -> None:
        """Provide 1-second visual checkmark feedback on copy figure button."""
        self.btn_copy_figure.setIcon(Icons.check())
        QTimer.singleShot(1000, self, self._reset_copy_feedback)

    def _reset_copy_feedback(self) -> None:
        try:
            from spectroview.view.theme import ThemeManager
            icon_color = ThemeManager.instance().icon_color if ThemeManager.instance() else None
            self.btn_copy_figure.setIcon(Icons.copy(color=icon_color))
        except (RuntimeError, AttributeError, TypeError):
            pass

    def apply_global_theme(self, theme: str = "") -> None:
        """Refresh widget-level theme colors without modifying MGraph.figure_theme."""
        tok = current_tokens()
        if hasattr(self, "lbl_coordinates"):
            self.lbl_coordinates.setStyleSheet(
                f"font-size: 11px; padding: 0px 4px; color: {tok.text_secondary};"
            )
        if hasattr(self, "btn_copy_figure"):
            self.btn_copy_figure.setIcon(Icons.copy(color=tok.text_primary))

    def _rescale(self):
        """Reset the plot to its original auto-scaled view -- the matplotlib
        toolbar's Home action, exposed for the Ctrl+R shortcut."""
        if getattr(self, 'toolbar', None) is not None:
            self.toolbar.home()

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

        self._auto_limits = {
            "x": self.ax.get_xlim(),
            "y": self.ax.get_ylim(),
            "y2": self.ax2.get_ylim() if self.ax2 else None,
            "y3": self.ax3.get_ylim() if self.ax3 else None,
            "x2": self.ax_x2.get_xlim() if self.ax_x2 else None,
        }
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
            self._set_axis_scale(self.df, reset_linear=True)
            self._set_limits(restore_unset=True)
            self._set_axis_direction()
            self._set_labels()
            self._set_figure_style()
            self._set_grid()
            self._set_rotation()
            self._set_legend()

        self.canvas.draw_idle()
        return True
