"""Axis, twin-axis, legend, grid, scale, limit, and figure-style mixin for VGraph."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullLocator

from spectroview import DEFAULT_COLORS, DEFAULT_MARKERS
from spectroview.i18n import tr
from spectroview.viewmodel.common.ui_utils import show_alert


class GraphAxesStyleMixin:
    """Mixin handling primary/twin axis plotting, legends, scales, limits, labels, and styling."""

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
                    'label': tr('All data'),
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
                show_alert(tr("Unsupported plot style"))

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
                style_kwargs['fontsize'] = getattr(self, 'legend_fontsize', None) or 9

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

    def _is_x_numeric(self) -> bool:
        """Return True if X axis represents numeric values, False otherwise."""
        x_num = getattr(self, 'x_as_numeric', None)
        if x_num is not None:
            return bool(x_num)
        if self.plot_style in ('wafer', '2Dmap'):
            return False
        if self.plot_style in ('point', 'box', 'bar'):
            return False
        if self.plot_style in ('scatter', 'line', 'trendline', 'histogram'):
            if self.df is not None and self.x and self.x in self.df.columns:
                num_vals = pd.to_numeric(self.df[self.x], errors='coerce')
                return bool(len(num_vals) > 0 and num_vals.notna().sum() == len(num_vals))
            return True
        return False

    def _is_y_numeric(self) -> bool:
        """Return True if Y axis represents numeric values, False otherwise."""
        y_num = getattr(self, 'y_as_numeric', None)
        if y_num is not None:
            return bool(y_num)
        if self.plot_style in ('wafer', '2Dmap'):
            return False
        if self.plot_style == 'histogram':
            return True
        if self.plot_style in ('scatter', 'line', 'trendline', 'point', 'box', 'bar'):
            if self.df is not None and self.y:
                y_cols = self.y if isinstance(self.y, list) else [self.y]
                for col in y_cols:
                    if col and col in self.df.columns:
                        num_vals = pd.to_numeric(self.df[col], errors='coerce')
                        if len(num_vals) == 0 or num_vals.notna().sum() != len(num_vals):
                            return False
                return True
            return True
        return False

    def _resolve_minor_ticks(self, side: str) -> bool:
        """Resolve effective visibility of minor ticks on a given side ('bottom', 'left', 'top', 'right')."""
        if self.plot_style == 'wafer':
            return False
        val = getattr(self, f'minor_ticks_{side}', None)
        if val is not None:
            return bool(val)
        if side == 'bottom':
            return bool(self._is_x_numeric())
        elif side == 'left':
            return bool(self._is_y_numeric())
        elif side in ('top', 'right'):
            return False
        return False

    def _set_grid(self):
        """Add grid for the plot (supports linear & log scale automatically)."""
        is_wafer = (self.plot_style == 'wafer')
        self.ax.set_axisbelow(True)

        # tick_direction only overrides matplotlib's default when set.
        extra_ticks = {'labelsize': getattr(self, 'tick_label_fontsize', None) or 9}
        if getattr(self, 'tick_direction', None):
            extra_ticks['direction'] = self.tick_direction

        show_minor_bottom = False if is_wafer else self._resolve_minor_ticks('bottom')
        show_minor_left = False if is_wafer else self._resolve_minor_ticks('left')
        show_minor_top = False if is_wafer else self._resolve_minor_ticks('top')
        show_minor_right = False if is_wafer else self._resolve_minor_ticks('right')

        self.ax.tick_params(
            which='major',
            bottom=False if is_wafer else True,
            left=True,
            top=False if is_wafer else show_minor_top,
            right=show_minor_right,
            **extra_ticks
        )

        if any([show_minor_bottom, show_minor_left, show_minor_top, show_minor_right]):
            self.ax.minorticks_on()
            self.ax.tick_params(
                which='minor',
                bottom=show_minor_bottom,
                top=show_minor_top,
                left=show_minor_left,
                right=show_minor_right,
                **({'direction': self.tick_direction} if getattr(self, 'tick_direction', None) else {})
            )
            if not show_minor_bottom and not show_minor_top:
                self.ax.xaxis.set_minor_locator(NullLocator())
            if not show_minor_left and not show_minor_right:
                self.ax.yaxis.set_minor_locator(NullLocator())
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

    def _apply_limit_pair(
        self,
        setter,
        vmin,
        vmax,
        axis_name: str,
        getter=None,
        auto_key: str = "",
        restore_unset: bool = False,
    ) -> None:
        """Apply a (min, max) limit pair via *setter* (e.g. ax.set_xlim),
        supporting single-sided overrides (only *vmin* or only *vmax* set)
        and skipping degenerate equal bounds instead of handing matplotlib a
        zero-width range (which triggers a "singular transformation"
        UserWarning and silently auto-expands anyway). Uses explicit
        `is not None` checks (not truthy checks) so a limit of exactly 0.0
        -- a common, legitimate axis bound -- is not silently ignored.
        """
        auto_bounds = getattr(self, "_auto_limits", {}).get(auto_key) if auto_key else None
        is_log = (
            (auto_key == "x" and self.ax.get_xscale() == "log")
            or (auto_key == "y" and self.ax.get_yscale() == "log")
        )
        if is_log and auto_bounds is not None and min(auto_bounds) <= 0:
            self.ax.relim()
            self.ax.autoscale_view(
                scalex=(auto_key == "x"), scaley=(auto_key == "y")
            )
            auto_bounds = getter() if getter is not None else None
        if auto_bounds is None and getter is not None:
            auto_bounds = getter()

        if vmin is None and vmax is None:
            if restore_unset and auto_bounds is not None:
                cur_a, cur_b = auto_bounds
                if cur_a != cur_b:
                    setter(float(cur_a), float(cur_b))
            return

        if vmin is None or vmax is None:
            if auto_bounds is None:
                return
            cur_a, cur_b = auto_bounds
            cur_lo, cur_hi = (cur_a, cur_b) if cur_a <= cur_b else (cur_b, cur_a)
            vmin = cur_lo if vmin is None else vmin
            vmax = cur_hi if vmax is None else vmax

        vmin, vmax = float(vmin), float(vmax)
        if vmin == vmax:
            self.notify.emit(tr("Skipping {axis} limits: min == max ({value}).", axis=axis_name, value=vmin))
            return
        setter(vmin, vmax)

    def _set_limits(self, restore_unset: bool = False):
        """Set the limits of axis."""
        self._apply_limit_pair(
            self.ax.set_xlim, self.xmin, self.xmax, "x-axis",
            getter=self.ax.get_xlim, auto_key="x", restore_unset=restore_unset,
        )
        self._apply_limit_pair(
            self.ax.set_ylim, self.ymin, self.ymax, "y-axis",
            getter=self.ax.get_ylim, auto_key="y", restore_unset=restore_unset,
        )
        if self.ax2:
            self._apply_limit_pair(
                self.ax2.set_ylim, self.y2min, self.y2max, "y2-axis",
                getter=self.ax2.get_ylim, auto_key="y2", restore_unset=restore_unset,
            )
        if self.ax3:
            self._apply_limit_pair(
                self.ax3.set_ylim, self.y3min, self.y3max, "y3-axis",
                getter=self.ax3.get_ylim, auto_key="y3", restore_unset=restore_unset,
            )
        if self.ax_x2:
            self._apply_limit_pair(
                self.ax_x2.set_xlim, self.x2min, self.x2max, "x2-axis",
                getter=self.ax_x2.get_xlim, auto_key="x2", restore_unset=restore_unset,
            )

    def _set_axis_scale(self, df, reset_linear: bool = False):
        """Apply log/symlog scale only if the corresponding axis column is numeric."""
        if df is None:
            return

        if self.xlogscale:
            x_data = df[self.x]
            if np.issubdtype(x_data.dtype, np.number):
                self.ax.set_xscale(self.xscale_mode if self.xscale_mode == 'symlog' else 'log')
            else:
                self.notify.emit(tr("Skipping x-logscale because '{col}' is categorical.", col=self.x))
        elif reset_linear and self.ax.get_xscale() != 'linear':
            self.ax.set_xscale('linear')

        if self.ylogscale and len(self.y) > 0:
            y_data = df[self.y[0]]
            if np.issubdtype(y_data.dtype, np.number):
                self.ax.set_yscale(self.yscale_mode if self.yscale_mode == 'symlog' else 'log')
            else:
                self.notify.emit(tr("Skipping y-logscale because '{col}' is categorical.", col=self.y[0]))
        elif reset_linear and self.ax.get_yscale() != 'linear':
            self.ax.set_yscale('linear')

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
                return tr("{peak} peak position (cm$^{{-1}}$)", peak=peaklabel)
            elif param == "fwhm":
                return tr("{peak} peak width (cm$^{{-1}}$)", peak=peaklabel)
            elif param == "ampli":
                return tr("{peak} peak intensity (a.u.)", peak=peaklabel)
            elif param == "area":
                return tr("{peak} peak area (a.u.)", peak=peaklabel)

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
                    self.ax.set_ylabel(tr("Frequency"))
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
        # switching to Dark left labels/title/ticks/spines black.
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
            # Subtitle always tracks the title size, 2pt smaller.
            fontsize = max(1, (self.title_fontsize or 12) - 2)
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
