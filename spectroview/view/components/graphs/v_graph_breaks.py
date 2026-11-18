"""Broken-axis layout and zoom-inset rendering mixin for VGraph."""


class GraphBreakInsetMixin:
    """Mixin handling broken X/Y axis splitting and zoom-inset rendering."""

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
