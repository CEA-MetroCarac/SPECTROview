import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from spectroview import DEFAULT_COLORS, DEFAULT_MARKERS
from spectroview.viewmodel.utils import show_alert

class PlotRenderer:
    """Class that handles all plot rendering logic, decoupled from the VGraph widget."""
    def __init__(self, vg):
        self.vg = vg # VGraph instance

    def _get_sorted_categories(self, series, df=None, sort_col=None):
        """Return unique values in their current order (dataframe is already sorted)."""
        return [c for c in series.unique() if pd.notna(c)]

    def _prepare_plot_data(self, df, y):
        """Prepare dataframe and X positions for plotting."""
        cols = []
        if self.vg.x not in cols: cols.append(self.vg.x)
        if y not in cols: cols.append(y)
        if self.vg.z and self.vg.z in df.columns and self.vg.z not in cols:
            cols.append(self.vg.z)
            
        plot_df = df[cols].copy()
        
        # --- Sort dataframe based on settings ---
        sort_enabled = getattr(self.vg, 'sort_data_enabled', True)
        sort_by = getattr(self.vg, 'sort_data_by', 'Z')
        
        if sort_enabled:
            col_to_sort = None
            if sort_by == 'X' and self.vg.x in plot_df.columns:
                col_to_sort = self.vg.x
            elif sort_by == 'Y' and y in plot_df.columns:
                col_to_sort = y
            elif sort_by == 'Z' and self.vg.z in plot_df.columns:
                col_to_sort = self.vg.z
                
            if col_to_sort:
                try:
                    plot_df = plot_df.sort_values(by=col_to_sort)
                except TypeError:
                    # Fallback to string sort if mixed types exist
                    plot_df['_sort_key'] = plot_df[col_to_sort].astype(str)
                    plot_df = plot_df.sort_values(by='_sort_key').drop(columns=['_sort_key'])
        
        treat_as_numeric = getattr(self.vg, 'x_as_numeric', None)
        # If 'Auto' (None), auto-detect numeric if plot style expects it by default
        if treat_as_numeric is None:
            if self.vg.plot_style in ['scatter', 'line', 'trendline', 'histogram']:
                num_vals = pd.to_numeric(plot_df[self.vg.x], errors='coerce')
                if len(num_vals) > 0 and num_vals.notna().sum() == len(num_vals):
                    treat_as_numeric = True
                else:
                    treat_as_numeric = False
            else:
                treat_as_numeric = False
        
        dropna_cols = [self.vg.x, y]
        if self.vg.z and self.vg.z in plot_df.columns:
            dropna_cols.append(self.vg.z)
            
        if treat_as_numeric:
            plot_df[self.vg.x] = pd.to_numeric(plot_df[self.vg.x], errors='coerce')
            plot_df = plot_df.dropna(subset=dropna_cols)
            # For numeric X, we MUST sort X so the line draws left-to-right
            plot_df = plot_df.sort_values(by=self.vg.x)
            x_unique = list(plot_df[self.vg.x].unique())
            x_positions = {v: v for v in x_unique}
        else:
            plot_df = plot_df.dropna(subset=dropna_cols)
            # Raw unique values in current (sorted) DataFrame order
            x_unique = list(dict.fromkeys(plot_df[self.vg.x]))
            x_positions = {v: i for i, v in enumerate(x_unique)}
            
        return plot_df, x_unique, x_positions, treat_as_numeric


    def _plot_point(self, df, y, colors, markers, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        edge_c = getattr(self.vg, 'scatter_edgecolor', 'black')
        ms = np.sqrt(self.vg.scatter_size) if hasattr(self.vg, 'scatter_size') else 7
        join = getattr(self.vg, 'join_for_point_plot', False)
        
        if self.vg.z and self.vg.z in plot_df.columns:
            categories = self._get_sorted_categories(plot_df[self.vg.z], df=plot_df)
            n_hue = len(categories)
            dodge = getattr(self.vg, 'dodge_point_plot', True) and not is_numeric
            if dodge and n_hue > 1:
                offsets = np.linspace(-0.2, 0.2, n_hue)
            else:
                offsets = np.zeros(n_hue)

            for idx, cat in enumerate(categories):
                subset = plot_df[plot_df[self.vg.z] == cat]
                if subset.empty: continue
                grouped = subset.groupby(self.vg.x)[y]
                
                x_cat_vals = list(grouped.groups.keys())
                x_vals = np.array([x_positions[val] + offsets[idx] for val in x_cat_vals], dtype=float)
                
                means = grouped.mean().values
                cis = grouped.sem().values * 1.96  # 95% CI
                cis = np.nan_to_num(cis)
                
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)] if markers else 'o'
                
                self.vg.ax.errorbar(
                    x_vals, means, yerr=cis,
                    fmt=marker, color=color, markersize=ms,
                    markeredgecolor=edge_c, markeredgewidth=0.5,
                    capsize=3, elinewidth=1,
                    linestyle='-' if join else 'none'
                )
                self.vg.ax.plot([], [], marker=marker, color=color, markersize=ms,
                             markeredgecolor=edge_c, markeredgewidth=0.5,
                             linestyle='-' if join else 'none', label=str(cat))
        else:
            grouped = plot_df.groupby(self.vg.x)[y]
            x_cat_vals = list(grouped.groups.keys())
            x_vals = np.array([x_positions[val] for val in x_cat_vals], dtype=float)
            means = grouped.mean().values
            cis = grouped.sem().values * 1.96
            cis = np.nan_to_num(cis)
            
            self.vg.ax.errorbar(
                x_vals, means, yerr=cis,
                fmt='o', color=c, markersize=ms,
                markeredgecolor=edge_c, markeredgewidth=0.5,
                capsize=3, elinewidth=1,
                linestyle='-' if join else 'none'
            )
            
        if not is_numeric:
            self.vg.ax.set_xticks(list(x_positions.values()))
            self.vg.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_scatter(self, df, y, colors, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        edge_c = getattr(self.vg, 'scatter_edgecolor', 'black')
        dodge = getattr(self.vg, 'dodge_scatter_plot', False) and not is_numeric
        
        if self.vg.z and self.vg.z in plot_df.columns:
            categories = self._get_sorted_categories(plot_df[self.vg.z], df=plot_df)
            n_hue = len(categories)
            if dodge and n_hue > 1:
                offsets = np.linspace(-0.3, 0.3, n_hue)
            else:
                offsets = np.zeros(n_hue)

            for idx, cat in enumerate(categories):
                subset = plot_df[plot_df[self.vg.z] == cat]
                if subset.empty: continue
                color = colors[idx % len(colors)]
                
                x_vals = np.array([x_positions[val] + offsets[idx] for val in subset[self.vg.x]], dtype=float)
                
                self.vg.ax.scatter(
                    x_vals, subset[y].values,
                    color=color, s=self.vg.scatter_size,
                    edgecolors=edge_c, linewidths=0.5, label=str(cat)
                )
        else:
            x_vals = np.array([x_positions[val] for val in plot_df[self.vg.x]], dtype=float)
            self.vg.ax.scatter(
                x_vals, plot_df[y].values,
                color=c, s=self.vg.scatter_size,
                edgecolors=edge_c, linewidths=0.5
            )
            
        if not is_numeric:
            self.vg.ax.set_xticks(list(x_positions.values()))
            self.vg.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_box(self, df, y, colors, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        
        if len(x_unique) < 2:
            box_width = 0.4
        else:
            if is_numeric:
                min_gap = min(x_positions[x_unique[i+1]] - x_positions[x_unique[i]] for i in range(len(x_unique) - 1))
            else:
                min_gap = 1.0
            box_width = min_gap * 0.6

        if self.vg.z and self.vg.z in plot_df.columns:
            hue_cats = self._get_sorted_categories(plot_df[self.vg.z], df=plot_df)
            n_hue = len(hue_cats)
            sub_width = box_width / n_hue
            offsets = np.linspace(-(box_width - sub_width) / 2,
                                  (box_width - sub_width) / 2, n_hue)
            legend_handles = []
            for h_idx, cat in enumerate(hue_cats):
                subset = plot_df[plot_df[self.vg.z] == cat]
                color = colors[h_idx % len(colors)]
                
                data_groups = []
                positions = []
                for xv in x_unique:
                    vals = subset[subset[self.vg.x] == xv][y].values
                    if len(vals) > 0:
                        data_groups.append(vals)
                        positions.append(x_positions[xv] + offsets[h_idx])
                
                if data_groups:
                    bp = self.vg.ax.boxplot(
                        data_groups, positions=positions, widths=sub_width * 0.9,
                        patch_artist=True, manage_ticks=False
                    )
                    for patch in bp['boxes']:
                        patch.set_facecolor(color)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(0.8)
                    for element in ['whiskers', 'caps', 'medians', 'fliers']:
                        for line in bp.get(element, []):
                            line.set_color('black')
                
                p = patches.Rectangle((0,0), 0, 0, facecolor=color, edgecolor='black', label=str(cat))
                self.vg.ax.add_patch(p)
        else:
            data_groups = []
            positions = []
            for xv in x_unique:
                vals = plot_df[plot_df[self.vg.x] == xv][y].values
                if len(vals) > 0:
                    data_groups.append(vals)
                    positions.append(x_positions[xv])
            
            if data_groups:
                bp = self.vg.ax.boxplot(
                    data_groups, positions=positions, widths=box_width,
                    patch_artist=True, manage_ticks=False
                )
                for patch in bp['boxes']:
                    patch.set_facecolor(c)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(0.8)
                for element in ['whiskers', 'caps', 'medians', 'fliers']:
                    for line in bp.get(element, []):
                        line.set_color('black')

        if is_numeric:
            self.vg.ax.set_xticks([x_positions[xv] for xv in x_unique])
            self.vg.ax.set_xticklabels([str(xv) for xv in x_unique])
        else:
            self.vg.ax.set_xticks(list(x_positions.values()))
            self.vg.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_line(self, df, y, colors, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        
        if self.vg.z and self.vg.z in plot_df.columns:
            categories = self._get_sorted_categories(plot_df[self.vg.z], df=plot_df)
            for idx, cat in enumerate(categories):
                subset = plot_df[plot_df[self.vg.z] == cat]
                if subset.empty: continue
                grouped = subset.groupby(self.vg.x)[y]
                
                x_cat_vals = list(grouped.groups.keys())
                x_vals = np.array([x_positions[val] for val in x_cat_vals], dtype=float)
                
                means = grouped.mean().values
                cis = grouped.sem().values * 1.96  # 95% CI
                cis = np.nan_to_num(cis)
                
                color = colors[idx % len(colors)]
                
                self.vg.ax.plot(x_vals, means, color=color, label=str(cat))
                self.vg.ax.fill_between(x_vals, means - cis, means + cis, color=color, alpha=0.2)
        else:
            grouped = plot_df.groupby(self.vg.x)[y]
            x_cat_vals = list(grouped.groups.keys())
            x_vals = np.array([x_positions[val] for val in x_cat_vals], dtype=float)
            means = grouped.mean().values
            cis = grouped.sem().values * 1.96
            cis = np.nan_to_num(cis)
            
            self.vg.ax.plot(x_vals, means, color=c)
            self.vg.ax.fill_between(x_vals, means - cis, means + cis, color=c, alpha=0.2)
            
        if not is_numeric:
            self.vg.ax.set_xticks(list(x_positions.values()))
            self.vg.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_bar(self, df, y, colors, c):
        plot_df, x_unique, x_positions, is_numeric = self._prepare_plot_data(df, y)
        
        if len(x_unique) < 2:
            bar_width = 0.4
        else:
            if is_numeric:
                min_gap = min(x_positions[x_unique[i+1]] - x_positions[x_unique[i]] for i in range(len(x_unique) - 1))
            else:
                min_gap = 1.0
            bar_width = min_gap * 0.6

        if self.vg.z and self.vg.z in plot_df.columns:
            hue_cats = self._get_sorted_categories(plot_df[self.vg.z], df=plot_df)
            n_hue = len(hue_cats)
            if n_hue == 0:
                return
            sub_width = bar_width / n_hue
            offsets = np.linspace(-(bar_width - sub_width) / 2,
                                  (bar_width - sub_width) / 2, n_hue)
            for h_idx, cat in enumerate(hue_cats):
                subset = plot_df[plot_df[self.vg.z] == cat]
                grouped = subset.groupby(self.vg.x)[y]
                means = grouped.mean()
                stds = grouped.std() if self.vg.show_bar_plot_error_bar else None
                color = colors[h_idx % len(colors)]
                
                positions = [x_positions[xv] + offsets[h_idx] for xv in x_unique]
                heights = [means.get(xv, 0) for xv in x_unique]
                yerr = [stds.get(xv, 0) for xv in x_unique] if stds is not None else None
                
                self.vg.ax.bar(
                    positions, heights, width=sub_width * 0.9,
                    color=color, edgecolor='black', linewidth=0.8,
                    yerr=yerr, capsize=3, ecolor='black',
                    label=str(cat)
                )
        else:
            grouped = plot_df.groupby(self.vg.x)[y]
            means = grouped.mean()
            stds = grouped.std() if self.vg.show_bar_plot_error_bar else None
            
            positions = [x_positions[xv] for xv in x_unique]
            heights = [means.get(xv, 0) for xv in x_unique]
            yerr = [stds.get(xv, 0) for xv in x_unique] if stds is not None else None
            
            self.vg.ax.bar(
                positions, heights, width=bar_width,
                color=c, edgecolor='black', linewidth=0.8,
                yerr=yerr, capsize=3, ecolor='black'
            )
            
        if is_numeric:
            self.vg.ax.set_xticks([x_positions[xv] for xv in x_unique])
            self.vg.ax.set_xticklabels([str(xv) for xv in x_unique])
        else:
            self.vg.ax.set_xticks(list(x_positions.values()))
            self.vg.ax.set_xticklabels([str(v) for v in x_unique])

    def _plot_trendline(self, df, y, colors, c):
        self.vg.trendline_equations = []  # reset before recomputing
        anchor = getattr(self.vg, 'trendline_anchor_enabled', False)
        
        if self.vg.z and self.vg.z in df.columns:
            categories = self._get_sorted_categories(df[self.vg.z], df=df)
            for idx, cat in enumerate(categories):
                subset = df[df[self.vg.z] == cat]
                color = colors[idx % len(colors)]
                
                try:
                    x_fit, y_fit, coeffs = self._fit_trendline(subset)
                except Exception:
                    continue
                
                self.vg.ax.scatter(
                    subset[self.vg.x], subset[y],
                    color=color, s=self.vg.scatter_size,
                    edgecolors=self.vg.scatter_edgecolor, linewidths=0.5,
                    label=str(cat), zorder=3
                )
                
                if anchor:
                    self.vg.ax.plot(x_fit, y_fit, color=color, linewidth=2)
                else:
                    self.vg.ax.plot(x_fit, y_fit, color=color, linewidth=2)
                    x_data = subset[self.vg.x].dropna().values.astype(float)
                    y_data = subset[y].dropna().values.astype(float)
                    if len(x_data) > 2:
                        p = np.poly1d(coeffs)
                        y_model = p(x_data)
                        t_targ = 1.96
                        se = np.sqrt(np.sum((y_data - y_model)**2) / (len(y_data) - 2))
                        ci = t_targ * se * np.sqrt(1/len(x_data) + (x_fit - x_data.mean())**2 / np.sum((x_data - x_data.mean())**2))
                        self.vg.ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color=color, alpha=0.2)
                    
                eq_str, r2 = self._build_equation_str(coeffs, subset)
                self.vg.trendline_equations.append({
                    'label': str(cat), 'equation': eq_str, 'r2': f"{r2:.4f}"
                })
        else:
            x_fit, y_fit, coeffs = self._fit_trendline(df)
            
            self.vg.ax.scatter(
                df[self.vg.x], df[y],
                color=c, s=self.vg.scatter_size,
                edgecolors=self.vg.scatter_edgecolor, linewidths=0.5,
                label='All data', zorder=3
            )
            
            if anchor:
                self.vg.ax.plot(x_fit, y_fit, color=c, linewidth=2)
            else:
                self.vg.ax.plot(x_fit, y_fit, color=c, linewidth=2)
                x_data = df[self.vg.x].dropna().values.astype(float)
                y_data = df[y].dropna().values.astype(float)
                if len(x_data) > 2:
                    p = np.poly1d(coeffs)
                    y_model = p(x_data)
                    t_targ = 1.96
                    se = np.sqrt(np.sum((y_data - y_model)**2) / (len(y_data) - 2))
                    ci = t_targ * se * np.sqrt(1/len(x_data) + (x_fit - x_data.mean())**2 / np.sum((x_data - x_data.mean())**2))
                    self.vg.ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color=c, alpha=0.2)
                
            eq_str, r2 = self._build_equation_str(coeffs, df)
            self.vg.trendline_equations.append({
                'label': 'All data', 'equation': eq_str, 'r2': f"{r2:.4f}"
            })

    def _plot_histogram(self, df, colors):
        from scipy import stats
        plot_df = df.dropna(subset=[self.vg.x])
        bins = self.vg.hist_bins
        hist_step = getattr(self.vg, 'hist_step', False)
        histtype = 'step' if hist_step else 'bar'
        alpha = 1.0 if hist_step else 0.7
        kde = getattr(self.vg, 'hist_kde', False)
        
        hist_kwargs = {'bins': bins, 'histtype': histtype, 'alpha': alpha}
        if histtype == 'bar':
            hist_kwargs['edgecolor'] = 'black'
            hist_kwargs['linewidth'] = 0.8
            
        if self.vg.z and self.vg.z in plot_df.columns:
            categories = self._get_sorted_categories(plot_df[self.vg.z], df=plot_df)
            data_list = []
            labels = []
            c_list = []
            
            for idx, cat in enumerate(categories):
                subset = plot_df[plot_df[self.vg.z] == cat][self.vg.x]
                if not subset.empty:
                    data_list.append(subset.values)
                    labels.append(str(cat))
                    c_list.append(colors[idx % len(colors)])
            
            if data_list:
                self.vg.ax.hist(data_list, color=c_list, label=labels, stacked=False, **hist_kwargs)
                
                if kde:
                    x_min, x_max = self.vg.ax.get_xlim()
                    x_grid = np.linspace(x_min, x_max, 200)
                    for i, data in enumerate(data_list):
                        if len(data) > 1:
                            try:
                                num_data = pd.to_numeric(data, errors='coerce')
                                num_data = num_data[~np.isnan(num_data)]
                                if len(num_data) > 1 and np.var(num_data) > 0:
                                    density = stats.gaussian_kde(num_data)
                                    bin_width = (x_max - x_min) / bins
                                    y_grid = density(x_grid) * len(num_data) * bin_width
                                    self.vg.ax.plot(x_grid, y_grid, color=c_list[i], linewidth=2)
                            except Exception:
                                pass
        else:
            data = plot_df[self.vg.x].values
            if len(data) > 0:
                color = colors[0] if colors else 'steelblue'
                self.vg.ax.hist(data, color=color, label='All data', **hist_kwargs)
                
                if kde and len(data) > 1:
                    try:
                        num_data = pd.to_numeric(data, errors='coerce')
                        num_data = num_data[~np.isnan(num_data)]
                        if len(num_data) > 1 and np.var(num_data) > 0:
                            x_min, x_max = self.vg.ax.get_xlim()
                            x_grid = np.linspace(x_min, x_max, 200)
                            density = stats.gaussian_kde(num_data)
                            bin_width = (x_max - x_min) / bins
                            y_grid = density(x_grid) * len(num_data) * bin_width
                            self.vg.ax.plot(x_grid, y_grid, color=color, linewidth=2)
                    except Exception:
                        pass

    def _plot_2dmap(self, df, y):
        """Plot 2D heatmap."""
        x_col = self.vg.x
        y_col = y if isinstance(self.vg.y, list) else self.vg.y
        z_col = self.vg.z
        
        xmin = df[x_col].min()
        xmax = df[x_col].max()
        ymin = df[y_col].min()
        ymax = df[y_col].max()
        
        heatmap_data = df.pivot(index=y_col, columns=x_col, values=z_col)
        vmin = self.vg.zmin if self.vg.zmin else heatmap_data.min().min()
        vmax = self.vg.zmax if self.vg.zmax else heatmap_data.max().max()
        
        heatmap = self.vg.ax.imshow(
            heatmap_data,
            aspect='equal',
            extent=[xmin, xmax, ymin, ymax],
            cmap=self.vg.color_palette,
            origin='lower',
            vmin=vmin,
            vmax=vmax
        )
        
        # Remove existing colorbar if present to prevent accumulation
        if hasattr(self.vg.ax, '_2dmap_colorbar') and self.vg.ax._2dmap_colorbar is not None:
            try:
                self.vg.ax._2dmap_colorbar.remove()
            except:
                pass
        
        # Add new colorbar and store reference
        self.vg.ax._2dmap_colorbar = plt.colorbar(heatmap, orientation='vertical')

    def _fit_trendline(self, df):
        """Fit polynomial trendline with optional anchor constraint.
        
        Returns (x_fit, y_fit, coefficients).
        """
        try:
            x_data = df[self.vg.x].dropna().values.astype(float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not convert values in X column '{self.vg.x}' to numeric values. "
                "Trendline fitting requires numeric columns."
            ) from e
        
        try:
            y_data = df[self.vg.y[0]].dropna().values.astype(float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not convert values in Y column '{self.vg.y[0]}' to numeric values. "
                "Trendline fitting requires numeric columns."
            ) from e
        
        # Align lengths after dropna
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if self.vg.trendline_anchor_enabled:
            # Determine anchor coordinates
            if self.vg.trendline_anchor_origin:
                ax_val, ay_val = 0.0, 0.0
            else:
                ax_val = float(self.vg.trendline_anchor_x)
                ay_val = float(self.vg.trendline_anchor_y)
            
            # Shift data so anchor becomes origin, then fit without intercept
            x_shifted = x_data - ax_val
            y_shifted = y_data - ay_val
            
            if self.vg.trendline_order == 1:
                # Force through shifted origin: y = m*x
                m = np.dot(x_shifted, y_shifted) / np.dot(x_shifted, x_shifted)
                coeffs = np.array([m, 0.0])  # slope, zero intercept (shifted)
                # Build coefficients back in original space: y = m*(x-ax)+ay
                # Represent as standard polyfit form shifted back:
                x_fit = np.linspace(x_data.min(), x_data.max(), 200)
                y_fit = m * (x_fit - ax_val) + ay_val
            else:
                # Higher order: fit shifted data with zero constant term (no intercept)
                # Use least squares with Vandermonde matrix excluding constant column
                A = np.column_stack([x_shifted**i for i in range(self.vg.trendline_order, 0, -1)])
                result = np.linalg.lstsq(A, y_shifted, rcond=None)
                poly_coeffs = result[0]
                x_fit = np.linspace(x_data.min(), x_data.max(), 200)
                x_fit_shifted = x_fit - ax_val
                y_fit = sum(poly_coeffs[i] * x_fit_shifted**(self.vg.trendline_order - i)
                            for i in range(self.vg.trendline_order)) + ay_val
                coeffs = np.append(poly_coeffs, 0.0)  # zero constant (shifted origin)
        else:
            # Standard unconstrained polynomial fit
            coeffs = np.polyfit(x_data, y_data, self.vg.trendline_order)
            x_fit = np.linspace(x_data.min(), x_data.max(), 200)
            y_fit = np.polyval(coeffs, x_fit)
        
        return x_fit, y_fit, coeffs

    def _build_equation_str(self, coeffs, df):
        """Build human-readable equation string and compute R²."""
        x_data = df[self.vg.x].dropna().values.astype(float)
        y_data = df[self.vg.y[0]].dropna().values.astype(float)
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        order = self.vg.trendline_order
        
        # Build equation string from coefficients (highest power first)
        _sup = {2: '\u00b2', 3: '\u00b3', 4: '\u2074', 5: '\u2075', 6: '\u2076', 7: '\u2077', 8: '\u2078', 9: '\u2079'}
        terms = []
        for i, c in enumerate(coeffs):
            power = order - i
            if power == 0:
                terms.append(f"{c:+.4f}")
            elif power == 1:
                terms.append(f"{c:+.4f}x")
            else:
                sup = _sup.get(power, f"^{power}")
                terms.append(f"{c:+.4f}x{sup}")
        eq_str = "y = " + " ".join(terms).lstrip("+")
        
        # Compute R²
        y_pred = np.polyval(coeffs, x_data)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        
        return eq_str, r2

    def _plot_wafer(self, df):
        """Plot wafer plot by creating an object of WaferPlot Class."""
        vmin = self.vg.zmin if self.vg.zmin else None
        vmax = self.vg.zmax if self.vg.zmax else None
        
        wdf = WaferPlot()
        wdf.plot(
            self.vg.ax,
            x=df[self.vg.x],
            y=df[self.vg.y[0]],
            z=df[self.vg.z],
            cmap=self.vg.color_palette,
            vmin=vmin,
            vmax=vmax,
            stats=self.vg.wafer_stats,
            r=(self.vg.wafer_size / 2)
        )
        
        # Annotate slot number if active filter
        if hasattr(self.vg, "filters") and isinstance(self.vg.filters, (list, dict)):
            filters_list = self.vg.filters if isinstance(self.vg.filters, list) else self.vg.filters.get("filters", [])
            for f in filters_list:
                expr = f.get("expression", "")
                state = f.get("state", False)
                if state and "Slot ==" in expr:
                    try:
                        slot_num = expr.split("==")[1].strip()
                        self.vg.ax.text(
                            0.02, 0.98, f"Slot {slot_num}",
                            transform=self.vg.ax.transAxes,
                            fontsize=12, color='black',
                            fontweight='bold',
                            verticalalignment='top',
                            horizontalalignment='left'
                        )
                    except Exception:
                        pass
                    break


class WaferPlot:
    """Class to plot wafer map."""
    
    def __init__(self, inter_method='linear'):
        self.inter_method = inter_method
    
    def plot(self, ax, x, y, z, cmap="jet", r=100, vmax=None, vmin=None, stats=True):
        """Plot wafer map with interpolated data."""
        xi, yi = np.meshgrid(np.linspace(-r, r, 600), np.linspace(-r, r, 600))
        from scipy.interpolate import griddata
        zi = griddata((x, y), z, (xi, yi), method=self.inter_method)
        
        im = ax.imshow(
            zi,
            extent=[-r - 1, r + 1, -r - 0.5, r + 0.5],
            origin='lower',
            cmap=cmap,
            interpolation='nearest', zorder=2
        )
        
        ax.scatter(x, y, facecolors='none', edgecolors='black', s=20, zorder=3)
        
        wafer_circle = patches.Circle((0, 0), radius=r, fill=False, color='black', linewidth=1)
        ax.add_patch(wafer_circle)
        
        ax.set_ylabel("Wafer size (mm)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', which='both', right=False, left=True)
        ax.set_xticklabels([])
        
        if vmax is not None and vmin is not None:
            im.set_clim(vmin, vmax)
        
        # Remove existing colorbar if present to prevent accumulation
        if hasattr(ax, '_wafer_colorbar') and ax._wafer_colorbar is not None:
            try:
                ax._wafer_colorbar.remove()
            except:
                pass
        
        # Add new colorbar and store reference
        ax._wafer_colorbar = plt.colorbar(im, ax=ax)
        
        if stats:
            self.stats(z, ax)
    
    def stats(self, z, ax):
        """Calculate and display statistical values in the wafer plot."""
        mean_value = z.mean()
        max_value = z.max()
        min_value = z.min()
        sigma_value = z.std()
        three_sigma_value = 3 * sigma_value
        
        ax.text(0.05, -0.1, f"3\u03C3: {three_sigma_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
        ax.text(0.99, -0.1, f"Max: {max_value:.2f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right')
        ax.text(0.99, -0.05, f"Min: {min_value:.2f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right')
        ax.text(0.05, -0.05, f"Mean: {mean_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
