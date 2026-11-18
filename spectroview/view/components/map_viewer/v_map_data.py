"""Heatmap data preparation, caching, masking, and profile interpolation for MapViewer."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from spectroview.view.components.map_viewer.map_selection import compute_2d_spatial_geometry


class MapDataProcessingMixin:
    """Data slicing, griddata caching, masking, outlier removal, and profile extraction."""

    def clear_cache_for_map(self, map_name: str):
        """Clear cached griddata for specific map (e.g., after fitting changes data).

        Args:
            map_name: Name of the map whose cache entries should be cleared
        """
        keys_to_remove = [k for k in self._griddata_cache if k[0] == map_name]
        for key in keys_to_remove:
            del self._griddata_cache[key]

    def _build_cache_key(self):
        """Construct cache key for the current heatmap configuration."""
        xmin, xmax = self.x_range_slider.value()
        parameter = self.cbb_zparameter.currentText()
        map_type = self.cbb_map_type.currentText()
        mask_enabled = self.cb_enable_mask.isChecked() if map_type == '2Dmap' else False
        remove_outliers = self.action_remove_outlier.isChecked()

        if mask_enabled:
            mask_config = (
                self.cbb_mask_param.currentText(),
                self.cbb_mask_operator.currentText(),
                self.spin_mask_threshold.value()
            )
        else:
            mask_config = None

        return (
            self.map_df_name,
            parameter,
            (xmin, xmax),
            map_type,
            mask_config,
            remove_outliers
        )

    def _get_data_for_heatmap(self):
        """Compute heatmap data from map DataFrame (with caching for wafer griddata).

        Returns:
            tuple: (heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col)
        """
        cache_key = self._build_cache_key()

        # Check cache first (avoid expensive griddata computation!)
        if cache_key in self._griddata_cache:
            cached = self._griddata_cache[cache_key]
            if len(cached) >= 8:
                self._current_dx = float(cached[6])
                self._current_dy = float(cached[7])
            return cached[:6]

        # Default returns
        heatmap_pivot = pd.DataFrame()
        extent = [0, 0, 0, 0]
        vmin, vmax = 0, 100
        grid_z = None
        final_z_col = None

        if self.map_df is None or self.map_df.empty:
            self._current_dx = 1.0
            self._current_dy = 1.0
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col

        # Get X-range filter
        xmin, xmax = self.x_range_slider.value()
        column_labels = self.map_df.columns[2:]

        try:
            numeric_cols = column_labels.astype(float)
            col_mask = (numeric_cols >= xmin) & (numeric_cols <= xmax)
            filtered_columns = column_labels[col_mask]
        except Exception:
            filtered_columns = column_labels

        if len(filtered_columns) == 0:
            self._current_dx = 1.0
            self._current_dy = 1.0
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col

        # Coordinates (cached in set_map_data)
        x_col, y_col = self._get_measurement_sites()

        # Compute Z values based on selected parameter
        parameter = self.cbb_zparameter.currentText()

        if parameter == 'Area':
            data_matrix = self.map_df[filtered_columns].to_numpy(dtype=float, copy=False)
            data_clean = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(data_clean, 0, None, out=data_clean)
            z_col = data_clean.sum(axis=1)
        elif parameter == 'Intensity':
            data_matrix = self.map_df[filtered_columns].to_numpy(dtype=float, copy=False)
            data_clean = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(data_clean, 0, None, out=data_clean)
            z_col = data_clean.max(axis=1)
        else:
            # Fit parameter - merge with map coordinates to handle filtered (active only) fit results
            if not self.df_fit_results.empty and parameter in self.df_fit_results.columns:
                filtered_results = self.df_fit_results[self.df_fit_results['Filename'] == self.map_df_name]
                if not filtered_results.empty and 'X' in filtered_results.columns and 'Y' in filtered_results.columns:
                    map_coords_temp = pd.DataFrame({
                        'X_round': np.round(x_col, 6),
                        'Y_round': np.round(y_col, 6)
                    })
                    filtered_results_temp = pd.DataFrame({
                        'X_round': np.round(filtered_results['X'].values.astype(float), 6),
                        'Y_round': np.round(filtered_results['Y'].values.astype(float), 6),
                        parameter: filtered_results[parameter].values
                    })

                    merged = map_coords_temp.merge(
                        filtered_results_temp,
                        on=['X_round', 'Y_round'], how='left'
                    )
                    # Fill missing values (unchecked spectra) with 0 ONLY for amplitude, area, height, or intensity related parameters.
                    if any(x in parameter.lower() for x in ['ampli', 'area', 'intensity', 'height', 'fraction', 'A', 'B']):
                        z_col = merged[parameter].fillna(0).values
                    else:
                        z_col = merged[parameter].values
                else:
                    z_col = np.full(len(x_col), np.nan)
            else:
                z_col = np.full(len(x_col), np.nan)

        if z_col is None or len(z_col) == 0:
            self._current_dx = 1.0
            self._current_dy = 1.0
            return heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col

        # Apply mask (2D map only)
        map_type = self.cbb_map_type.currentText()
        if map_type == '2Dmap' and self.cb_enable_mask.isChecked():
            z_col = self._apply_mask(z_col, self.map_df)

        # Remove outliers if requested
        if self.action_remove_outlier.isChecked():
            z_col = self._remove_outliers(z_col)

        final_z_col = z_col

        # Compute vmin/vmax
        try:
            valid_z = final_z_col[~np.isnan(final_z_col)]
            if len(valid_z) > 0:
                vmin = float(valid_z.min())
                vmax = float(valid_z.max())
            else:
                vmin, vmax = 0.0, 100.0
            if np.isnan(vmin) or np.isnan(vmax):
                vmin, vmax = 0.0, 100.0
        except Exception:
            vmin, vmax = 0.0, 100.0

        # Different approach for wafer vs 2D maps
        if map_type != '2Dmap' and self.number_of_points >= 4:
            # Wafer maps: Use griddata for scattered point interpolation
            r = self._get_wafer_radius(map_type)
            if r:
                grid_size = 80
                grid_x, grid_y = np.meshgrid(
                    np.linspace(-r, r, grid_size),
                    np.linspace(-r, r, grid_size)
                )

                # Filter out NaN values before interpolation (masked points)
                # This prevents interpolation from filling in masked regions
                valid_mask = ~pd.isna(final_z_col)
                if valid_mask.any():
                    x_valid = x_col[valid_mask]
                    y_valid = y_col[valid_mask]
                    z_valid = final_z_col[valid_mask]

                    # Linear interpolation for smooth result (only on valid points)
                    from scipy.interpolate import griddata
                    grid_z = griddata((x_valid, y_valid), z_valid,
                                     (grid_x, grid_y), method='linear')
                else:
                    grid_z = np.full_like(grid_x, np.nan)

                half_step = r / (grid_size - 1)
                extent = [-r - half_step, r + half_step, -r - half_step, r + half_step]
                heatmap_pivot = pd.DataFrame()  # Not used for wafer
                width = extent[1] - extent[0]
                height = extent[3] - extent[2]
                dx = width / float(grid_size)
                dy = height / float(grid_size)
                self._current_dx = float(dx)
                self._current_dy = float(dy)
        else:
            geom = self._get_2d_spatial_geometry()
            dx = geom['dx']
            dy = geom['dy']
            extent = geom['extent']
            self._current_dx = float(dx)
            self._current_dy = float(dy)

            if geom['is_regular']:
                # Regular 2D maps: Use fast pivot table on uniform grid
                heatmap_data = pd.DataFrame({'X': x_col, 'Y': y_col, 'Z': final_z_col})
                if geom['has_duplicates']:
                    heatmap_pivot = heatmap_data.pivot_table(
                        index='Y', columns='X', values='Z', aggfunc='mean'
                    )
                else:
                    heatmap_pivot = heatmap_data.pivot(index='Y', columns='X', values='Z')
                grid_z = None
            else:
                # Irregular / scattered 2D map (e.g. wafer files or non-uniform coordinates in 2Dmap mode):
                # Avoid pivot + imshow which distorts coordinates by assuming uniform row/col spacing.
                heatmap_pivot = pd.DataFrame()
                grid_z = None

        dx_val = getattr(self, '_current_dx', 1.0)
        dy_val = getattr(self, '_current_dy', 1.0)
        result = (heatmap_pivot, extent, vmin, vmax, grid_z, final_z_col, float(dx_val), float(dy_val))

        # Cache the result to avoid recomputing expensive griddata
        self._griddata_cache[cache_key] = result

        return result[:6]

    def _get_2d_spatial_geometry(self) -> dict:
        """Compute and cache 2D spatial grid regularity, pixel size (dx, dy), and extent for current sites."""
        cached = getattr(self, '_cached_2d_geometry', None)
        if cached is not None:
            return cached

        x_col, y_col = self._get_measurement_sites()
        geom = compute_2d_spatial_geometry(x_col, y_col)
        self._cached_2d_geometry = geom
        return geom

    def _apply_mask(self, z_col, filtered_df):
        """Apply mask to z_col based on mask settings."""
        mask_param = self.cbb_mask_param.currentText()
        mask_operator = self.cbb_mask_operator.currentText()
        threshold = self.spin_mask_threshold.value()

        mask_values = None

        # Try to get mask values from map DataFrame
        if mask_param in filtered_df.columns:
            mask_values = filtered_df[mask_param].astype(float).values
        # Or from fit results
        elif not self.df_fit_results.empty and mask_param in self.df_fit_results.columns:
            filtered_results = self.df_fit_results[self.df_fit_results['Filename'] == self.map_df_name]
            if not filtered_results.empty:
                mask_values = filtered_results[mask_param].astype(float).values

        if mask_values is None:
            return z_col

        # Apply operator
        if mask_operator == ">":
            valid = mask_values > threshold
        elif mask_operator == "<":
            valid = mask_values < threshold
        elif mask_operator == ">=":
            valid = mask_values >= threshold
        elif mask_operator == "<=":
            valid = mask_values <= threshold
        elif mask_operator == "==":
            valid = mask_values == threshold
        else:
            valid = np.ones_like(mask_values, dtype=bool)

        # Set invalid points to NaN
        return np.where(valid, z_col, np.nan)

    def _remove_outliers(self, z_col):
        """Remove outliers using IQR method with interpolation."""
        try:
            s = pd.Series(z_col)
            Q1 = s.quantile(0.05)
            Q3 = s.quantile(0.95)
            IQR = Q3 - Q1

            outlier_mask = (s < (Q1 - 1.5 * IQR)) | (s > (Q3 + 1.5 * IQR))

            s_interpolated = s.copy()
            s_interpolated[outlier_mask] = np.nan
            s_interpolated = s_interpolated.interpolate(
                method='linear', limit_direction='both'
            )
            return s_interpolated.to_numpy()
        except Exception:
            return z_col

    def _get_wafer_radius(self, map_type_text):
        """Extract wafer radius from map type string."""
        match = re.search(r'Wafer_(\d+)mm', map_type_text)
        if match:
            diameter = int(match.group(1))
            return diameter / 2
        return None

    def _get_measurement_sites(self):
        """Get all measurement site coordinates."""
        if hasattr(self, '_sites_x') and self._sites_x is not None and len(self._sites_x) > 0:
            return self._sites_x, self._sites_y
        if self.map_df is None or self.map_df.empty:
            return np.array([]), np.array([])
        return self.map_df['X'].values, self.map_df['Y'].values

    def _extract_profile(self) -> pd.DataFrame | None:
        """Extract a profile from 2D map plot via interpolation.

        Returns:
            DataFrame with columns: X, Y, distance, values
            or None if extraction fails
        """
        if len(self.selected_points) != 2:
            return None

        (x1, y1), (x2, y2) = self.selected_points
        heatmap_pivot, _, _, _, grid_z, final_z_col = self._get_data_for_heatmap()
        num_samples = 100
        x_samples = np.linspace(x1, x2, num_samples)
        y_samples = np.linspace(y1, y2, num_samples)
        dists_from_start = np.sqrt((x_samples - x1) ** 2 + (y_samples - y1) ** 2)

        if not heatmap_pivot.empty:
            x_values = heatmap_pivot.columns.values.astype(float)
            y_values = heatmap_pivot.index.values.astype(float)
            z_values = heatmap_pivot.values

            from scipy.interpolate import RegularGridInterpolator
            try:
                interpolator = RegularGridInterpolator((y_values, x_values), z_values)
                sample_points = np.column_stack((y_samples, x_samples))
                z_samples = interpolator(sample_points)
                return pd.DataFrame({
                    'X': x_samples,
                    'Y': y_samples,
                    'distance': dists_from_start,
                    'values': z_samples
                })
            except Exception as e:
                print(f"Profile interpolation failed: {e}")
                return None
        else:
            x_col, y_col = self._get_measurement_sites()
            if len(x_col) == 0 or final_z_col is None:
                return None
            valid = ~np.isnan(final_z_col)
            if not valid.any():
                return None
            from scipy.interpolate import griddata
            try:
                z_samples = griddata(
                    (x_col[valid], y_col[valid]), final_z_col[valid],
                    (x_samples, y_samples), method='linear'
                )
                if np.isnan(z_samples).all():
                    z_samples = griddata(
                        (x_col[valid], y_col[valid]), final_z_col[valid],
                        (x_samples, y_samples), method='nearest'
                    )
                return pd.DataFrame({
                    'X': x_samples,
                    'Y': y_samples,
                    'distance': dists_from_start,
                    'values': z_samples
                })
            except Exception as e:
                print(f"Profile interpolation failed: {e}")
                return None
