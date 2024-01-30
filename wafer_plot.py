import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import griddata


class WaferPlot:
    """ Class to plot wafer map """

    def __init__(self, wafer_df, wafer_size, margin, hue=None,
                 inter_method='linear'):
        self.wafer_df = wafer_df
        self.wafer_size = wafer_size
        self.margin = margin
        self.hue = hue
        self.inter_method = inter_method  # interpolation method

    def plot(self, ax, spec, stats=True):
        # Extract X, Y, and parameter columns
        x = self.wafer_df[spec.get("selected_x_column")]
        y = self.wafer_df[spec.get("selected_y_column")]
        hue = self.wafer_df[self.hue]

        radius = (self.wafer_size / 2)

        wafer_name = spec.get("wafer_name")

        # Generate a meshgrid for the wafer
        xi, yi = np.meshgrid(
            np.linspace(-radius, radius, 100),
            np.linspace(-radius, radius, 100))

        # Interpolate hue onto the meshgrid
        zi = self.interpolate_data(x, y, hue, xi, yi)

        # Plot the wafer map
        im = ax.imshow(zi, extent=[-radius - 1, radius + 1, -radius - 0.5,
                                   radius + 0.5],
                       origin='lower', cmap=spec["palette_colors"],
                       interpolation='nearest')
        # Add open circles corresponding to measurement points
        ax.scatter(x, y, facecolors='none', edgecolors='black', s=30)

        # Add a circle as a decoration
        wafer_circle = patches.Circle((0, 0), radius=radius, fill=False,
                                      color='black', linewidth=1)
        ax.add_patch(wafer_circle)

        ax.set_ylabel("Wafer size (mm)")

        # Remove unnescessary axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', which='both', right=False, left=True)
        ax.set_xticklabels([])

        # set hue as plot title
        label = spec["selected_hue_column"] if not spec["hueaxis_title"] else \
            spec["hueaxis_title"]

        spec["plot_title"] = label
        ax.set_title(spec["plot_title"])
        plt.colorbar(im, ax=ax)

        if wafer_name:
            ax.text(0.02, 0.98, f"{wafer_name}",
                    transform=ax.transAxes, fontsize=13, fontweight='bold',
                    verticalalignment='top', horizontalalignment='left')

        if stats:
            self.stats(hue, ax)

    def stats(self, hue, ax):
        """ to calculate and display statistical values within wafer plot"""
        # Calculate statistical values
        mean_value = hue.mean()
        max_value = hue.max()
        min_value = hue.min()
        sigma_value = hue.std()
        three_sigma_value = 3 * sigma_value

        # Annotate the plot with statistical values
        ax.text(0.05, - 0.1, f"3\u03C3: {three_sigma_value:.2f}",
                transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom')
        ax.text(0.99, - 0.1, f"Max: {max_value:.2f}",
                transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                horizontalalignment='right')
        ax.text(0.99, - 0.05, f"Min: {min_value:.2f}",
                transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                horizontalalignment='right')
        ax.text(0.05, - 0.05, f"Mean: {mean_value:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom')

    def interpolate_data(self, x, y, hue, xi, yi):
        # Interpolate data using griddata
        zi = griddata((x, y), hue, (xi, yi),
                      method=self.inter_method)
        return zi
