"""Example: full hyperspectral-map workflow with spectroview.api.

Load -> fit every pixel -> collect results -> heatmap -> save.

Run from the repository root:
    python examples/api_workflow_map.py

The output .maps file can be opened directly in the SPECTROview GUI:
    spectroview examples/api_workflow_map_output.maps
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed to run this script
import matplotlib.pyplot as plt

from spectroview.api import fitting, workspace

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "fit_benchmarking_data"


def main():
    # 1. Load a hyperspectral map into a stateful, GUI-file-compatible session.
    ws = workspace.MapsWorkspace()
    [map_name] = ws.load_files([DATA_DIR / "2Dmap_Si.txt"])
    print(f"Loaded map '{map_name}' with {ws.store.get_map_data(map_name).n_spectra} pixels")

    # 2. Load a fit-model template exported from the GUI, and fit every pixel
    #    in one vectorized call.
    fit_model = fitting.load_fit_model_template(DATA_DIR / "predefined_fit_models" / "fit_model_Si_.json")
    ws.set_fit_model(fit_model, names=[map_name])
    ws.fit(map_names=[map_name])

    # 3. Collect per-pixel fit results.
    df = ws.collect_results()
    print(df.head())

    # 4. Build a heatmap of the fitted amplitude and save it as an image.
    value_col = [c for c in df.columns if "ampli" in c][0]
    xi, yi, zi = ws.get_heatmap(map_name, value_col)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin="lower", cmap="jet")
    fig.colorbar(im, ax=ax, label=value_col)
    ax.set_title(f"{map_name}: {value_col}")
    plt.tight_layout()
    plot_path = HERE / "api_workflow_map_output.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved heatmap to {plot_path}")

    # 5. Save the session — this file opens directly in the SPECTROview GUI.
    out_path = HERE / "api_workflow_map_output.maps"
    ws.save(out_path)
    print(f"Saved workspace to {out_path}")


if __name__ == "__main__":
    main()
