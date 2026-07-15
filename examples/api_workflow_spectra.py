"""Example: full discrete-spectra workflow with spectroview.api.

Load -> preprocess -> fit -> collect results -> plot -> save.

Run from the repository root:
    python examples/api_workflow_spectra.py

The output .spectra file can be opened directly in the SPECTROview GUI:
    spectroview examples/api_workflow_spectra_output.spectra
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed to run this script
import matplotlib.pyplot as plt

from spectroview.api import fitting, graphs, workspace

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "fit_benchmarking_data"


def main():
    # 1. Load a few discrete spectra into a stateful, GUI-file-compatible session.
    ws = workspace.SpectraWorkspace()
    ws.load_files([
        DATA_DIR / "spectrum1_1ML.txt",
        DATA_DIR / "spectrum2_1ML.txt",
        DATA_DIR / "spectrum3_3ML.txt",
    ])
    print(ws)

    # 2. Preprocess: crop around the Si peak and remove a smooth background.
    ws.crop(range_min=490.0, range_max=550.0)
    ws.set_baseline({"mode": "Polynomial", "order_max": 1, "attached": False,
                      "points": [[490.0, 550.0], [0.0, 0.0]]})
    ws.subtract_baseline()

    # 3. Build and attach a single-peak fit model, then fit every spectrum.
    fit_model = fitting.build_fit_model(peaks=[
        {"model": "Lorentzian",
         "x0": {"value": 520.0, "min": 515.0, "max": 525.0},
         "ampli": {"value": 1000.0, "min": 0.0, "max": 1e7},
         "fwhm": {"value": 4.0, "min": 0.5, "max": 15.0}},
    ])
    ws.set_fit_model(fit_model)
    ws.fit()

    # 4. Collect fit results into a tidy DataFrame.
    df = ws.collect_results()
    print(df)

    # 5. Plot the fitted peak position for each spectrum.
    fig, ax = plt.subplots(figsize=(6, 4))
    graphs.plot_scatter(df.reset_index(), x="index", y="P1_x0", title="Fitted Si Peak Position")
    plt.tight_layout()
    plot_path = HERE / "api_workflow_spectra_output.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    # 6. Save the session — this file opens directly in the SPECTROview GUI.
    out_path = HERE / "api_workflow_spectra_output.spectra"
    ws.save(out_path)
    print(f"Saved workspace to {out_path}")


if __name__ == "__main__":
    main()
