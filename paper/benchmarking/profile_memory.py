import time
import tracemalloc
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from spectroview.model.m_io import load_map_file, load_wdf_map
from spectroview.model.workspace_io import WorkspaceIO
from spectroview.model.spectra_store import SpectraStore
from spectroview.fit_engine.vbf_engine import VBFengine
from spectroview.viewmodel.utils import closest_index
from spectroview.fit_engine.baseline import eval_baseline_batch

def get_xy_from_map(df: pd.DataFrame):
    """Extract X array and Y matrix from map DataFrame."""
    wavenumbers = [float(col) for col in df.columns[2:]]
    x = np.array(wavenumbers, dtype=np.float64)
    Y = df.iloc[:, 2:].to_numpy(dtype=np.float64)
    return x, Y

def load_fit_model(json_path: Path) -> dict:
    """Load fit model configuration from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if "0" in data:
        return data["0"]
    return data

def run_benchmark(name, data_path, json_path):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")
    
    print("Loading data...", flush=True)
    # 1. Load data
    if data_path.suffix == '.maps':
        metadata, arrays, dataframes, is_legacy = WorkspaceIO.load_workspace(str(data_path))
        store = SpectraStore()
        store_meta = metadata.get('store_meta', {})
        for map_name, meta in store_meta.items():
            SpectraStore.load_map_from_npz(arrays, meta, map_name, store=store)
        
        Y_parts = []
        x = None
        for map_name in store.map_names:
            md = store.get_map_data(map_name)
            Y_parts.append(md.Y0)
            if x is None:
                x = md.x0
        Y = np.vstack(Y_parts)
    elif data_path.suffix == '.wdf':
        map_df, meta = load_wdf_map(data_path)
        x, Y = get_xy_from_map(map_df)
    else:
        map_df = load_map_file(data_path)
        x, Y = get_xy_from_map(map_df)
        
    # 2. Load model
    fit_model = load_fit_model(json_path)
    fit_params = fit_model.get("fit_params", {})

    print(f"Raw Data shape: {Y.shape} (spectra x points)", flush=True)
    M_initial = Y.shape[1]

    # 3. Apply PREPROCESSING (Cropping & Baseline) exactly like the app does!
    # Cropping
    xmin = fit_model.get("range_min")
    xmax = fit_model.get("range_max")
    if xmin is not None and xmax is not None:
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        i_min = closest_index(x, xmin)
        i_max = closest_index(x, xmax)
        if i_min > i_max:
            i_min, i_max = i_max, i_min
        
        x = x[i_min:i_max+1].copy()
        Y = Y[:, i_min:i_max+1].copy()
    
    print(f"Cropped Data shape: {Y.shape} (spectra x points)", flush=True)

    # Baseline evaluation and subtraction
    bl_info = fit_model.get("baseline")
    if bl_info and bl_info.get("mode"):
        Y_baseline = eval_baseline_batch(x, Y, bl_info)
        if bl_info.get("is_subtracted", False):
            Y = Y - Y_baseline

    Y_sub = Y
    weights = np.ones_like(Y_sub)
    
    # We construct the noise filtering exactly as VBFthread does to mirror the app exactly
    fit_negative = bool(fit_params.get("fit_negative", False))
    if not fit_negative: 
        weights[Y_sub < 0] = 0.0

    engine = VBFengine()

    print("Starting fit engine and profiling...", flush=True)
    
    # Force Garbage Collection before profiling
    gc.collect()
    
    # 4. Start tracemalloc and timer
    tracemalloc.start()
    start_time = time.time()

    def progress_callback(c, t):
        pass

    # 5. Execute fitting
    p_full, success, rsquared, best_fits, Y_peaks, param_names = engine.fit_spectra(
        x=x,
        Y=Y_sub,
        fit_model=fit_model,
        weights=weights,
        fit_params=fit_params,
        progress_callback=progress_callback,
        cancel_check=lambda: False
    )

    end_time = time.time()
    
    # 6. Take memory snapshot
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n--- Profiling Results ---", flush=True)
    N = Y_sub.shape[0]
    M_cropped = Y_sub.shape[1]
    K = len(param_names)
    tensor_size = N * M_cropped * K
    print(f"Initial Points per Spectrum: {M_initial}")
    print(f"Cropped Points per Spectrum: {M_cropped}")
    print(f"Tensor Dimensions: N={N} (spectra) x M={M_cropped} (points) x K={K} (params)", flush=True)
    print(f"Total Tensor Size (N * M * K): {tensor_size:,} elements", flush=True)
    print(f"Execution time: {end_time - start_time:.2f} seconds", flush=True)
    print(f"Current memory usage: {current_mem / 10**6:.2f} MB", flush=True)
    print(f"Peak memory usage: {peak_mem / 10**6:.2f} MB", flush=True)
    print(f"Fits converged: {np.sum(success)}/{len(Y_sub)}", flush=True)
    
    good_r2 = rsquared[success]
    if len(good_r2) > 0:
        print(f"Mean R2 (converged): {np.mean(good_r2):.4f}", flush=True)


def main():
    EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples"
    DATA_DIR = EXAMPLES_DIR / "fit_benchmarking_data"

    datasets = [
        {
            "name": "1. MoS2 flake",
            "data": DATA_DIR / "2_MoS2_map.txt",
            "json": DATA_DIR / "2_fit_MoS2map_NEW.json"
        },
        {
            "name": "2. Array of Pixel (WDF)",
            "data": DATA_DIR / "3_3721map.wdf",
            "json": DATA_DIR / "3_3721map.json"
        },
        {
            "name": "3. 4 Wafer Maps (Project)",
            "data": EXAMPLES_DIR / "wafers_NEW.maps",
            "json": DATA_DIR / "5_MoS2_wafers_NEW.json"
        },
        {
            "name": "4. CL Map",
            "data": DATA_DIR / "1_CL_map.txt",
            "json": DATA_DIR / "1_CL_map.json"
        }
    ]

    for ds in datasets:
        if not ds["data"].exists() or not ds["json"].exists():
            print(f"Skipping {ds['name']} because files are missing.")
            print(f"  Data: {ds['data']} (Exists: {ds['data'].exists()})")
            print(f"  JSON: {ds['json']} (Exists: {ds['json'].exists()})")
            continue
            
        run_benchmark(ds["name"], ds["data"], ds["json"])

if __name__ == "__main__":
    main()
