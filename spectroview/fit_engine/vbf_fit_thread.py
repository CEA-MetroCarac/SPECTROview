"""QThread wrapper for the Batch Fitting Engine."""
import time
import numpy as np
import sys
from PySide6.QtCore import QThread, Signal
from spectroview.fit_engine.vbf_fit_engine import VBFengine

class VBFthread(QThread):
    progress_changed = Signal(int, int, int, float)
    timings_ready = Signal(str)

    def __init__(self, store, tasks: list[dict]):
        super().__init__()
        if sys.platform == "darwin":
            self.setStackSize(8 * 1024 * 1024)
            
        self.store = store
        self.tasks = tasks
        self._is_cancelled = False

    def stop(self):
        self._is_cancelled = True

    def run(self):
        t_start = time.perf_counter()
        
        # Calculate total spectra to process across all tasks
        total_spectra = sum(len(task["indices"]) for task in self.tasks)
        processed_spectra = 0
        
        timings = []

        for task in self.tasks:
            if self._is_cancelled:
                break
                
            map_name = task["map_name"]
            indices = task["indices"]
            fit_model = task["fit_model"]
            
            md = self.store.get_map_data(map_name)
            if md is None:
                continue
                
            x = md.x if md.x is not None else md.x0
            Y = md.Y if md.Y is not None else md.Y0
            
            Y_sub = Y[indices]
            N = len(indices)
            
            fit_params = fit_model.get("fit_params", {})
            
            # Construct vectorized fit weights (negative-masking and noise-masking)
            N_fit, M_fit = Y_sub.shape
            weights = np.ones((N_fit, M_fit), dtype=np.float64)
            
            fit_negative = bool(fit_params.get("fit_negative", False))
            if not fit_negative:
                weights[Y_sub < 0] = 0.0
                
            coef_noise = float(fit_params.get("coef_noise", 0))
            if coef_noise > 0:
                dy = np.diff(Y_sub, axis=1)
                ampli_noise = np.median(np.abs(dy), axis=1) / 0.6745 * np.sqrt(2) # (N,)
                Y_padded = np.pad(Y_sub, ((0, 0), (2, 2)), mode='edge')
                ymean = (Y_padded[:, 0:-4] + Y_padded[:, 1:-3] + Y_padded[:, 2:-2] + Y_padded[:, 3:-1] + Y_padded[:, 4:]) / 5.0
                noise_level = coef_noise * ampli_noise # (N,)
                weights[ymean < noise_level[:, None]] = 0.0
            
            def on_progress(current, total):
                elapsed = time.perf_counter() - t_start
                # Map task progress to overall progress
                overall_current = processed_spectra + current
                pct = int(100 * overall_current / max(total_spectra, 1))
                self.progress_changed.emit(overall_current, total_spectra, pct, elapsed)

            engine = VBFengine()
            try:
                p_full, success, rsquared, best_fits, Y_peaks, param_names = engine.fit_spectra(
                    x=x,
                    Y=Y_sub,
                    fit_model=fit_model,
                    weights=weights,
                    fit_params=fit_params,
                    progress_callback=on_progress,
                    cancel_check=lambda: self._is_cancelled,
                )
            except Exception as e:
                print(f"[VBFthread] fit failed for {map_name}: {e}")
                processed_spectra += N
                continue
                
            if not self._is_cancelled:
                self.store.set_fit_results(
                    map_name=map_name,
                    indices=indices,
                    peak_params=p_full,
                    success=success,
                    r2=rsquared,
                    param_names=param_names,
                    fit_model=fit_model,
                )
                
                if md.Y_bestfit is None:
                    md.Y_bestfit = np.zeros_like(Y)
                if md.Y_baseline is None:
                    md.Y_baseline = np.zeros_like(Y)
                
                md.Y_bestfit[indices] = best_fits
                
                if Y_peaks:
                    if md.Y_peaks is None or len(md.Y_peaks) != len(Y_peaks):
                        md.Y_peaks = [np.zeros_like(Y) for _ in Y_peaks]
                    for p_idx, p_arr in enumerate(Y_peaks):
                        md.Y_peaks[p_idx][indices] = p_arr
                        
                timings.append(f"[{map_name}] Fit time: {time.perf_counter() - t_start:.2f}s")
                
            processed_spectra += N
            
        elapsed = time.perf_counter() - t_start
        self.progress_changed.emit(total_spectra, total_spectra, 100, elapsed)
        
        timings_str = f"Total Fit time: {elapsed:.2f}s\n" + "\n".join(timings)
        self.timings_ready.emit(timings_str)
