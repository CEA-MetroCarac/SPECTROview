"""QThread wrapper for the Batch Fitting Engine."""
import time
import numpy as np
import sys
from PySide6.QtCore import QThread, Signal
from spectroview.fit_engine.vbf_engine import VBFengine
from spectroview.fit_engine.weights import compute_fit_weights

class VBFthread(QThread):
    progress_changed = Signal(int, int, int, float, int)
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
        total_spectra = 0
        for task in self.tasks:
            if "map_names" in task:
                total_spectra += len(task["map_names"])
            else:
                total_spectra += len(task["indices"])
        processed_spectra = 0
        
        timings = []

        for task in self.tasks:
            if self._is_cancelled:
                break
                
            map_name = task.get("map_name")
            map_names = task.get("map_names")
            grouped_2d_maps = task.get("grouped_2d_maps")
            indices = task["indices"]
            fit_model = task["fit_model"]
            
            if grouped_2d_maps is not None:
                # Mega-batch: stack all maps into one Y matrix, fit once, scatter back.
                map_boundaries = task["map_boundaries"]  # [(name, start, end), ...]
                md_list = []
                Y_list = []
                x = None

                for name, start, end in map_boundaries:
                    md = self.store.get_map_data(name)
                    if md is None:
                        continue
                    md_list.append(md)
                    x_arr = md.x if md.x is not None else md.x0
                    y_arr = md.Y if md.Y is not None else md.Y0
                    Y_list.append(y_arr)
                    if x is None:
                        x = x_arr

                if not Y_list or x is None:
                    continue

                Y_sub = np.vstack(Y_list)
                N = Y_sub.shape[0]

                weights = self._prepare_weights(Y_sub, fit_model.get("fit_params", {}))
                
                try:
                    p_full, success, rsquared, best_fits, Y_peaks, param_names = self._execute_fit(
                        x, Y_sub, fit_model, weights, total_spectra, processed_spectra, t_start
                    )
                except Exception as e:
                    print(f"[VBFthread] grouped 2D-maps fit failed: {e}")
                    processed_spectra += N
                    continue

                if not self._is_cancelled:
                    md_idx = 0
                    for name, start, end in map_boundaries:
                        if md_idx >= len(md_list): break
                        md = md_list[md_idx]
                        md_idx += 1
                        n_local = end - start
                        
                        self._write_results(
                            md, name, np.arange(n_local),
                            p_full[start:end], success[start:end], rsquared[start:end],
                            best_fits[start:end], [p[start:end] for p in Y_peaks] if Y_peaks else None,
                            param_names, fit_model
                        )
                    timings.append(f"[Grouped 2D Maps] Fit time: {time.perf_counter() - t_start:.2f}s ({N} spectra across {len(map_boundaries)} maps)")
                processed_spectra += N

            elif map_names is not None:
                # Group spectra by x-axis length to safely batch them
                groups = {}
                for name in map_names:
                    md = self.store.get_map_data(name)
                    if md is not None:
                        x_arr = md.x if md.x is not None else md.x0
                        m_len = len(x_arr)
                        if m_len not in groups:
                            groups[m_len] = {"x": x_arr, "Y_list": [], "valid_names": [], "md_list": []}
                        groups[m_len]["Y_list"].append(md.Y[0] if md.Y is not None else md.Y0[0])
                        groups[m_len]["valid_names"].append(name)
                        groups[m_len]["md_list"].append(md)
                        
                for m_len, grp in groups.items():
                    if self._is_cancelled: break
                    
                    x = grp["x"]
                    Y_sub = np.vstack(grp["Y_list"])
                    N = len(grp["valid_names"])
                    
                    weights = self._prepare_weights(Y_sub, fit_model.get("fit_params", {}))
                    
                    try:
                        p_full, success, rsquared, best_fits, Y_peaks, param_names = self._execute_fit(
                            x, Y_sub, fit_model, weights, total_spectra, processed_spectra, t_start
                        )
                    except Exception as e:
                        print(f"[VBFthread] grouped fit failed: {e}")
                        processed_spectra += N
                        continue
                        
                    if not self._is_cancelled:
                        for i, name in enumerate(grp["valid_names"]):
                            md = grp["md_list"][i]
                            self._write_results(
                                md, name, np.array([0]),
                                p_full[i:i+1], success[i:i+1], rsquared[i:i+1],
                                best_fits[i:i+1], [p[i:i+1] for p in Y_peaks] if Y_peaks else None,
                                param_names, fit_model
                            )
                        timings.append(f"[Batch Spectra Fit] Fit time: {time.perf_counter() - t_start:.2f}s ({N} spectra)")
                    processed_spectra += N
            else:
                md = self.store.get_map_data(map_name)
                if md is None:
                    continue
                    
                x = md.x if md.x is not None else md.x0
                Y = md.Y if md.Y is not None else md.Y0
                
                Y_sub = Y[indices]
                N = len(indices)
                
                weights = self._prepare_weights(Y_sub, fit_model.get("fit_params", {}))
                
                try:
                    p_full, success, rsquared, best_fits, Y_peaks, param_names = self._execute_fit(
                        x, Y_sub, fit_model, weights, total_spectra, processed_spectra, t_start
                    )
                except Exception as e:
                    print(f"[VBFthread] fit failed for {map_name}: {e}")
                    processed_spectra += N
                    continue
                    
                if not self._is_cancelled:
                    self._write_results(
                        md, map_name, indices,
                        p_full, success, rsquared, best_fits, Y_peaks,
                        param_names, fit_model
                    )
                    timings.append(f"[{map_name}] Fit time: {time.perf_counter() - t_start:.2f}s")
                    
                processed_spectra += N
            
        elapsed = time.perf_counter() - t_start
        self.progress_changed.emit(total_spectra, total_spectra, 100, elapsed, total_spectra)
        
        timings_str = f"Total Fit time: {elapsed:.2f}s\n" + "\n".join(timings)
        self.timings_ready.emit(timings_str)

    def _prepare_weights(self, Y_sub, fit_params):
        return compute_fit_weights(Y_sub, fit_params)

    def _execute_fit(self, x, Y_sub, fit_model, weights, total_spectra, current_processed, t_start):
        def on_progress(current, total):
            elapsed = time.perf_counter() - t_start
            overall_current = current_processed + current
            pct = int(100 * overall_current / max(total_spectra, 1))
            self.progress_changed.emit(overall_current, total_spectra, pct, elapsed, overall_current)

        engine = VBFengine()
        return engine.fit_spectra(
            x=x, Y=Y_sub, fit_model=fit_model, weights=weights, fit_params=fit_model.get("fit_params", {}),
            progress_callback=on_progress, cancel_check=lambda: self._is_cancelled,
        )

    def _write_results(self, md, map_name, indices, p_full, success, rsquared, best_fits, Y_peaks, param_names, fit_model):
        self.store.set_fit_results(
            map_name=map_name, indices=indices,
            peak_params=p_full,
            success=success,
            r2=rsquared,
            param_names=param_names, fit_model=fit_model,
        )

        y_full = md.Y if md.Y is not None else md.Y0
        if md.Y_bestfit is None:
            md.Y_bestfit = np.zeros_like(y_full)
        if md.Y_baseline is None:
            md.Y_baseline = np.zeros_like(y_full)
        md.Y_bestfit[indices] = best_fits

        if Y_peaks:
            if md.Y_peaks is None or len(md.Y_peaks) != len(Y_peaks):
                md.Y_peaks = [np.zeros_like(y_full) for _ in Y_peaks]
            for p_idx, p_arr in enumerate(Y_peaks):
                md.Y_peaks[p_idx][indices] = p_arr

   