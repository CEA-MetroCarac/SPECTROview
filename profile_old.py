"""Profile the OLD tensor fit engine from commit 1e1d15c."""
import time
import json
import numpy as np

from spectroview.model.m_settings import MSettings
from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
from spectroview.viewmodel.utils import apply_custom_fit_model
from spectroview.fit_engine.evaluator import TensorEvaluator
from spectroview.fit_engine.optimizer import batched_levenberg_marquardt
from spectroview.fit_engine.scalar_models import FitResult

settings = MSettings()
vm = VMWorkspaceMaps(settings)
vm.load_map_files(["examples/2Dmap_MoS2.txt"])
n = len(vm.spectra)
print(f"Loaded {n} spectra")

with open("examples/fit_model2.json", "r") as f:
    fit_model = json.load(f)["0"]

t0 = time.perf_counter()
for s in vm.spectra:
    apply_custom_fit_model(s, fit_model, s.fname)
print(f"apply_model: {time.perf_counter()-t0:.3f}s")

spectra = list(vm.spectra)
fit_params = fit_model.get("fit_params", {})
evaluator = TensorEvaluator.from_fit_model(fit_model)

# preprocess
t0 = time.perf_counter()
for sp in spectra:
    if not getattr(sp, 'is_preprocessed', False):
        sp.preprocess()
print(f"preprocess: {time.perf_counter()-t0:.3f}s")

# build_weights
from spectroview.fit_engine.tensor_engine import TensorFittingEngine
engine = TensorFittingEngine()
t0 = time.perf_counter()
weights_matrix = engine._build_fit_weights(spectra, fit_params)
print(f"build_weights: {time.perf_counter()-t0:.3f}s")

# data matrix
x_array = spectra[0].x
M = len(x_array)
Y_matrix = np.empty((n, M), dtype=np.float64)
for i, s in enumerate(spectra):
    Y_matrix[i] = s.y if s.y is not None and len(s.y) == M else 0.0

# build p0
t0 = time.perf_counter()
p0 = np.empty((n, evaluator.n_params_free))
for i, s in enumerate(spectra):
    p0[i] = evaluator.extract_p0_from_spectrum(s)
print(f"build_p0: {time.perf_counter()-t0:.3f}s")

# noise_threshold 1
t0 = time.perf_counter()
evaluator.apply_noise_threshold(spectra, p0, fit_params)
print(f"noise_threshold_1: {time.perf_counter()-t0:.3f}s")

# tensor fit
xtol = float(fit_params.get("xtol", 1e-4))
ftol = float(fit_params.get("ftol", 1e-4))
max_ite = int(fit_params.get("max_ite", 200))

t0 = time.perf_counter()
p_opt, success, cost = batched_levenberg_marquardt(
    x=x_array,
    Y_data=Y_matrix,
    evaluate_fn=evaluator.evaluate,
    jacobian_fn=evaluator.jacobian,
    p0=p0,
    lower_bounds=evaluator.lower_bounds,
    upper_bounds=evaluator.upper_bounds,
    weights=weights_matrix,
    max_iter=max_ite,
    xtol=xtol,
    ftol=ftol,
)
print(f"tensor_fit: {time.perf_counter()-t0:.3f}s ({success.sum()}/{n} converged)")

# noise_threshold 2
t0 = time.perf_counter()
evaluator.apply_noise_threshold(spectra, p_opt, fit_params)
print(f"noise_threshold_2: {time.perf_counter()-t0:.3f}s")

# write_back
t0 = time.perf_counter()
for i, spectrum in enumerate(spectra):
    fr = evaluator.build_result(p_opt[i], spectrum.x, spectrum.y, bool(success[i]))
    if weights_matrix is not None:
        fr.best_fit = fr.best_fit.copy()
        fr.best_fit[weights_matrix[i] == 0] = 0.0
    evaluator.write_back_to_spectrum(spectrum, fr)
print(f"write_back: {time.perf_counter()-t0:.3f}s")
