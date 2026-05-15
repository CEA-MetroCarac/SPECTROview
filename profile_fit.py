"""Profile the full fitting flow as user experiences it (apply_model=True)."""
import time
import json
import numpy as np

from spectroview.model.m_settings import MSettings
from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps

settings = MSettings()
vm = VMWorkspaceMaps(settings)
vm.load_map_files(["examples/2Dmap_MoS2.txt"])
n = len(vm.spectra)
print(f"Loaded {n} spectra")

with open("examples/fit_model2.json", "r") as f:
    fit_model = json.load(f)["0"]

from spectroview.fit_engine.tensor_engine import TensorFittingEngine

# Run #1: apply_model=True (first time applying fit model — what user does)
engine1 = TensorFittingEngine()
spectra1 = list(vm.spectra)
fit_params = fit_model.get("fit_params", {})

print("\n=== RUN 1: apply_model_to_spectra=True (user applies fit model) ===")
t0 = time.perf_counter()
results1 = engine1.fit_spectra(
    spectra1,
    fit_model,
    fit_params=fit_params,
    apply_model_to_spectra=True,
)
t_total = time.perf_counter() - t0
print(f"TOTAL: {t_total:.3f}s")
for k, v in engine1.timings.items():
    print(f"  {k}: {v}")

# Run #2: apply_model=False (re-fitting — already has models)
engine2 = TensorFittingEngine()
spectra2 = list(vm.spectra)

print("\n=== RUN 2: apply_model_to_spectra=False (re-fit) ===")
t0 = time.perf_counter()
results2 = engine2.fit_spectra(
    spectra2,
    fit_model,
    fit_params=fit_params,
    apply_model_to_spectra=False,
)
t_total2 = time.perf_counter() - t0
print(f"TOTAL: {t_total2:.3f}s")
for k, v in engine2.timings.items():
    print(f"  {k}: {v}")
