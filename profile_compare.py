"""Profile the OLD tensor fit engine from commit 1e1d15c to compare.
Uses git show to load old code at runtime."""
import time
import json
import sys
import types
import numpy as np

# First, load old evaluator and engine source from the fast commit
import subprocess

def load_old_module(commit, module_path, module_name):
    """Load a Python module from a specific git commit."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{module_path}"],
        capture_output=True, text=True, cwd="."
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to load {module_path} from {commit}: {result.stderr}")
    return result.stdout

# Load old engine source to compare structure
old_engine_src = load_old_module("1e1d15c", "spectroview/fit_engine/tensor_engine.py", "old_engine")
old_eval_src = load_old_module("1e1d15c", "spectroview/fit_engine/evaluator.py", "old_eval")

# Compare key function signatures
print("=== OLD ENGINE: build_p0_matrix signature ===")
for line in old_eval_src.split("\n"):
    if "def build_p0_matrix" in line:
        print(f"  {line.strip()}")

print("\n=== OLD ENGINE: build_result signature ===")
for line in old_eval_src.split("\n"):
    if "def build_result" in line:
        print(f"  {line.strip()}")

print("\n=== OLD ENGINE: _build_fit_weights calls ===")
for i, line in enumerate(old_engine_src.split("\n")):
    if "noise" in line.lower() or "weight" in line.lower() or "preprocess" in line.lower():
        print(f"  L{i+1}: {line.rstrip()}")

# Now run the CURRENT code with step-by-step timing
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

with open("examples/fit_model2.json", "r") as f:
    fit_model = json.load(f)["0"]

# Apply model
for s in vm.spectra:
    apply_custom_fit_model(s, fit_model, s.fname)

spectra = list(vm.spectra)
fit_params = fit_model.get("fit_params", {})
evaluator = TensorEvaluator.from_fit_model(fit_model)

# Measure: old build_result (no R²) vs new build_result (with R²)
print(f"\n=== CURRENT build_result has R² computation ===")
# Check if build_result computes R²
import inspect
src = inspect.getsource(evaluator.build_result)
has_rsquared = "rsquared" in src
print(f"  Has R² computation: {has_rsquared}")
print(f"  Lines in build_result: {len(src.split(chr(10)))}")

# Count lines in old build_result
old_br_lines = []
capture = False
for line in old_eval_src.split("\n"):
    if "def build_result" in line:
        capture = True
    elif capture and (line.strip().startswith("def ") or (line.strip() and not line.startswith(" ") and not line.startswith("\t"))):
        break
    if capture:
        old_br_lines.append(line)
print(f"  OLD build_result lines: {len(old_br_lines)}")
print(f"  OLD build_result R²: {'rsquared' in chr(10).join(old_br_lines)}")
