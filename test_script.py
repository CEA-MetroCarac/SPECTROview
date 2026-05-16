import json
from pathlib import Path
from spectroview.model.m_spectra import MSpectra

spectra = MSpectra()
try:
    fit_model = spectra.load_model("examples/predefined_fit_models/fit_model_Si_.json", ind=0)
    print("LOADED successfully")
except Exception as e:
    print(f"FAILED: {e}")
