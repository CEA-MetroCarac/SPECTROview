import sys, time, json, os
from PySide6.QtWidgets import QApplication
from spectroview.model.m_settings import MSettings
from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum

from unittest.mock import patch

app = QApplication.instance() or QApplication(sys.argv)
REAL_FILE = r"examples\spectroscopic_data\CL_map_saved_with results.maps"

print(f"\nLoading raw json for state rebuild...")
with open(REAL_FILE, 'r') as f:
    data = json.load(f)

spectra = MSpectra()
for spectrum_id, sd in data.get('spectrums_data', {}).items():
    sd_copy = dict(sd)
    sd_copy.pop('metadata', None)
    spectrum = MSpectrum()
    # we don't bother linking maps/kdtree here, just test the save serialization
    spectrum.set_attributes(sd_copy)
    spectra.append(spectrum)

print(f"Loaded {len(spectra)} spectra.")
print("="*60)
t = time.time()
res = spectra.save(is_map=True)
print(f"spectra.save(is_map=True): {time.time()-t:.3f} s")
print("="*60)
