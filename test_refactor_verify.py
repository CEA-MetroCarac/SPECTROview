"""Verify refactored MSpectra save/load methods."""
import sys
sys.path.insert(0, 'c:/Users/VL251876/Documents/Python/SPECTROview-1')

from pathlib import Path
from spectroview.model.m_io import load_wdf_map
from spectroview.model.m_spectrum import MSpectrum
from spectroview.model.m_spectra import MSpectra
import json, gzip, numpy as np
from io import StringIO
import pandas as pd

print("="*60)
print("TEST 1: Maps workspace (WDF) save/load using MSpectra methods")
print("="*60)
df, metadata = load_wdf_map(Path('examples/spectroscopic_data/2dmap.wdf'))
wavenumber_cols = [c for c in df.columns if c not in ['X', 'Y']]
x_data = np.asarray([float(c) for c in wavenumber_cols])

spectra = MSpectra()
for idx in range(3):
    s = MSpectrum()
    s.fname = f"2dmap_({float(df['X'].iloc[idx])}, {float(df['Y'].iloc[idx])})"
    s.x = x_data.copy(); s.x0 = x_data.copy()
    s.y = df[wavenumber_cols].iloc[idx].values.copy(); s.y0 = s.y.copy()
    s.baseline.mode = "Linear"; s.baseline.sigma = 4; s.xcorrection_value = 0
    s.metadata = metadata.copy()
    spectra.add(s)

# Save using new method
spectrums_data = spectra.save(is_map=True)
has_meta = any('metadata' in spectrums_data[k] for k in spectrums_data)
print(f"  Metadata in per-spectrum data: {has_meta} (expected: False)")

# Load
data = json.loads(json.dumps({
    'spectrums_data': spectrums_data,
    'maps': {'2dmap': gzip.compress(df.to_csv(index=False).encode('utf-8')).hex()},
    'maps_metadata': {'2dmap': metadata},
}))
loaded_maps = {}
for mn, hd in data['maps'].items():
    loaded_maps[mn] = pd.read_csv(StringIO(gzip.decompress(bytes.fromhex(hd)).decode('utf-8')))

loaded_spectra = MSpectra()
for sid, sd in data.get('spectrums_data', {}).items():
    # Load using new method
    s = MSpectra.load_from_dict(
        spectrum_class=MSpectrum,
        spectrum_data=sd,
        maps=loaded_maps,
        is_map=True
    )
    s.preprocess()
    loaded_spectra.append(s)

# Restore metadata from maps_metadata
for s in loaded_spectra:
    mn = s.fname.rsplit('_', 1)[0]
    mm = data.get('maps_metadata', {}).get(mn, {})
    if mm: s.metadata = mm.copy()

all_ok = all(s.x0 is not None and s.y0 is not None and bool(s.metadata) for s in loaded_spectra)
for i, s in enumerate(loaded_spectra):
    print(f"  {'✓' if all_ok else '✗'} Spectrum {i}: x0={s.x0 is not None}, meta={len(s.metadata)} keys")
print(f"  {'✅ PASSED' if all_ok else '❌ FAILED'}")

print("\n" + "="*60)
print("TEST 2: Spectra workspace save/load using MSpectra methods")
print("="*60)
spectra2 = MSpectra()
for i in range(2):
    s = MSpectrum()
    s.fname = f"test_spectrum_{i}"
    s.x = np.linspace(100, 3000, 100)
    s.x0 = s.x.copy(); s.y = np.random.rand(100); s.y0 = s.y.copy()
    s.baseline.mode = "Linear"; s.baseline.sigma = 4; s.xcorrection_value = 0
    s.metadata = {"File Format": "Test", "Laser Wavelength (nm)": "532.00"}
    spectra2.add(s)

# Save using new method
spectrums_data2 = spectra2.save(is_map=False)
has_meta2 = any('metadata' in spectrums_data2[k] for k in spectrums_data2)
print(f"  Metadata in per-spectrum data: {has_meta2} (expected: True)")

# Load
data2 = json.loads(json.dumps({'spectrums': spectrums_data2}))
loaded2 = MSpectra()
for sid, sd in data2['spectrums'].items():
    # Load using new method
    s = MSpectra.load_from_dict(
        spectrum_class=MSpectrum,
        spectrum_data=sd,
        is_map=False
    )
    loaded2.append(s)

all_ok2 = all(bool(s.metadata) for s in loaded2)
for i, s in enumerate(loaded2):
    print(f"  {'✓' if bool(s.metadata) else '✗'} Spectrum {i}: meta={len(s.metadata)} keys, vals={s.metadata}")
print(f"  {'✅ PASSED' if all_ok2 else '❌ FAILED'}")
