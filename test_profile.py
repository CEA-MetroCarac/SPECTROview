import time, json
from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum

spectra = MSpectra()
for i in range(50000):
    s = MSpectrum()
    s.fname = f"map_test_({i%100}, {i//100})"
    s.baseline.mode = "Linear"
    spectra.append(s)

t0 = time.time()
spectrums_data = spectra.save(is_map=True)
t1 = time.time()
print(f"spectra.save() took {t1-t0} s")

# Dump
j = json.dumps(spectrums_data)
t2 = time.time()
print(f"json.dumps (no indent) took {t2-t1} s. Size: {len(j)/1024/1024:.2f} MB")

# Load
t3 = time.time()
loaded = json.loads(j)
t4 = time.time()
print(f"json.loads took {t4-t3} s")
