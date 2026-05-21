import time
from spectroview.model.m_spectrum import MSpectrum
from fitspy import PEAK_MODELS
from fitspy.core.spectrum import create_model

t0 = time.time()
spec_meta = {
    'peak_models': {
        0: {'Gaussian': {}},
        1: {'Gaussian': {}},
        2: {'Gaussian': {}},
        3: {'Gaussian': {}},
        4: {'Gaussian': {}}
    }
}
spectra = []
for i in range(10000):
    s = MSpectrum()
    s.set_attributes(spec_meta)
    spectra.append(s)
t1 = time.time()
print(f"10,000 spectra with 5 peaks took {t1-t0} s")
