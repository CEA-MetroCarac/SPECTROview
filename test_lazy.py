import time
from spectroview.model.m_spectrum import MSpectrum

t0 = time.time()
spec_meta = {
    'fname': 'test',
    'is_active': True,
    'color': 'red',
    'label': 'l1',
    'peak_models': {
        0: {'Gaussian': {}},
        1: {'Gaussian': {}},
        2: {'Gaussian': {}},
        3: {'Gaussian': {}},
        4: {'Gaussian': {}}
    }
}
spectra = []
for i in range(50000):
    s = MSpectrum()
    s.fname = spec_meta['fname']
    s.is_active = spec_meta.get('is_active', True)
    s.color = spec_meta.get('color', None)
    s.label = spec_meta.get('label', None)
    s._lazy_meta = spec_meta
    spectra.append(s)
t1 = time.time()
print(f"50,000 lazy spectra took {t1-t0} s")
