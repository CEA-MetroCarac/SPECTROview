import time
import numpy as np
import pandas as pd

coords = np.random.rand(50000, 2).astype(np.float64)
wavenumbers = np.random.rand(1000).astype(np.float64)
intensities = np.random.rand(50000, 1000).astype(np.float32)

t0 = time.time()
col_names = ['X', 'Y'] + [str(w) for w in wavenumbers]
data_combined = np.hstack([coords, intensities])
df = pd.DataFrame(data_combined, columns=col_names, dtype=np.float64)
t1 = time.time()
print(f"DataFrame creation took {t1-t0} s")
