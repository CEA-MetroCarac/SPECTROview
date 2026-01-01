import time
import numpy as np
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from .composite import evaluate_model

class FitEngine:

    def __init__(self, model_defs, bounds, max_nfev=40, n_jobs=-1):
        self.model_defs = model_defs
        self.bounds = bounds
        self.max_nfev = max_nfev
        self.n_jobs = n_jobs

    def residuals(self, p, x, y):
        return evaluate_model(x, p, self.model_defs) - y

    def fit_one(self, x, y, p0):
        res = least_squares(
            self.residuals,
            p0,
            args=(x, y),
            bounds=self.bounds,
            max_nfev=self.max_nfev
        )
        return res.x, res.success

    def fit(self, x, spectra, p0):
        t0 = time.perf_counter()
        results = Parallel(self.n_jobs)(
            delayed(self.fit_one)(x, spectra[i], p0[i])
            for i in range(len(spectra))
        )
        params, success = zip(*results)
        return np.array(params), np.array(success), time.perf_counter() - t0
