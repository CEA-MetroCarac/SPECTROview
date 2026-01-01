import numpy as np
import numba as nb
from .peaks import *

@nb.njit(fastmath=True)
def evaluate_model(x, p, model_defs):
    """
    p = flat parameter array
    model_defs = [(type_id, offset, n_params), ...]
    """
    y = np.zeros_like(x)
    for (kind, off, _) in model_defs:
        if kind == 0:
            y += gaussian(x, p[off], p[off+1], p[off+2])
        elif kind == 1:
            y += gaussian_asym(x, p[off], p[off+1], p[off+2], p[off+3])
        elif kind == 2:
            y += lorentzian(x, p[off], p[off+1], p[off+2])
        elif kind == 3:
            y += lorentzian_asym(x, p[off], p[off+1], p[off+2], p[off+3])
        elif kind == 4:
            y += pseudovoigt(x, p[off], p[off+1], p[off+2], p[off+3])
    return y
