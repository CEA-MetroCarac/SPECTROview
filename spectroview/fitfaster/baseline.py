import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def arpls(y, lam=1e4, ratio=0.05, niter=10):
    n = len(y)
    D = sparse.eye(n, format='csc')
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]
    H = lam * (D.T @ D)
    w = np.ones(n)

    for _ in range(niter):
        W = sparse.diags(w)
        Z = W + H
        z = spsolve(Z, w*y)
        d = y - z
        dn = d[d < 0]
        if len(dn) == 0:
            break
        m, s = dn.mean(), dn.std()
        w_new = 1 / (1 + np.exp(2*(d-(2*s-m))/s))
        if np.linalg.norm(w-w_new)/np.linalg.norm(w) < ratio:
            break
        w = w_new
    return z
