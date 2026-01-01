import numpy as np

def propagate_initial_guesses(coords, p0):
    order = np.lexsort((coords[:,1], coords[:,0]))
    p0_new = p0.copy()
    for i in range(1, len(order)):
        p0_new[order[i]] = p0_new[order[i-1]]
    return p0_new
