import numpy as np

def detect_noise_level(y: np.ndarray) -> float:
    """Estimate noise amplitude using median absolute deviation of differences."""
    dy = np.diff(y)
    return float(np.median(np.abs(dy)) / 0.6745 * np.sqrt(2))
