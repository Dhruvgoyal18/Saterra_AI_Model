"""Per-image percentile normalization -- vendored from
../../../backend/app/ml/preprocessing.py (itself from ../../../src/dataset.py).

Kept as the exact same 1st/99th-percentile logic used in Checkpoint 1's
training, so the model sees tiles normalized the same way at inference time
as it did during training.
"""

import numpy as np


def compute_percentile_norm_stats(image: np.ndarray, low_pctile: float = 1.0, high_pctile: float = 99.0) -> tuple[float, float]:
    lo = float(np.percentile(image, low_pctile))
    hi = float(np.percentile(image, high_pctile))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def normalize_tile(tile: np.ndarray, lo: float, hi: float) -> np.ndarray:
    tile = tile.astype(np.float32)
    tile = np.clip(tile, lo, hi)
    return (tile - lo) / (hi - lo)
