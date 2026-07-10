"""Approximate tissue-mask estimation -- vendored from ../../../src/masks.py.

All 8 images (including the 5 that have ground-truth Tissue polygons in
Checkpoint 1) go through this same Otsu-on-log-block-mean estimator here,
uniformly. The web app is demonstrating the trained detector's inference
pipeline end-to-end, not re-implementing GeoJSON mask loading -- the
GeoJSON annotations were never uploaded to the bucket (only the two derived
image assets were, per the Phase A plan), so there is no ground-truth mask
available to this service for *any* image, labeled or not.

Validated in Checkpoint 1 at IoU 0.92-0.96 against the real ground-truth
tissue masks for images 1-5 (see step_of_implementation_checkpoint1.md §1)
-- good enough to gate background tiles out of inference, not a claim of
pixel-perfect tissue segmentation.
"""

import numpy as np


def _otsu_threshold_1d(values: np.ndarray, nbins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=nbins)
    hist = hist.astype(np.float64)
    mids = (edges[:-1] + edges[1:]) / 2

    weight_below = np.cumsum(hist)
    weight_above = np.cumsum(hist[::-1])[::-1]
    weight_below_safe = np.where(weight_below == 0, 1, weight_below)
    weight_above_safe = np.where(weight_above == 0, 1, weight_above)
    mean_below = np.cumsum(hist * mids) / weight_below_safe
    mean_above = (np.cumsum((hist * mids)[::-1])[::-1]) / weight_above_safe

    inter_class_var = weight_below[:-1] * weight_above[1:] * (mean_below[:-1] - mean_above[1:]) ** 2
    return float(mids[int(np.argmax(inter_class_var))])


def _block_mean(image: np.ndarray, block: int) -> np.ndarray:
    height, width = image.shape
    h2, w2 = (height // block) * block, (width // block) * block
    cropped = image[:h2, :w2]
    return cropped.reshape(h2 // block, block, w2 // block, block).mean(axis=(1, 3))


def estimate_tissue_mask(image: np.ndarray, block: int = 64) -> np.ndarray:
    block_means = _block_mean(image, block)
    threshold = _otsu_threshold_1d(np.log1p(block_means).ravel())
    block_mask = np.log1p(block_means) > threshold

    mask = np.repeat(np.repeat(block_mask, block, axis=0), block, axis=1)
    full_mask = np.zeros(image.shape, dtype=bool)
    h2, w2 = mask.shape
    full_mask[:h2, :w2] = mask
    return full_mask
