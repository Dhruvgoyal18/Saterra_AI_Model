"""Shared display-image generation logic: percentile-normalize a full-res
DAPI array to 8-bit and downsample it for browser display.

Originally lived only in scripts/prepare_display_images.py. Moved here so
`storage/hf_storage.py` (which now generates display PNGs on demand, one
image at a time, the first time each is requested) and the now-optional
local debug script both call the exact same code -- there is only one
definition of "how a display PNG is made," not two copies that could drift.
"""

import numpy as np
from PIL import Image

TARGET_LONG_SIDE = 2000
LOW_PCTILE, HIGH_PCTILE = 1.0, 99.0


def to_display_png(image: np.ndarray) -> tuple[Image.Image, tuple[int, int]]:
    """Same 1st/99th-percentile normalization used everywhere else in this
    project (Checkpoint 1's dataset.py, this backend's ml/preprocessing.py)
    -- reused here so the display image looks like what the model actually
    sees, not an independently-chosen display stretch.

    Returns (PIL image, (width, height)) of the downsampled result.
    """
    lo, hi = np.percentile(image, LOW_PCTILE), np.percentile(image, HIGH_PCTILE)
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(image, lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)

    height, width = image.shape
    scale = TARGET_LONG_SIDE / max(height, width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))

    pil_img = Image.fromarray(scaled, mode="L").resize((new_w, new_h), Image.LANCZOS)
    return pil_img, (new_w, new_h)
