"""Shared display-image generation logic: percentile-normalize a full-res
DAPI array to 8-bit and downsample it for browser display.

Originally lived only in scripts/prepare_display_images.py. Moved here so
`storage/hf_storage.py` (which now generates display PNGs on demand, one
image at a time, the first time each is requested) and the now-optional
local debug script both call the exact same code -- there is only one
definition of "how a display PNG is made," not two copies that could drift.
"""

import cv2
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
    # Percentiles computed on the full-resolution array -- must happen
    # before any downsampling so the contrast stretch is calibrated
    # identically to before, not an approximation from a downsampled preview.
    lo, hi = np.percentile(image, LOW_PCTILE), np.percentile(image, HIGH_PCTILE)
    if hi <= lo:
        hi = lo + 1.0

    height, width = image.shape
    scale = TARGET_LONG_SIDE / max(height, width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))

    # Downsample the raw uint16 array BEFORE clip/rescale. Clipping against
    # float percentile bounds upcasts uint16 -> float64; doing that at full
    # resolution (~9000x9000) allocated a ~637MB intermediate array, which
    # was blowing past low-memory hosts. cv2 (not PIL) for the resize --
    # cv2.resize handles uint16 input correctly, whereas PIL's 16-bit mode
    # support is less consistent across versions.
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    clipped = np.clip(small, lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)

    pil_img = Image.fromarray(scaled, mode="L")
    return pil_img, (new_w, new_h)
