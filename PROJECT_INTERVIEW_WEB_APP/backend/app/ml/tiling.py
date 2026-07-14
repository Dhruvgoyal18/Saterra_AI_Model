"""Deep Zoom (DZI) pyramid generation for a single whole-slide DAPI image.

None of the 8 source .ome.tiff files have a native multi-resolution pyramid
(confirmed via tifffile: `len(tif.series[0].levels) == 1` for every file
checked) -- so the tile pyramid a deep-zoom viewer needs is generated here,
server-side, from the single full-resolution level. This intentionally does
NOT use openslide-python: that library would only be downsampling from the
same single base level anyway (no free native pyramid to exploit), and adding
it would mean a new system-package layer (libopenslide0) in the Dockerfile,
which is currently a plain `python:3.11-slim` with no apt-get step at all.
Everything here uses libraries already in requirements.txt (numpy, cv2,
Pillow) -- same category of engineering as ml/display_prep.py.

DZI level numbering (the standard convention, NOT the same direction as
"how far downsampled"): level 0 is a 1x1 image, level `max_level` is the
full native resolution. `build_pyramid` returns a dict keyed directly by
this DZI level number (not a list in some other index order) specifically
to remove any "which direction does the index go" ambiguity at the seam
between this module and the DZI descriptor/tile routes -- ask for level 7,
get exactly the array for level 7, no translation required by the caller.

The classifier's own tile size (preprocessing_config.json: 64px, a level-0
grid used only for the confidence overlay) is unrelated to DZI_TILE_SIZE
below (256px, the deep-zoom viewer's tile size) -- these are two independent
tilings over the same image and must never be conflated.
"""

import math

import cv2
import numpy as np
from PIL import Image

DZI_TILE_SIZE = 256
DZI_OVERLAP = 0
LOW_PCTILE, HIGH_PCTILE = 1.0, 99.0


def compute_normalization_stats(image: np.ndarray) -> tuple[float, float]:
    """Same 1st/99th-percentile logic as display_prep.to_display_png,
    computed ONCE per image here and reused for every tile at every level --
    never recomputed per tile, which would make brightness inconsistent
    from one tile to its neighbor as a viewer pans across a tile boundary.
    """
    lo, hi = np.percentile(image, LOW_PCTILE), np.percentile(image, HIGH_PCTILE)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def dzi_max_level(width: int, height: int) -> int:
    """Standard DZI numbering: the level at which the image is full
    resolution. Level 0 is always a 1x1 image."""
    return math.ceil(math.log2(max(width, height)))


def dzi_level_dims(level: int, max_level: int, width: int, height: int) -> tuple[int, int]:
    """Width/height of a given DZI level, per the standard DZI formula --
    ceil(original_dim / 2^(max_level - level)), never a plain halving-with-
    rounding recurrence. This exact formula is what OpenSeadragon itself uses
    client-side to compute expected level dimensions from the descriptor's
    single (full-res) Size field, so the arrays this module produces must
    match it exactly -- a naive "halve and round" recurrence would drift from
    this formula over several levels on odd dimensions, silently breaking
    tile/row/col counts at a level's right/bottom edge.
    """
    factor = 2 ** (max_level - level)
    return max(1, math.ceil(width / factor)), max(1, math.ceil(height / factor))


def build_pyramid(image: np.ndarray) -> dict[int, np.ndarray]:
    """Build every DZI level's array, keyed by DZI level number, all the way
    down to level 0 (1x1). Each level is resized from the nearest larger
    already-computed level (not re-resized from the full-res array each
    time) via cv2.resize -- cheap, since every step after the first works on
    an already-small array, and the total memory for the whole dict is only
    about 1.33x the full-res array's size (geometric series of quarter-area
    steps), nowhere near the class of blowup display_prep.py's fix addresses.

    Deliberately NO clip/rescale/percentile step happens in this function --
    every array stays in its original dtype (uint16). That float64-upcasting
    op only ever happens in extract_tile, and only against one tile's slice
    at a time (see that function's docstring) -- this is what keeps this
    function safe to call at full resolution without repeating the fixed
    Render OOM bug (display_prep.py's history: clipping the FULL array
    against float percentile bounds allocated a ~637MB intermediate for one
    9000x9000 image).
    """
    height, width = image.shape
    max_level = dzi_max_level(width, height)
    pyramid: dict[int, np.ndarray] = {max_level: image}
    current = image
    for level in range(max_level - 1, -1, -1):
        target_w, target_h = dzi_level_dims(level, max_level, width, height)
        current = cv2.resize(current, (target_w, target_h), interpolation=cv2.INTER_AREA)
        pyramid[level] = current
    return pyramid


def extract_tile(
    level_array: np.ndarray, col: int, row: int, tile_size: int, lo: float, hi: float
) -> Image.Image:
    """Slice ONE tile_size x tile_size (or smaller, at right/bottom edges)
    region out of level_array FIRST, THEN clip/rescale that small slice to
    uint8. This is the direct generalization of display_prep.py's existing
    OOM fix (downsample-before-clip) to the tiled case: the float64 upcast
    during clip/rescale never touches more than one tile's worth of pixels
    (<= tile_size^2 * 8 bytes), at ANY pyramid level including the native-
    resolution level -- unlike the flat-PNG path, which only ever needs a
    downsampled preview, this path DOES eventually have to normalize
    full-resolution pixel data, but never all of it at once.

    Caller is expected to have already validated col/row are in range for
    this level (see storage/tile_storage.py's InvalidTileError) -- this
    function assumes valid input and does not re-validate bounds itself.
    """
    height, width = level_array.shape
    y0, y1 = row * tile_size, min((row + 1) * tile_size, height)
    x0, x1 = col * tile_size, min((col + 1) * tile_size, width)
    tile = level_array[y0:y1, x0:x1]

    clipped = np.clip(tile, lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")
