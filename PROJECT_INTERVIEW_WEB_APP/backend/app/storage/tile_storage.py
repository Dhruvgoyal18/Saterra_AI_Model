"""Deep Zoom (DZI) tile cache/orchestration -- get-or-create, same shape as
hf_storage.get_display_image_path, but for a full tile pyramid instead of one
flat PNG.

Deliberately a sibling module to hf_storage.py, not an edit to it: this keeps
hf_storage.py byte-for-byte untouched so the deep-zoom feature has zero risk
of regressing the existing display/detect paths. This module never talks to
huggingface_hub directly -- it goes through hf_storage.get_original_path, so
hf_storage.py's "only module that talks to HF Hub" invariant still holds.

Two caches, both explicitly ephemeral (lost on container restart, same
accepted model as hf_storage's display-PNG cache on HF Spaces' no-persistent-
disk containers):
  - Per-tile JPEGs + a per-image meta sidecar on disk (settings.hf_tile_cache_dir)
    -- the "don't regenerate this again" cache.
  - A small process-local, in-memory, LRU-capped cache of each image's full
    decoded pyramid (dict[level -> np.ndarray]) -- purely a same-image-burst
    optimization so a run of tile requests for the image currently open in
    the viewer doesn't re-decode+re-resize the original TIFF per tile. Not a
    durability guarantee; rebuilt from scratch on a miss, same cost the
    display-PNG path and /detect already pay per call today.
"""

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import tifffile

from ..config import settings
from ..ml.tiling import (
    DZI_OVERLAP,
    DZI_TILE_SIZE,
    build_pyramid,
    compute_normalization_stats,
    dzi_level_dims,
    dzi_max_level,
    extract_tile,
)
from .hf_storage import get_original_path

logger = logging.getLogger("artifact-detector")

_PYRAMID_CACHE_MAX_IMAGES = 2


class InvalidTileError(Exception):
    """level/col/row out of range for this image's pyramid, or a malformed
    tile filename. Distinct from ImageNotFoundError, which means image_id
    itself doesn't exist -- this means image_id is fine but the requested
    tile coordinates aren't."""


_pyramid_cache_lock = threading.Lock()
_pyramid_cache: "OrderedDict[str, dict[int, object]]" = OrderedDict()


def _get_or_build_pyramid(image_id: str) -> dict[int, object]:
    """In-memory LRU (cap _PYRAMID_CACHE_MAX_IMAGES) of full decoded pyramids,
    keyed by image_id. Held under a lock since FastAPI runs sync routes in a
    threadpool -- multiple real OS threads can call this concurrently, and
    the evict-then-insert sequence below isn't safe as separate uncoordinated
    dict operations.
    """
    with _pyramid_cache_lock:
        cached = _pyramid_cache.get(image_id)
        if cached is not None:
            _pyramid_cache.move_to_end(image_id)
            return cached

        original_path = get_original_path(image_id)
        with tifffile.TiffFile(original_path) as tif:
            image = tif.series[0].levels[0].asarray()
        pyramid = build_pyramid(image)

        _pyramid_cache[image_id] = pyramid
        _pyramid_cache.move_to_end(image_id)
        if len(_pyramid_cache) > _PYRAMID_CACHE_MAX_IMAGES:
            evicted_id, _ = _pyramid_cache.popitem(last=False)
            logger.info("Evicted %s from the in-memory tile-pyramid cache", evicted_id)

        logger.info("Built in-memory tile pyramid for %s (%d levels)", image_id, len(pyramid))
        return pyramid


def _meta_path(image_id: str) -> Path:
    return settings.hf_tile_cache_dir / f"{image_id}.meta.json"


def _tile_path(image_id: str, level: int, col: int, row: int) -> Path:
    return settings.hf_tile_cache_dir / image_id / str(level) / f"{col}_{row}.jpeg"


def get_or_build_meta(image_id: str) -> dict:
    """{"width", "height", "max_level", "lo", "hi"} for image_id.

    Cache hit: read the sidecar, no HF traffic, no decode.
    Cache miss: fetch the original (hf_storage.get_original_path -- reuses
    huggingface_hub's own cache, the same original file the display-PNG and
    /detect paths already fetch, no new download path), decode the full
    array once, compute normalization stats once, write the sidecar. This is
    the ONLY place percentiles get computed for this image's tile pyramid --
    every tile request thereafter reads lo/hi from this cached dict, never
    recomputes.
    """
    meta_path = _meta_path(image_id)
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)

    settings.hf_tile_cache_dir.mkdir(parents=True, exist_ok=True)

    original_path = get_original_path(image_id)
    with tifffile.TiffFile(original_path) as tif:
        image = tif.series[0].levels[0].asarray()

    height, width = image.shape
    lo, hi = compute_normalization_stats(image)
    max_level = dzi_max_level(width, height)

    meta = {"width": width, "height": height, "max_level": max_level, "lo": lo, "hi": hi}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    logger.info("Built tile-pyramid meta for %s (%dx%d, max_level=%d)", image_id, width, height, max_level)
    return meta


def get_dzi_descriptor_xml(image_id: str) -> str:
    """Standard Deep Zoom Image XML descriptor. OpenSeadragon consumes this
    natively (`tileSources: '.../dzi.dzi'`) with no custom TileSource code."""
    meta = get_or_build_meta(image_id)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
        f'Format="jpeg" Overlap="{DZI_OVERLAP}" TileSize="{DZI_TILE_SIZE}">'
        f'<Size Width="{meta["width"]}" Height="{meta["height"]}"/>'
        "</Image>"
    )


def get_or_build_tile(image_id: str, level: int, col: int, row: int) -> Path:
    """Get-or-create one tile's JPEG path.

    Cache hit: JPEG already on disk, return immediately, zero decode.
    Cache miss: validate level/col/row against this image's real pyramid
    dimensions (raises InvalidTileError if out of range -- this must happen
    BEFORE touching the pyramid cache, since an out-of-range level would
    otherwise KeyError inside the pyramid dict instead of producing a clean
    404), get this image's in-memory pyramid (build/reuse via
    _get_or_build_pyramid), extract the one requested tile, save it as JPEG,
    return its path.
    """
    meta = get_or_build_meta(image_id)
    max_level, width, height = meta["max_level"], meta["width"], meta["height"]

    if not (0 <= level <= max_level):
        raise InvalidTileError(f"Invalid level {level} for '{image_id}' (must be 0..{max_level})")

    level_w, level_h = dzi_level_dims(level, max_level, width, height)
    n_cols = -(-level_w // DZI_TILE_SIZE)
    n_rows = -(-level_h // DZI_TILE_SIZE)
    if not (0 <= col < n_cols) or not (0 <= row < n_rows):
        raise InvalidTileError(
            f"Invalid tile ({col},{row}) at level {level} for '{image_id}' "
            f"(level is {level_w}x{level_h}, valid cols 0..{n_cols - 1}, rows 0..{n_rows - 1})"
        )

    tile_path = _tile_path(image_id, level, col, row)
    if tile_path.exists():
        return tile_path

    pyramid = _get_or_build_pyramid(image_id)
    level_array = pyramid[level]
    tile_img = extract_tile(level_array, col, row, DZI_TILE_SIZE, meta["lo"], meta["hi"])

    tile_path.parent.mkdir(parents=True, exist_ok=True)
    tile_img.save(tile_path, format="JPEG", quality=90)
    return tile_path
