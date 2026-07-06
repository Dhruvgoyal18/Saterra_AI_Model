"""The single inference entrypoint: detect_artifacts(image) -> {mask, per_tile_confidence, summary}.

This is the contract Checkpoints 2 and 3 are meant to bolt onto directly --
everything upstream (data prep, LOIO CV, final training) is throwaway once
the weights exist; this function is not.
"""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from . import config
from .dataset import compute_percentile_norm_stats, normalize_tile
from .model import SmallCNN

_MODEL_CACHE: dict[str, SmallCNN] = {}


def _load_model(weights_path: Path, device: str = "cpu") -> SmallCNN:
    key = str(weights_path)
    if key not in _MODEL_CACHE:
        model = SmallCNN()
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def detect_artifacts(
    image: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    tile_size: int = config.TILE_SIZE,
    background_threshold: float = config.BACKGROUND_THRESHOLD,
    weights_path: Path = config.MODEL_WEIGHTS_PATH,
    prob_threshold: float = 0.5,
    device: str = "cpu",
    batch_size: int = 512,
) -> dict:
    """Tile `image`, classify every tissue tile, and stitch the result.

    Parameters
    ----------
    image: full-resolution single-channel (H, W) DAPI array.
    tissue_mask: optional boolean (H, W) mask restricting where tiles are
        evaluated. Pass the ground-truth mask when available; when None
        (e.g. for images with no annotation), every tile is evaluated and
        `summary["total_artifact_area_frac_of_tissue"]` falls back to a
        frac-of-image-area instead. Use masks.estimate_tissue_mask(image) to
        get an approximate tissue mask for unlabeled images.

    Returns
    -------
    dict with:
      mask: (H, W) bool -- full-resolution stitched artifact prediction.
      per_tile_confidence: (n_rows, n_cols) float32 -- sigmoid probability per
          tile on the coarse tile grid (0 for tiles that were skipped as
          background).
      summary: dict with total_artifact_area_px, total_artifact_area_frac,
          num_connected_regions, largest_region_area_px, n_tiles_evaluated,
          n_tiles_total -- exactly the fields Checkpoint 2's UI needs to
          render without any further computation.
    """
    model = _load_model(weights_path, device)
    height, width = image.shape
    n_rows, n_cols = height // tile_size, width // tile_size

    lo, hi = compute_percentile_norm_stats(image)

    mask = np.zeros((height, width), dtype=bool)
    confidence = np.zeros((n_rows, n_cols), dtype=np.float32)
    tile_evaluated = np.zeros((n_rows, n_cols), dtype=bool)

    tiles_batch: list[np.ndarray] = []
    coords_batch: list[tuple[int, int]] = []

    def flush_batch() -> None:
        if not tiles_batch:
            return
        batch = torch.from_numpy(np.stack(tiles_batch)).unsqueeze(1)  # (B, 1, H, W)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).numpy()
        for (r, c), p in zip(coords_batch, probs):
            confidence[r, c] = p
            tile_evaluated[r, c] = True
            if p >= prob_threshold:
                row0, col0 = r * tile_size, c * tile_size
                mask[row0 : row0 + tile_size, col0 : col0 + tile_size] = True
        tiles_batch.clear()
        coords_batch.clear()

    for r in range(n_rows):
        row0 = r * tile_size
        for c in range(n_cols):
            col0 = c * tile_size
            if tissue_mask is not None:
                tissue_tile = tissue_mask[row0 : row0 + tile_size, col0 : col0 + tile_size]
                if tissue_tile.mean() < background_threshold:
                    continue
            tile = image[row0 : row0 + tile_size, col0 : col0 + tile_size]
            tiles_batch.append(normalize_tile(tile, lo, hi))
            coords_batch.append((r, c))
            if len(tiles_batch) >= batch_size:
                flush_batch()
    flush_batch()

    if tissue_mask is not None:
        mask &= tissue_mask
        tissue_px = int(tissue_mask.sum())
    else:
        tissue_px = height * width

    labeled, num_regions = ndimage.label(mask)
    if num_regions > 0:
        region_sizes = ndimage.sum(mask, labeled, index=np.arange(1, num_regions + 1))
        largest_region_px = int(region_sizes.max())
    else:
        largest_region_px = 0

    total_artifact_px = int(mask.sum())
    summary = {
        "total_artifact_area_px": total_artifact_px,
        "total_artifact_area_frac_of_tissue": (total_artifact_px / tissue_px) if tissue_px else 0.0,
        "num_connected_regions": int(num_regions),
        "largest_region_area_px": largest_region_px,
        "n_tiles_evaluated": int(tile_evaluated.sum()),
        "n_tiles_total": int(n_rows * n_cols),
    }

    return {"mask": mask, "per_tile_confidence": confidence, "summary": summary}
