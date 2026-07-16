"""detect_artifacts -- the Checkpoint 1 inference contract, vendored from
../../../backend/app/ml/inference.py and reused as-is by this service.

Returns numpy arrays; app/main.py is responsible for converting them into
the JSON-friendly payload the backend expects (grid_shape, tile_confidence
as nested lists, summary, level0_dims, tile_size).
"""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .model import SmallCNN
from .preprocessing import compute_percentile_norm_stats, normalize_tile

_MODEL_CACHE: dict[str, SmallCNN] = {}

DEFAULT_TILE_SIZE = 64
DEFAULT_BACKGROUND_THRESHOLD = 0.5


def load_model(weights_path: Path, device: str = "cpu") -> SmallCNN:
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
    weights_path: Path,
    tissue_mask: np.ndarray | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    background_threshold: float = DEFAULT_BACKGROUND_THRESHOLD,
    prob_threshold: float = 0.5,
    device: str = "cpu",
    batch_size: int = 512,
) -> dict:
    model = load_model(weights_path, device)
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
        batch = torch.from_numpy(np.stack(tiles_batch)).unsqueeze(1)
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
