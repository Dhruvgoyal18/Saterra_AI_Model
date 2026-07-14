"""Image loading, per-image percentile normalization, and the tile Dataset."""

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from . import config, hf_data
from .tiling import TileRecord


def load_dapi_image(image_id: int) -> np.ndarray:
    """Load pyramid level 0 explicitly (see test/data_validation.py for why)."""
    path = hf_data.download_image(image_id)
    with tifffile.TiffFile(path) as tif:
        return tif.series[0].levels[0].asarray()


def compute_percentile_norm_stats(image: np.ndarray) -> tuple[float, float]:
    """Per-image 1st/99th percentile clip range.

    Absolute DAPI intensity drifts between slides/scans, so normalizing off a
    fixed range (e.g. divide by 65535) would bake in whatever brightness this
    particular batch of slides happened to have. Per-image percentile scaling
    is what lets a model trained on images 1-5 transfer to the unseen,
    differently-scanned images 6-8.
    """
    lo = float(np.percentile(image, config.INTENSITY_LOW_PCTILE))
    hi = float(np.percentile(image, config.INTENSITY_HIGH_PCTILE))
    if hi <= lo:  # degenerate (near-constant) image; avoid divide-by-zero
        hi = lo + 1.0
    return lo, hi


def normalize_tile(tile: np.ndarray, lo: float, hi: float) -> np.ndarray:
    tile = tile.astype(np.float32)
    tile = np.clip(tile, lo, hi)
    return (tile - lo) / (hi - lo)


class ArtifactTileDataset(Dataset):
    """Wraps a list of TileRecords over one or more images already cached in RAM.

    Only 5-8 images total (each ~9000x9000 uint16, ~170MB), so caching every
    image in memory for the run is simpler and faster than re-reading tiles
    from disk per __getitem__, and easily fits in RAM.
    """

    def __init__(
        self,
        records: list[TileRecord],
        images: dict[int, np.ndarray],
        norm_stats: dict[int, tuple[float, float]],
    ):
        self.records = records
        self.images = images
        self.norm_stats = norm_stats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = self.images[record.image_id]
        tile = image[record.row : record.row + record.tile_size, record.col : record.col + record.tile_size]
        lo, hi = self.norm_stats[record.image_id]
        tile = normalize_tile(tile, lo, hi)
        tensor = torch.from_numpy(tile).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(float(record.label), dtype=torch.float32)
        return tensor, label
