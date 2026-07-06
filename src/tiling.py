"""Non-overlapping tiling + the tile labeling rule.

Two decisions per tile, as laid out in the checkpoint plan:
  1. Include or exclude? Drop tiles that are mostly background.
  2. Artifact or clean? Majority vote (tau) on artifact coverage *within the
     tissue portion of the tile* (not the whole tile), so a tile that's half
     background/half-tissue isn't penalized in its artifact fraction just for
     touching the tissue edge.
"""

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class TileRecord:
    image_id: int
    row: int  # top-left row (pixel coords) at full resolution
    col: int  # top-left col (pixel coords) at full resolution
    tile_size: int
    label: int  # 0 = clean, 1 = artifact
    tissue_frac: float  # fraction of *all* tile pixels that are tissue
    artifact_frac: float  # fraction of *tissue* pixels in the tile that are artifact


def generate_tile_records(
    image_id: int,
    tissue_mask: np.ndarray,
    artifact_mask: np.ndarray,
    tile_size: int = config.TILE_SIZE,
    tau: float = config.TAU,
    background_threshold: float = config.BACKGROUND_THRESHOLD,
    purity: float | None = None,
) -> list[TileRecord]:
    """Slide a non-overlapping tile_size window over the full image grid.

    purity: if set (e.g. 0.95), additionally discards "mixed" boundary tiles
    whose artifact_frac falls strictly between (1-purity) and purity, keeping
    only near-pure clean/artifact tiles. Left as None by default (keep every
    tile that passes the background filter) -- see docs for the tradeoff.
    """
    height, width = tissue_mask.shape
    n_rows = height // tile_size
    n_cols = width // tile_size
    total_px = tile_size * tile_size

    records: list[TileRecord] = []
    for r in range(n_rows):
        row0 = r * tile_size
        for c in range(n_cols):
            col0 = c * tile_size
            tissue_tile = tissue_mask[row0 : row0 + tile_size, col0 : col0 + tile_size]
            tissue_px = int(tissue_tile.sum())
            tissue_frac = tissue_px / total_px
            if tissue_frac < background_threshold:
                continue  # mostly-background tile: excluded entirely

            artifact_tile = artifact_mask[row0 : row0 + tile_size, col0 : col0 + tile_size]
            artifact_px = int(artifact_tile.sum())
            artifact_frac = artifact_px / tissue_px if tissue_px else 0.0

            if purity is not None and (1 - purity) < artifact_frac < purity:
                continue  # ambiguous boundary tile: excluded under the "pure tiles only" variant

            label = 1 if artifact_frac >= tau else 0
            records.append(TileRecord(image_id, row0, col0, tile_size, label, tissue_frac, artifact_frac))
    return records


def tile_label_counts(records: list[TileRecord]) -> dict[str, int]:
    n_artifact = sum(r.label == 1 for r in records)
    n_clean = len(records) - n_artifact
    return {"total": len(records), "clean": n_clean, "artifact": n_artifact}


def build_stitched_mask(
    records: list[TileRecord], predicted_labels: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Paint predicted-positive tiles back onto a full-resolution boolean mask."""
    mask = np.zeros(shape, dtype=bool)
    for record, pred in zip(records, predicted_labels):
        if pred:
            mask[record.row : record.row + record.tile_size, record.col : record.col + record.tile_size] = True
    return mask
