"""GeoJSON -> pixel mask conversion.

Builds four full-resolution boolean masks per image: tissue, artifact
(union of every "Artifact - *" class), clean_tissue (tissue minus artifact),
and background (everything outside tissue).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import config


def load_features(geojson_path: Path) -> list[dict]:
    with open(geojson_path) as f:
        data = json.load(f)
    return data["features"]


def classify_feature(feature: dict) -> str:
    """Tissue / artifact / other, based on properties.classification.name.

    Matching by name prefix ("Artifact" -> artifact) rather than hard-coding
    "Artifact - folds" so this generalizes to other artifact subtypes
    (e.g. "Artifact - bubbles") without code changes -- the spec explicitly
    asks to collapse every artifact subtype into one positive class.
    """
    name = feature.get("properties", {}).get("classification", {}).get("name", "")
    if name == config.TISSUE_CLASS_NAME:
        return "tissue"
    if name.startswith(config.ARTIFACT_CLASS_PREFIX):
        return "artifact"
    return "other"


def _polygon_rings(geometry: dict) -> list[list[list[float]]]:
    """Normalize Polygon/MultiPolygon geometry into a list of polygons,
    each a list of rings (ring 0 = exterior, rest = holes).

    None of the 5 provided annotation files contain holes or MultiPolygons
    (verified: every feature is a single-ring Polygon), but real QuPath/GeoJSON
    exports can contain either, so both are handled rather than assumed away.
    """
    geom_type = geometry["type"]
    if geom_type == "Polygon":
        return [geometry["coordinates"]]
    if geom_type == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError(f"Unsupported geometry type for a tissue/artifact annotation: {geom_type}")


def rasterize_features(features: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Union-rasterize a list of features into one boolean mask at `shape` = (H, W).

    Uses PIL's ImageDraw, whose polygon() takes (x, y) points directly -- this
    matches GeoJSON's (x, y) convention with no manual row/col swap, avoiding
    the classic skimage.draw.polygon (row, col) pitfall (see
    test/data_validation.py for the explicit checks that ruled this out for
    the 5 labeled images).

    Holes are punched by drawing each hole ring with fill=0 after the exterior
    ring is drawn with fill=255 -- correct for simple (non-self-overlapping)
    polygons, which is what these annotations are.
    """
    height, width = shape
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for feature in features:
        for rings in _polygon_rings(feature["geometry"]):
            exterior, *holes = rings
            draw.polygon([tuple(p) for p in exterior], fill=255)
            for hole in holes:
                draw.polygon([tuple(p) for p in hole], fill=0)
    return np.array(mask_img, dtype=bool)


def build_masks(geojson_path: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Build tissue / artifact / clean_tissue / background masks for one image."""
    features = load_features(geojson_path)
    tissue_features = [f for f in features if classify_feature(f) == "tissue"]
    artifact_features = [f for f in features if classify_feature(f) == "artifact"]

    if not tissue_features:
        raise ValueError(f"No '{config.TISSUE_CLASS_NAME}' feature found in {geojson_path.name}")

    tissue_mask = rasterize_features(tissue_features, shape)
    if artifact_features:
        artifact_mask = rasterize_features(artifact_features, shape)
    else:
        artifact_mask = np.zeros(shape, dtype=bool)
    artifact_mask &= tissue_mask  # artifacts are annotated within tissue; clip defensively

    clean_tissue_mask = tissue_mask & ~artifact_mask
    background_mask = ~tissue_mask

    return {
        "tissue": tissue_mask,
        "artifact": artifact_mask,
        "clean_tissue": clean_tissue_mask,
        "background": background_mask,
    }


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

    # Otsu: maximize inter-class variance between the "below" and "above" splits
    inter_class_var = weight_below[:-1] * weight_above[1:] * (mean_below[:-1] - mean_above[1:]) ** 2
    return float(mids[int(np.argmax(inter_class_var))])


def _block_mean(image: np.ndarray, block: int) -> np.ndarray:
    height, width = image.shape
    h2, w2 = (height // block) * block, (width // block) * block
    cropped = image[:h2, :w2]
    return cropped.reshape(h2 // block, block, w2 // block, block).mean(axis=(1, 3))


def estimate_tissue_mask(image: np.ndarray, block: int = config.TILE_SIZE) -> np.ndarray:
    """Approximate a tissue foreground mask, for the 3 *unlabeled* test images
    which have no ground-truth Tissue polygon to rasterize.

    Naive per-pixel Otsu fails badly here (validated at IoU ~0.05 against the
    5 labeled images' ground truth): DAPI images are a 3-population mixture --
    black slide background, mid-gray tissue, and a long tail of very bright
    individual nuclei -- and plain Otsu on raw pixel values latches onto the
    nuclei-vs-tissue split instead of the background-vs-tissue split.

    Fix: (1) block-average the image (default block=tile_size) to smooth out
    nuclei speckle, collapsing the distribution to a cleaner two-population
    background/tissue split, then (2) run Otsu on the *log* of those block
    means, since the tissue population's long right tail otherwise still
    skews the variance-maximizing split away from the true boundary. This
    combination validated at IoU 0.92-0.96 against ground truth across all 5
    labeled images (plain per-pixel Otsu: ~0.05; Otsu on raw block means,
    without the log transform: 0.37-0.77, unstable). Still an approximation,
    not a substitute for a real annotation -- see docs for full validation.
    """
    block_means = _block_mean(image, block)
    threshold = _otsu_threshold_1d(np.log1p(block_means).ravel())
    block_mask = np.log1p(block_means) > threshold

    mask = np.repeat(np.repeat(block_mask, block, axis=0), block, axis=1)
    full_mask = np.zeros(image.shape, dtype=bool)
    h2, w2 = mask.shape
    full_mask[:h2, :w2] = mask
    return full_mask
