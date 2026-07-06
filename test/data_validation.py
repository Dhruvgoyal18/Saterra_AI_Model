"""
QC check: verify that each DAPI OME-TIFF and its Tissue annotation
polygon are aligned before they get used downstream (tiling, labeling).

For every image with a matching annotation file, this script:
  1. Reports the pyramid levels stored in the file (shape + dtype).
  2. Loads level 0 explicitly (never trusts tifffile.imread's default series).
  3. Rasterizes the "Tissue" polygon at level 0's resolution.
  4. Sanity-checks the polygon bounding box against the image shape (catches
     wrong-pyramid-level bugs).
  5. Checks whether the (x, y) axis assignment is unambiguous given the image's
     dimensions, i.e. whether a row/col swap would put points out of bounds.
  6. Checks that DAPI intensity is substantially brighter inside the mask than
     outside it -- a correct mask sits on real tissue signal, a swapped/flipped
     one lands on close-to-random image content and shows little contrast.
  7. Saves a DAPI + mask overlay PNG as final, human-eyeballed confirmation.

Run directly: python test/data_validation.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; script only writes PNGs, never shows one
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DAPI_DIR = ROOT / "DAPI_images"
ANNOTATION_DIR = ROOT / "artifact_annotations"
OUTPUT_DIR = ROOT / "test" / "qc_overlays"

TISSUE_CLASS_NAME = "Tissue"
DOWNSAMPLE_FOR_DISPLAY = 8  # only affects PNG size/speed, not correctness
BBOX_TOLERANCE = 0.05  # allowed fractional deviation of polygon bbox vs image shape
MIN_INTENSITY_CONTRAST = 2.0  # mean_inside / mean_outside must clear this to pass


def get_pyramid_levels(tiff_path: Path) -> list[tuple[tuple[int, ...], np.dtype]]:
    """Return (shape, dtype) for every pyramid level in series 0."""
    with tifffile.TiffFile(tiff_path) as tif:
        return [(level.shape, level.dtype) for level in tif.series[0].levels]


def load_level(tiff_path: Path, level: int = 0) -> np.ndarray:
    """Load one explicit pyramid level -- never rely on tifffile.imread's default."""
    with tifffile.TiffFile(tiff_path) as tif:
        return tif.series[0].levels[level].asarray()


def load_tissue_polygon(geojson_path: Path) -> np.ndarray:
    """Return the Tissue polygon's outer ring as an (N, 2) array of (x, y) points."""
    with open(geojson_path) as f:
        data = json.load(f)

    tissue_features = [
        feat
        for feat in data["features"]
        if feat["properties"].get("classification", {}).get("name") == TISSUE_CLASS_NAME
    ]
    if not tissue_features:
        raise ValueError(f"No '{TISSUE_CLASS_NAME}' feature found in {geojson_path.name}")
    if len(tissue_features) > 1:
        raise ValueError(
            f"Expected exactly one '{TISSUE_CLASS_NAME}' feature in {geojson_path.name}, "
            f"found {len(tissue_features)}"
        )

    outer_ring = tissue_features[0]["geometry"]["coordinates"][0]
    return np.array(outer_ring, dtype=np.float64)  # columns are (x, y)


def rasterize_polygon(polygon_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize an (x, y) polygon onto a boolean mask of the given (H, W) shape.

    Uses PIL's ImageDraw, which takes points as (x, y) pairs directly -- unlike
    skimage.draw.polygon, which wants (row, col) i.e. (y, x) and will silently
    produce a transposed mask if the axes are passed in the wrong order.
    """
    height, width = shape
    mask_img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_img).polygon([tuple(pt) for pt in polygon_xy], fill=255)
    return np.array(mask_img, dtype=bool)


def check_bounding_box(
    polygon_xy: np.ndarray, shape: tuple[int, int], tolerance: float = BBOX_TOLERANCE
) -> tuple[bool, float, float]:
    """Sanity check: the polygon bbox should roughly match the image shape.

    A large mismatch here usually means the wrong pyramid level was rasterized
    against (mask much smaller/larger than the image), as opposed to an axis
    swap, which keeps the bbox size the same but rotates/mirrors the shape.
    """
    height, width = shape
    poly_w = polygon_xy[:, 0].max() - polygon_xy[:, 0].min()
    poly_h = polygon_xy[:, 1].max() - polygon_xy[:, 1].min()

    w_ratio = poly_w / width
    h_ratio = poly_h / height
    ok = (1 - tolerance) <= w_ratio <= (1 + tolerance) and (1 - tolerance) <= h_ratio <= (1 + tolerance)
    return ok, w_ratio, h_ratio


def check_axis_assignment(polygon_xy: np.ndarray, shape: tuple[int, int]) -> tuple[str, bool | None]:
    """Check whether (x, y) -> (col, row) is the only assignment that fits in bounds.

    Returns ("confirmed", True/False) when the image is non-square and only one
    axis assignment keeps every point in bounds -- a swap would then index out
    of range. Returns ("inconclusive", None) when both assignments fit (e.g. a
    square image, or coordinates well within both dimensions), in which case
    this check can't rule out a swap and check_intensity_contrast should decide.
    """
    height, width = shape
    x_max, y_max = polygon_xy[:, 0].max(), polygon_xy[:, 1].max()

    fits_as_xy = x_max <= width and y_max <= height  # correct: x->col, y->row
    fits_as_yx = x_max <= height and y_max <= width  # swapped: x->row, y->col

    if not fits_as_xy:
        return "confirmed", False
    if fits_as_yx:
        return "inconclusive", None
    return "confirmed", True


def check_intensity_contrast(
    dapi: np.ndarray, mask: np.ndarray, min_ratio: float = MIN_INTENSITY_CONTRAST
) -> tuple[bool, float, float, float]:
    """A correctly-placed tissue mask should sit on real DAPI signal.

    A swapped or flipped mask lands on close-to-random image content, so the
    inside/outside intensity ratio comes out near 1.0 instead of clearly > 1.0.
    """
    mean_inside = float(dapi[mask].mean())
    mean_outside = float(dapi[~mask].mean())
    ratio = mean_inside / mean_outside
    return ratio >= min_ratio, mean_inside, mean_outside, ratio


def save_overlay(dapi: np.ndarray, mask: np.ndarray, out_path: Path, step: int = DOWNSAMPLE_FOR_DISPLAY) -> None:
    dapi_small = dapi[::step, ::step]
    mask_small = mask[::step, ::step]

    plt.figure(figsize=(8, 8))
    plt.imshow(dapi_small, cmap="gray")
    # Explicit RGBA overlay (red where mask, alpha=0 elsewhere) rather than a
    # colormap on a masked array: when every unmasked value is identical (all
    # True), matplotlib's auto-normalization degenerates and the "Reds"
    # colormap barely tints anything -- this guarantees a visible red regardless.
    red_overlay = np.zeros((*mask_small.shape, 4), dtype=np.float32)
    red_overlay[..., 0] = 1.0
    red_overlay[..., 3] = np.where(mask_small, 0.45, 0.0)
    plt.imshow(red_overlay)
    plt.title(out_path.stem)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def validate_image(image_id: int) -> None:
    tiff_path = DAPI_DIR / f"Image_{image_id}.ome.tiff"
    geojson_path = ANNOTATION_DIR / f"artifact_annotation_image_{image_id}.geojson"

    print(f"\n=== Image_{image_id} ===")
    for i, (shape, dtype) in enumerate(get_pyramid_levels(tiff_path)):
        print(f"  Level {i}: shape={shape}, dtype={dtype}")

    dapi = load_level(tiff_path, level=0)
    polygon_xy = load_tissue_polygon(geojson_path)
    mask = rasterize_polygon(polygon_xy, dapi.shape)

    ok, w_ratio, h_ratio = check_bounding_box(polygon_xy, dapi.shape)
    status = "OK" if ok else "MISMATCH"
    print(f"  Tissue bbox vs image shape: width_ratio={w_ratio:.3f}, height_ratio={h_ratio:.3f} [{status}]")
    if not ok:
        print("  -> ratio far from 1.0 usually means the wrong pyramid level was used, not an axis swap.")

    axis_result, axis_ok = check_axis_assignment(polygon_xy, dapi.shape)
    if axis_result == "confirmed":
        print(f"  Axis assignment (x=col, y=row): {'OK, unambiguous' if axis_ok else 'FAIL, out of bounds'}")
    else:
        print("  Axis assignment: inconclusive from bounds alone (both orderings fit) -- relying on intensity check")

    contrast_ok, mean_in, mean_out, ratio = check_intensity_contrast(dapi, mask)
    contrast_status = "OK" if contrast_ok else "FAIL"
    print(
        f"  DAPI intensity inside/outside mask: {mean_in:.1f} / {mean_out:.1f} "
        f"(ratio={ratio:.2f}) [{contrast_status}]"
    )
    if not contrast_ok:
        print("  -> low contrast suggests the mask isn't sitting on real tissue signal (swap/flip/rotation).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"check_alignment_image_{image_id}.png"
    save_overlay(dapi, mask, out_path)
    print(f"  Saved overlay -> {out_path}")


def main() -> None:
    geojson_files = sorted(ANNOTATION_DIR.glob("artifact_annotation_image_*.geojson"))
    image_ids = sorted(int(p.stem.rsplit("_", 1)[-1]) for p in geojson_files)

    if not image_ids:
        raise SystemExit(f"No annotation files found in {ANNOTATION_DIR}")

    for image_id in image_ids:
        tiff_path = DAPI_DIR / f"Image_{image_id}.ome.tiff"
        if not tiff_path.exists():
            print(f"Skipping Image_{image_id}: no matching TIFF at {tiff_path}")
            continue
        validate_image(image_id)


if __name__ == "__main__":
    main()
