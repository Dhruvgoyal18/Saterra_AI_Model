"""Phase A: generate display PNGs + manifest.json for all 8 DAPI images.

Run with the Checkpoint 1 venv, which already has tifffile/PIL/numpy:
  ../mod/Scripts/python.exe scripts/prepare_display_images.py   (from this dir)

Originals are deliberately NOT copied here -- for local dev, LocalStorage
(backend/app/storage.py) reads them directly from the sibling
../DAPI_images/ folder, avoiding a ~1.4GB duplicate. Only two things are
produced per image: a downsampled, percentile-normalized display PNG, and an
entry in manifest.json recording level0_dims/display_dims, which is what
lets the frontend map a confidence-grid cell back to an exact display-pixel
rectangle (a pure scale factor, since the PNG is a pure downsample of level 0).

scripts/upload_to_hf.py uploads this same bucket_staging/ output (plus the
real originals) to a private Hugging Face Hub dataset repo for production.
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent  # PROJECT_INTERVIEW_WEB_APP/
ORIGINALS_DIR = ROOT.parent / "DAPI_images"
OUTPUT_DIR = ROOT / "backend" / "bucket_staging"

LABELED_IDS = {1, 2, 3, 4, 5}
ALL_IDS = range(1, 9)

TARGET_LONG_SIDE = 2000
LOW_PCTILE, HIGH_PCTILE = 1.0, 99.0


def load_level0(image_id: int) -> np.ndarray:
    path = ORIGINALS_DIR / f"Image_{image_id}.ome.tiff"
    with tifffile.TiffFile(path) as tif:
        return tif.series[0].levels[0].asarray()


def to_display_png(image: np.ndarray) -> tuple[Image.Image, tuple[int, int]]:
    """Same 1st/99th-percentile normalization used everywhere else in this
    project (Checkpoint 1's dataset.py, this backend's preprocessing.py) --
    reused here so the display image looks like what the model actually sees,
    not an independently-chosen display stretch.
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


def main() -> None:
    manifest = {"images": {}}
    for image_id in ALL_IDS:
        print(f"Processing Image_{image_id}...")
        image = load_level0(image_id)
        height, width = image.shape

        display_img, (disp_w, disp_h) = to_display_png(image)

        image_dir = OUTPUT_DIR / f"image_{image_id}"
        image_dir.mkdir(parents=True, exist_ok=True)
        display_path = image_dir / "display.png"
        display_img.save(display_path, format="PNG", optimize=True)

        manifest["images"][f"image_{image_id}"] = {
            "level0_dims": [width, height],
            "display_dims": [disp_w, disp_h],
            "labeled": image_id in LABELED_IDS,
        }
        print(f"  level0={width}x{height} -> display={disp_w}x{disp_h}  saved -> {display_path}")

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
