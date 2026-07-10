"""OPTIONAL local debug tool -- NOT part of the app's runtime path anymore.

Historically this script was step 1 of local dev: it pre-generated every
display PNG + a manifest.json before the backend ever started. That's no
longer how this works. The backend now generates each image's display PNG
on demand, the first time it's requested, straight from the HF dataset repo
(see backend/app/storage/hf_storage.py:get_display_image_path) -- there is
no required pre-generation step, and no manifest.json anymore.

What this script is still useful for: eyeballing the percentile-normalize +
downsample output for a TIFF you already have sitting on local disk,
without going through the backend/HF path at all. It shares the exact same
image-processing code the backend uses (backend/app/ml/display_prep.py), so
what you see here matches what the backend would produce for the same file.

Usage (only if you have a local .ome.tiff to test against):
  ../mod/Scripts/python.exe scripts/prepare_display_images.py path/to/some.ome.tiff [output.png]
"""

import sys
from pathlib import Path

import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.ml.display_prep import to_display_png  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    tiff_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else tiff_path.with_suffix(".display.png")

    with tifffile.TiffFile(tiff_path) as tif:
        image = tif.series[0].levels[0].asarray()
    height, width = image.shape

    display_img, (disp_w, disp_h) = to_display_png(image)
    display_img.save(out_path, format="PNG", optimize=True)
    print(f"level0={width}x{height} -> display={disp_w}x{disp_h}  saved -> {out_path}")


if __name__ == "__main__":
    main()
