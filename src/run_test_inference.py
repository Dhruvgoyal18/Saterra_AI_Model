"""Apply detect_artifacts to the 3 unlabeled images (6, 7, 8).

No ground truth exists for these, so this is qualitative-only: an
Otsu-based tissue estimate (validated against the 5 labeled images'
ground truth in docs) stands in for the missing Tissue polygon, and the
output is a saved overlay PNG + a JSON summary per image -- there is no
number to report here, on purpose.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, hf_data
from .dataset import load_dapi_image
from .inference import detect_artifacts
from .masks import estimate_tissue_mask

DOWNSAMPLE_FOR_DISPLAY = 8


def save_overlay(image: np.ndarray, mask: np.ndarray, out_path, title: str, step: int = DOWNSAMPLE_FOR_DISPLAY) -> None:
    image_small = image[::step, ::step]
    mask_small = mask[::step, ::step]

    plt.figure(figsize=(8, 8))
    plt.imshow(image_small, cmap="gray")
    # Explicit RGBA overlay, not a colormap on a masked array -- see
    # test/data_validation.py's save_overlay for why the colormap approach
    # silently produces near-invisible tinting when every unmasked value is
    # identical (all True).
    red_overlay = np.zeros((*mask_small.shape, 4), dtype=np.float32)
    red_overlay[..., 0] = 1.0
    red_overlay[..., 3] = np.where(mask_small, 0.45, 0.0)
    plt.imshow(red_overlay)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run(image_ids: list[int] | None = None) -> dict:
    if image_ids is None:
        _, image_ids = hf_data.discover_image_ids()
    if not image_ids:
        raise ValueError("No unlabeled images found in the HF dataset repo (every image has a matching .geojson).")

    out_dir = config.OUTPUTS_DIR / "test_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for image_id in image_ids:
        print(f"\n=== Image_{image_id} (unlabeled) ===")
        image = load_dapi_image(image_id)
        tissue_mask = estimate_tissue_mask(image)
        print(f"  shape={image.shape}  estimated tissue fraction={tissue_mask.mean():.3f}")

        result = detect_artifacts(image, tissue_mask=tissue_mask)
        summary = result["summary"]
        print(
            f"  artifact area: {summary['total_artifact_area_px']} px "
            f"({summary['total_artifact_area_frac_of_tissue']:.4f} of estimated tissue), "
            f"regions={summary['num_connected_regions']}, "
            f"largest_region={summary['largest_region_area_px']} px"
        )

        overlay_path = out_dir / f"prediction_overlay_image_{image_id}.png"
        save_overlay(image, result["mask"], overlay_path, title=f"Image_{image_id}: predicted artifacts (qualitative only)")
        print(f"  Saved overlay -> {overlay_path}")

        all_summaries[f"Image_{image_id}"] = summary

    summary_path = out_dir / "test_predictions_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved combined summary -> {summary_path}")
    return all_summaries


if __name__ == "__main__":
    run()
