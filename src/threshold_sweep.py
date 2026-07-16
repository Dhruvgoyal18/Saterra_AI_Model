"""Pick a decision threshold from real out-of-fold precision/recall data,
instead of the default prob_threshold=0.5 baked into the web app.

Reuses the same LOIO folds as train.run_loio_cv() (train.py's docstring
explains why image-level folds, not tile-level, are the leak-free split
here) so this is an honest generalization estimate, not resubstitution:
model_final.pt is trained on all 5 labeled images and can't be validly
evaluated against any of them, but each fold's held-out prediction comes
from a model that never saw that image. Pooling those out-of-fold
(y_true, y_prob) pairs across all 5 folds gives one PR curve computed on
scores the model was never trained against.
"""

import json

import numpy as np
import torch

from . import config, hf_data
from .metrics import sweep_thresholds
from .train import _predict, _train_model, prepare_image_data


def run_threshold_sweep(
    image_ids: list[int] | None = None,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    device: str = "cpu",
    seed: int = config.RANDOM_SEED,
) -> dict:
    if image_ids is None:
        image_ids, _ = hf_data.discover_image_ids()
    if not image_ids:
        raise ValueError("No labeled images found in the HF dataset repo (no matching .geojson files).")

    torch.manual_seed(seed)

    print("Preparing tile records for all labeled images...")
    images, norm_stats, _, tile_records = prepare_image_data(image_ids)

    y_true_all, y_prob_all = [], []
    for held_out in image_ids:
        print(f"\n--- Fold: held-out = Image_{held_out} ---")
        train_ids = [i for i in image_ids if i != held_out]
        train_records = [r for i in train_ids for r in tile_records[i]]
        val_records = tile_records[held_out]

        model = _train_model(train_records, images, norm_stats, epochs, batch_size, lr, device)
        y_true, y_prob = _predict(model, val_records, images, norm_stats, batch_size, device)
        y_true_all.append(y_true)
        y_prob_all.append(y_prob)

    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)

    sweep = sweep_thresholds(y_true_all, y_prob_all)
    best_f1 = max(sweep, key=lambda row: row["f1"])
    precision_50 = next((row for row in sweep if row["precision"] >= 0.5), None)

    results = {
        "sweep": sweep,
        "recommended_threshold": best_f1["threshold"],
        "recommended_criterion": "argmax F1",
        "recommended_metrics": best_f1,
        "first_precision_ge_0.5": precision_50,
        "n_pooled_tiles": int(len(y_true_all)),
        "n_pooled_artifact_tiles": int(y_true_all.sum()),
        "held_out_images": image_ids,
        "config": {
            "tile_size": config.TILE_SIZE,
            "tau": config.TAU,
            "background_threshold": config.BACKGROUND_THRESHOLD,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
    }
    return results


def save_results(results: dict, path=config.ARTIFACTS_DIR / "threshold_sweep_results.json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    results = run_threshold_sweep()
    save_results(results)

    print(f"\n=== Threshold sweep (pooled out-of-fold, {results['n_pooled_tiles']} tiles) ===")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for row in results["sweep"]:
        print(f"{row['threshold']:>10.2f} {row['precision']:>10.3f} {row['recall']:>10.3f} {row['f1']:>10.3f}")

    print(f"\nRecommended threshold ({results['recommended_criterion']}): {results['recommended_threshold']:.2f}")
    rm = results["recommended_metrics"]
    print(f"  precision={rm['precision']:.3f} recall={rm['recall']:.3f} f1={rm['f1']:.3f}")
    if results["first_precision_ge_0.5"]:
        p50 = results["first_precision_ge_0.5"]
        print(
            f"Lowest threshold with precision>=0.5: {p50['threshold']:.2f} "
            f"(precision={p50['precision']:.3f} recall={p50['recall']:.3f} f1={p50['f1']:.3f})"
        )
    else:
        print("No threshold in the sweep reaches precision>=0.5.")

    print(f"\nSaved -> {config.ARTIFACTS_DIR / 'threshold_sweep_results.json'}")
