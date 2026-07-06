"""Central configuration for the artifact-detection pipeline (Checkpoint 1).

Every path and hyperparameter that other modules need is defined here so the
whole pipeline can be re-tuned (tile size, tau, thresholds) from one place.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAPI_DIR = ROOT / "DAPI_images"
ANNOTATION_DIR = ROOT / "artifact_annotations"
ARTIFACTS_DIR = ROOT / "artifacts"  # trained weights + preprocessing config + CV results
OUTPUTS_DIR = ROOT / "outputs"  # overlays, predictions, reports (visual, human-facing)

TISSUE_CLASS_NAME = "Tissue"
ARTIFACT_CLASS_PREFIX = "Artifact"  # matches "Artifact - folds", "Artifact - *", etc.

# --- Tiling / labeling ---
TILE_SIZE = 64  # see docs: 64px chosen as the fast baseline; 32px compared for the tradeoff
TAU = 0.5  # artifact-pixel fraction *of tissue pixels in the tile* needed to label it "artifact"
BACKGROUND_THRESHOLD = 0.5  # a tile needs >= this much tissue coverage to be kept at all

# --- Intensity normalization ---
INTENSITY_LOW_PCTILE = 1.0
INTENSITY_HIGH_PCTILE = 99.0

# --- Data splits ---
LABELED_IMAGE_IDS = [1, 2, 3, 4, 5]
UNLABELED_IMAGE_IDS = [6, 7, 8]

RANDOM_SEED = 42

# --- Training defaults ---
# 3 epochs / batch_size=512 chosen for CPU-only training time (~15-20 min for the
# full 5-fold LOIO CV + final model): the spec is explicit that Checkpoint 1 is
# graded on methodology, not on squeezing out maximum accuracy, so training
# budget here is deliberately capped rather than tuned for best possible score.
EPOCHS = 3
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

MODEL_WEIGHTS_PATH = ARTIFACTS_DIR / "model_final.pt"
PREPROCESSING_CONFIG_PATH = ARTIFACTS_DIR / "preprocessing_config.json"
LOIO_RESULTS_PATH = ARTIFACTS_DIR / "loio_cv_results.json"
