"""The single "run the model on an image" path -- used by both POST /detect
and /chat's run_artifact_detection tool. Extracted out of main.py so there is
exactly one implementation of this path, not a second copy for the chat tool
to call over HTTP or reimplement.
"""

import json
import logging

import tifffile

from . import storage
from .config import settings
from .ml.inference import detect_artifacts
from .ml.tissue import estimate_tissue_mask
from .schemas import DetectResponse

logger = logging.getLogger("artifact-detector")

with open(settings.preprocessing_config_path) as f:
    _preprocessing_config = json.load(f)
TILE_SIZE = _preprocessing_config["tile_size"]
BACKGROUND_THRESHOLD = _preprocessing_config["background_threshold"]


def run_detection(image_id: str) -> DetectResponse:
    """Fetch image_id's original, run detect_artifacts, return the same
    DetectResponse shape POST /detect has always returned.

    Raises storage.ImageNotFoundError / storage.HFAuthError -- callers (the
    /detect route, chat.py's tool executor) are responsible for turning
    those into whatever response shape their own protocol needs (HTTP status
    codes for /detect, a tool-result message for /chat).
    """
    original_path = storage.get_original_path(image_id)

    with tifffile.TiffFile(original_path) as tif:
        image = tif.series[0].levels[0].asarray()

    logger.info("Running detect_artifacts on %s (shape=%s)", image_id, image.shape)
    tissue_mask = estimate_tissue_mask(image)
    result = detect_artifacts(
        image,
        weights_path=settings.model_weights_path,
        tissue_mask=tissue_mask,
        tile_size=TILE_SIZE,
        background_threshold=BACKGROUND_THRESHOLD,
        prob_threshold=settings.prob_threshold,
    )

    confidence = result["per_tile_confidence"]
    height, width = image.shape
    return DetectResponse(
        image_id=image_id,
        grid_shape=(confidence.shape[0], confidence.shape[1]),
        tile_confidence=confidence.tolist(),
        summary=result["summary"],
        level0_dims=(width, height),
        tile_size=TILE_SIZE,
    )
