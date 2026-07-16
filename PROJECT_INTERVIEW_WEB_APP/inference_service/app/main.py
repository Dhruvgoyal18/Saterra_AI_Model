"""Model-serving endpoint: image_id in, predictions out.

Deployed as its own Modal app, independent of the web backend. Modal
enforces a hard 150s timeout on any single HTTP request -- directly
uploading the ~85MB DAPI images as a request body blew well past that
window (measured: even a 10MB upload didn't finish in 90-120s). So this
service fetches the image itself, directly from the same private HF
dataset repo the backend uses, rather than receiving it as an upload --
the backend just tells it which image_id to look at, keeping every request
tiny regardless of image size.

Gated by a bearer token check against INFERENCE_TOKEN (injected by Modal via
a modal.Secret at deploy time -- see modal_app.py). /health stays open for
uptime checks without needing the token.
"""

import json
import logging
import os
from pathlib import Path

import tifffile
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from huggingface_hub import hf_hub_download

from .inference import detect_artifacts
from .tissue import estimate_tissue_mask

logger = logging.getLogger("inference-service")

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_WEIGHTS_PATH = _WEIGHTS_DIR / "model_final.pt"
_ORIGINAL_SUFFIX = ".ome.tiff"

with open(_WEIGHTS_DIR / "preprocessing_config.json") as f:
    _preprocessing_config = json.load(f)
TILE_SIZE = _preprocessing_config["tile_size"]
BACKGROUND_THRESHOLD = _preprocessing_config["background_threshold"]

app = FastAPI(title="saterra-model-inference")


def require_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("INFERENCE_TOKEN")
    if not expected:
        raise HTTPException(status_code=500, detail="INFERENCE_TOKEN not configured on this service")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", dependencies=[Depends(require_token)])
def predict(image_id: str = Query(...), prob_threshold: float = Query(default=0.7)) -> dict:
    hf_token = os.environ.get("HF_TOKEN")
    hf_dataset_repo = os.environ.get("HF_DATASET_REPO")
    if not hf_token or not hf_dataset_repo:
        raise HTTPException(status_code=500, detail="HF_TOKEN/HF_DATASET_REPO not configured on this service")

    try:
        original_path = hf_hub_download(
            filename=f"{image_id}{_ORIGINAL_SUFFIX}", repo_id=hf_dataset_repo, repo_type="dataset", token=hf_token
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Could not fetch '{image_id}': {exc}") from exc

    try:
        with tifffile.TiffFile(original_path) as tif:
            image = tif.series[0].levels[0].asarray()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode TIFF: {exc}") from exc

    logger.info("Running detect_artifacts on %s (shape=%s, prob_threshold=%s)", image_id, image.shape, prob_threshold)
    tissue_mask = estimate_tissue_mask(image)
    result = detect_artifacts(
        image,
        weights_path=_WEIGHTS_PATH,
        tissue_mask=tissue_mask,
        tile_size=TILE_SIZE,
        background_threshold=BACKGROUND_THRESHOLD,
        prob_threshold=prob_threshold,
    )

    confidence = result["per_tile_confidence"]
    height, width = image.shape
    return {
        "grid_shape": [confidence.shape[0], confidence.shape[1]],
        "tile_confidence": confidence.tolist(),
        "summary": result["summary"],
        "level0_dims": [width, height],
        "tile_size": TILE_SIZE,
    }
