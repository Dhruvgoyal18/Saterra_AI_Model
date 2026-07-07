"""FastAPI service wrapping detect_artifacts.

Two things this does that the plan's "public-read bucket" version doesn't
need to: the storage is a *private* HF dataset repo (per requirement), so
this service also proxies the display PNG through itself
(GET /images/{id}/display) rather than the browser fetching it directly from
the bucket -- otherwise a private bucket would be unreachable from the
browser at all. The bucket credentials (HF_TOKEN) never leave this server.
"""

import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .config import settings
from .ml.inference import detect_artifacts
from .ml.tissue import estimate_tissue_mask
from .schemas import DetectRequest, DetectResponse, ImageInfo, ImagesResponse
from .storage import ImageNotFoundError, get_storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("artifact-detector")

app = FastAPI(title="DAPI Artifact Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(settings.preprocessing_config_path) as f:
    _preprocessing_config = json.load(f)
TILE_SIZE = _preprocessing_config["tile_size"]
BACKGROUND_THRESHOLD = _preprocessing_config["background_threshold"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "storage_backend": settings.storage_backend}


@app.get("/images", response_model=ImagesResponse)
def list_images() -> ImagesResponse:
    storage = get_storage()
    try:
        manifest = storage.get_manifest()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    images = [
        ImageInfo(
            image_id=image_id,
            labeled=info["labeled"],
            level0_dims=tuple(info["level0_dims"]),
            display_dims=tuple(info["display_dims"]),
        )
        for image_id, info in sorted(manifest["images"].items())
    ]
    return ImagesResponse(images=images)


@app.get("/images/{image_id}/display")
def get_display_image(image_id: str) -> Response:
    storage = get_storage()
    try:
        png_bytes = storage.get_display_png_bytes(image_id)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=png_bytes, media_type="image/png")


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    storage = get_storage()
    try:
        image = storage.get_original_array(request.image_id)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    logger.info("Running detect_artifacts on %s (shape=%s)", request.image_id, image.shape)
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
        image_id=request.image_id,
        grid_shape=(confidence.shape[0], confidence.shape[1]),
        tile_confidence=confidence.tolist(),
        summary=result["summary"],
        level0_dims=(width, height),
        tile_size=TILE_SIZE,
    )
