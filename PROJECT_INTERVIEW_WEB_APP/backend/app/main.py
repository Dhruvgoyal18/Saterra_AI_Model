"""FastAPI service wrapping detect_artifacts.

The Hugging Face dataset repo is the single source of truth for image data,
in every environment including local dev -- see storage/hf_storage.py and
steps_of_implementation_checkpoint2.md for why. Since that repo is private,
this service also proxies the display PNG through itself
(GET /images/{id}/display) rather than the browser fetching it directly from
the bucket -- otherwise a private bucket would be unreachable from the
browser at all. HF_TOKEN never leaves this server.
"""

import json
import logging

import httpx
import tifffile
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import storage
from .config import settings
from .ml.inference import detect_artifacts
from .ml.tissue import estimate_tissue_mask
from .schemas import DetectRequest, DetectResponse, ImageInfo, ImagesResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("artifact-detector")

# Fail fast and clearly at startup if the repo is private and no credential
# was given, rather than letting a confusing httpx/huggingface_hub traceback
# surface from deep inside the first request (acceptance criterion: an
# auth-config problem must read as "you forgot to set this," not a bug).
if not settings.hf_token or not settings.hf_dataset_repo:
    raise RuntimeError(
        "HF_TOKEN and HF_DATASET_REPO must both be set -- the image dataset repo is "
        "private, so there is no local-data fallback. Set them in backend/.env "
        "(copy backend/.env.example) for local dev, or as secrets on your hosting "
        "platform for a deployment."
    )

if not settings.supabase_url or not settings.supabase_anon_key:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_ANON_KEY must both be set -- every image/detect "
        "endpoint is gated behind a Supabase-issued session token (see require_user "
        "below). Set them in backend/.env for local dev, or as secrets/variables on "
        "your hosting platform for a deployment."
    )

app = FastAPI(title="DAPI Artifact Detector", version="2.0.0")

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


_http_client = httpx.AsyncClient()


async def require_user(authorization: str | None = Header(None)) -> dict:
    """Verify the Supabase-issued session token on every gated request.

    This is the actual access boundary, not the frontend's login screen --
    anyone can call this API directly with curl/Postman, bypassing the
    frontend entirely, so the frontend's own auth check is UX only and this
    dependency is what actually enforces invite-only access.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ")
    resp = await _http_client.get(
        f"{settings.supabase_url}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": settings.supabase_anon_key},
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return resp.json()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "hf_dataset_repo": settings.hf_dataset_repo}


@app.get("/images", response_model=ImagesResponse, dependencies=[Depends(require_user)])
def list_images() -> ImagesResponse:
    try:
        image_ids = storage.list_images()
    except storage.HFAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    images = []
    for image_id in image_ids:
        dims = storage.get_cached_dims(image_id)  # None until first display fetch -- doesn't force one
        images.append(
            ImageInfo(
                image_id=image_id,
                level0_dims=tuple(dims["level0_dims"]) if dims else None,
                display_dims=tuple(dims["display_dims"]) if dims else None,
            )
        )
    return ImagesResponse(images=images)


@app.get("/images/{image_id}/display", dependencies=[Depends(require_user)])
def get_display_image(image_id: str) -> FileResponse:
    try:
        png_path = storage.get_display_image_path(image_id)
    except storage.HFAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except storage.ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(png_path, media_type="image/png")


@app.post("/detect", response_model=DetectResponse, dependencies=[Depends(require_user)])
def detect(request: DetectRequest) -> DetectResponse:
    try:
        original_path = storage.get_original_path(request.image_id)
    except storage.HFAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except storage.ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    with tifffile.TiffFile(original_path) as tif:
        image = tif.series[0].levels[0].asarray()

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
