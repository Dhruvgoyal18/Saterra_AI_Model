"""FastAPI service wrapping detect_artifacts.

The Hugging Face dataset repo is the single source of truth for image data,
in every environment including local dev -- see storage/hf_storage.py and
steps_of_implementation_checkpoint2.md for why. Since that repo is private,
this service also proxies the display PNG through itself
(GET /images/{id}/display) rather than the browser fetching it directly from
the bucket -- otherwise a private bucket would be unreachable from the
browser at all. HF_TOKEN never leaves this server.
"""

import logging

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from . import chat as chat_module
from . import storage
from .config import settings
from .detection import run_detection
from .schemas import ChatRequest, ChatResponse, DetectRequest, DetectResponse, ImageInfo, ImagesResponse

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

_http_client = httpx.AsyncClient(timeout=10.0)


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
    try:
        resp = await _http_client.get(
            f"{settings.supabase_url}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": settings.supabase_anon_key},
        )
    except httpx.TransportError as exc:
        # A transient network hiccup reaching Supabase (seen in practice: a
        # slow response tripped httpx's default timeout) previously surfaced
        # as an unhandled 500 with a raw traceback -- this is a real,
        # retryable condition, not an invalid session, so it gets its own
        # clear 503 instead of being indistinguishable from a code bug.
        raise HTTPException(status_code=503, detail="Auth service temporarily unavailable -- try again") from exc
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


@app.get("/images/{image_id}/dzi.dzi", dependencies=[Depends(require_user)])
def get_dzi_descriptor(image_id: str) -> Response:
    """Deep Zoom XML descriptor for image_id -- what OpenSeadragon's
    `tileSources` option points at to open a real pan/zoom tile pyramid,
    generated server-side since these .ome.tiff files have no native
    multi-resolution pyramid of their own (see ml/tiling.py)."""
    try:
        xml = storage.get_dzi_descriptor_xml(image_id)
    except storage.HFAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except storage.ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=xml, media_type="application/xml")


@app.get("/images/{image_id}/dzi_files/{level}/{tile_name}", dependencies=[Depends(require_user)])
def get_dzi_tile(image_id: str, level: int, tile_name: str) -> FileResponse:
    """One deep-zoom tile, e.g. tile_name="3_5.jpeg" -- DZI's own filename
    convention (col_row.ext), parsed by hand since it isn't a FastAPI
    path-converter pattern."""
    try:
        col_str, row_str = tile_name.removesuffix(".jpeg").split("_")
        col, row = int(col_str), int(row_str)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Malformed tile name '{tile_name}'") from exc

    try:
        tile_path = storage.get_or_build_tile(image_id, level, col, row)
    except storage.HFAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except storage.ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except storage.InvalidTileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(tile_path, media_type="image/jpeg")


@app.post("/detect", response_model=DetectResponse, dependencies=[Depends(require_user)])
def detect(request: DetectRequest) -> DetectResponse:
    try:
        return run_detection(request.image_id)
    except storage.HFAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except storage.ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_user)])
def chat(request: ChatRequest) -> ChatResponse:
    """Checkpoint 3: natural-language trigger for the same run_detection path
    /detect uses, via Groq function-calling (see chat.py). Gated by the same
    require_user dependency as every other data-bearing route -- no new auth
    model for this endpoint.

    storage.ImageNotFoundError/HFAuthError from a bad tool-call argument are
    deliberately NOT caught here -- chat.py's tool executor already turns
    those into a tool-result message the LLM phrases as a natural-language
    decline, which is the whole point of routing them through the model
    rather than surfacing an HTTP error code for what's really a
    conversational turn. Only the LLM call itself (a genuine external-service
    failure, not a user-input problem) gets a clean error here.
    """
    try:
        return chat_module.handle_chat_message(request.message, request.conversation_id)
    except Exception as exc:
        logger.exception("Chat completion failed")
        raise HTTPException(status_code=502, detail=f"Chat service temporarily unavailable: {exc}") from exc
