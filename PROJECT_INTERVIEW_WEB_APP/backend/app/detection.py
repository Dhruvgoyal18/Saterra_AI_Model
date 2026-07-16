"""The single "run the model on an image" path -- used by both POST /detect
and /chat's run_artifact_detection tool. Extracted out of main.py so there is
exactly one implementation of this path, not a second copy for the chat tool
to call over HTTP or reimplement.

Checkpoint 2 cloud-inference split: the model itself now runs in its own
hosted Modal service (../../inference_service), not in this process. That
service fetches the image itself directly from the same private HF dataset
repo (Modal enforces a hard 150s per-request timeout, and uploading the
~85MB DAPI images as a request body blew well past that -- see
../../steps_of_implementation_checkpoint2.md), so this function just tells
it which image_id to look at rather than fetching/forwarding the bytes
itself.
"""

import logging

import httpx

from . import storage
from .config import settings
from .schemas import DetectResponse

logger = logging.getLogger("artifact-detector")


class InferenceServiceError(Exception):
    """The inference endpoint (../../inference_service) is unreachable, timed
    out, or returned an error response -- a genuine external-service failure,
    distinct from storage.HFAuthError/ImageNotFoundError (which are about the
    bucket, not the model).
    """


def run_detection(image_id: str) -> DetectResponse:
    """Validate image_id against the bucket's live listing, forward it to the
    hosted inference endpoint, return the same DetectResponse shape POST
    /detect has always returned.

    Raises storage.ImageNotFoundError / storage.HFAuthError (bucket-side) or
    InferenceServiceError (inference endpoint) -- callers (the /detect route,
    chat.py's tool executor) are responsible for turning those into whatever
    response shape their own protocol needs.
    """
    storage.validate_image_id(image_id)

    logger.info("Sending %s to inference endpoint %s", image_id, settings.inference_endpoint_url)
    try:
        resp = httpx.post(
            f"{settings.inference_endpoint_url}/predict",
            params={"image_id": image_id, "prob_threshold": settings.prob_threshold},
            headers={"Authorization": f"Bearer {settings.inference_service_token}"},
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        raise InferenceServiceError(f"Could not reach inference endpoint: {exc}") from exc

    if resp.status_code != 200:
        raise InferenceServiceError(f"Inference endpoint returned {resp.status_code}: {resp.text}")

    payload = resp.json()
    return DetectResponse(
        image_id=image_id,
        grid_shape=tuple(payload["grid_shape"]),
        tile_confidence=payload["tile_confidence"],
        summary=payload["summary"],
        level0_dims=tuple(payload["level0_dims"]),
        tile_size=payload["tile_size"],
    )
