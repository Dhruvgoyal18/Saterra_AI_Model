"""The single module that talks to Hugging Face Hub. Nothing else in this
codebase should import huggingface_hub directly -- every other module goes
through the functions here.

The private HF dataset repo (`settings.hf_dataset_repo`) is the single
source of truth for image data, including for local development. There is
no local-file fallback anywhere in this module, by design -- see
../../steps_of_implementation_checkpoint2.md for the reasoning.

Two independent caches are involved, and this module only manages one of
them:
  - huggingface_hub's own cache (~/.cache/huggingface/hub, or HF_HOME) holds
    downloaded original .ome.tiff files. `hf_hub_download` is called with no
    custom `cache_dir`, deliberately -- it already handles "don't re-download
    a file we already have," so layering a second cache on top of it would
    just be redundant bookkeeping.
  - `settings.hf_display_cache_dir` (a new, small cache this module owns)
    holds derived, downsampled display PNGs plus a small dims sidecar JSON
    per image. This is the get-or-create cache described below.
"""

import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from ..config import settings
from ..ml.display_prep import to_display_png

logger = logging.getLogger("artifact-detector")

_ORIGINAL_SUFFIX = ".ome.tiff"


class ImageNotFoundError(Exception):
    pass


class HFAuthError(Exception):
    """HF_TOKEN missing, invalid, or lacking access to the dataset repo.

    Always meant to be caught at the route layer and reported as a clear,
    actionable 401/403 response -- never allowed to surface as a raw
    httpx/huggingface_hub traceback to the client.
    """


def _client_kwargs() -> dict:
    return {"repo_id": settings.hf_dataset_repo, "repo_type": "dataset", "token": settings.hf_token}


def list_images() -> list[str]:
    """List available image ids by listing files in the HF repo directly --
    not from a locally-cached manifest. Add a 9th image to the repo and it
    shows up here with no local file to regenerate.
    """
    try:
        files = HfApi(token=settings.hf_token).list_repo_files(**_client_kwargs())
    except HfHubHTTPError as exc:
        raise HFAuthError(
            f"Could not list files in HF dataset repo '{settings.hf_dataset_repo}'. "
            "Check that HF_TOKEN is set and valid, and has read access to this repo."
        ) from exc

    return sorted(f[: -len(_ORIGINAL_SUFFIX)] for f in files if f.endswith(_ORIGINAL_SUFFIX))


def get_original_path(image_id: str) -> Path:
    """Download (or reuse huggingface_hub's own cached copy of) the original
    .ome.tiff for image_id, returning its local path -- this is what
    detect_artifacts should be pointed at for inference.
    """
    try:
        path = hf_hub_download(filename=f"{image_id}{_ORIGINAL_SUFFIX}", **_client_kwargs())
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403, 404) or "RepositoryNotFound" in type(exc).__name__:
            raise HFAuthError(
                f"Could not fetch '{image_id}{_ORIGINAL_SUFFIX}' from '{settings.hf_dataset_repo}'. "
                "Check that HF_TOKEN is set and valid, and has read access to this repo."
            ) from exc
        raise ImageNotFoundError(f"No '{image_id}{_ORIGINAL_SUFFIX}' in {settings.hf_dataset_repo}") from exc
    return Path(path)


def _png_path(image_id: str) -> Path:
    return settings.hf_display_cache_dir / f"{image_id}.png"


def _dims_path(image_id: str) -> Path:
    return settings.hf_display_cache_dir / f"{image_id}.dims.json"


def get_display_image_path(image_id: str) -> Path:
    """Get-or-create the downsampled display PNG for image_id.

    Cache hit: the PNG already exists in HF_DISPLAY_CACHE_DIR -- return it
    immediately, zero HF Hub traffic.
    Cache miss: fetch the original (get_original_path, itself cached by
    huggingface_hub so a *second* image reusing the same original is still
    cheap), generate the PNG with the same percentile-normalize + downsample
    logic used everywhere else in this project (ml/display_prep.py), write
    the PNG plus a small dims sidecar next to it, and return the PNG path.
    """
    png_path = _png_path(image_id)
    if png_path.exists():
        return png_path

    settings.hf_display_cache_dir.mkdir(parents=True, exist_ok=True)

    original_path = get_original_path(image_id)

    import tifffile  # local import: keeps this module's top-level imports free of ML deps

    with tifffile.TiffFile(original_path) as tif:
        image = tif.series[0].levels[0].asarray()

    height, width = image.shape
    display_img, (disp_w, disp_h) = to_display_png(image)
    display_img.save(png_path, format="PNG", optimize=True)

    with open(_dims_path(image_id), "w") as f:
        json.dump({"level0_dims": [width, height], "display_dims": [disp_w, disp_h]}, f)

    logger.info(
        "Generated display PNG for %s (level0=%dx%d -> display=%dx%d)", image_id, width, height, disp_w, disp_h
    )
    return png_path


def get_cached_dims(image_id: str) -> dict | None:
    """{"level0_dims": [w,h], "display_dims": [w,h]} if this image's display
    has been generated at least once, else None. Read-only -- never triggers
    a download or a generation. Used by GET /images so listing images stays
    cheap (no forced fetch of every original just to report dimensions).
    """
    dims_path = _dims_path(image_id)
    if not dims_path.exists():
        return None
    with open(dims_path) as f:
        return json.load(f)
