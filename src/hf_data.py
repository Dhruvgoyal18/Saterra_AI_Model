"""The one module in src/ that talks to Hugging Face Hub directly.

Mirrors the role of PROJECT_INTERVIEW_WEB_APP/backend/app/storage/hf_storage.py
for this separate, local-training context -- deliberately not imported from
there. This project's established "vendor, don't import across project
trees" precedent applies here too: src/ runs on a local dev machine for
training, backend/ runs in a Docker container; they're different deployable/
runnable contexts and shouldn't share code across the boundary.

HF_TOKEN and HF_DATASET_REPO must be set as environment variables before
running (export them, or however your shell already has them set) -- plain
os.environ.get(), no python-dotenv, matching the one other script outside
the web app that already talks to this repo
(PROJECT_INTERVIEW_WEB_APP/scripts/upload_to_hf.py). Deliberately not reading
the project-root .env: that file is currently tracked by git with no
.gitignore protection (unlike the web app's own .env files), so wiring a
config loader to it would risk a real token landing in a commit.

hf_hub_download is called with no custom cache_dir -- reuses
huggingface_hub's own download cache rather than building a second one on
top, same reasoning already established in hf_storage.py.
"""

import functools
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

_REPO_TYPE = "dataset"
_IMAGE_PREFIX = "Image_"
_IMAGE_SUFFIX = ".ome.tiff"
_ANNOTATION_PREFIX = "artifact_annotation_image_"
_ANNOTATION_SUFFIX = ".geojson"


class HFConfigError(Exception):
    """HF_TOKEN/HF_DATASET_REPO missing, or a real HF Hub call failed.

    One error type is enough here (unlike hf_storage.py's ImageNotFoundError/
    HFAuthError split) -- that split exists to map failures to distinct HTTP
    status codes for a deployed service's many callers. This module has one
    caller context: a human running a training script once, who needs one
    clear, actionable message, not a taxonomy.
    """


def _repo_id() -> str:
    repo_id = os.environ.get("HF_DATASET_REPO")
    if not repo_id:
        raise HFConfigError("Set the HF_DATASET_REPO environment variable (e.g. 'Dhruvgoyal18/saterra-project').")
    return repo_id


def _token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise HFConfigError("Set the HF_TOKEN environment variable -- the dataset repo is private.")
    return token


def _client_kwargs() -> dict:
    return {"repo_id": _repo_id(), "repo_type": _REPO_TYPE, "token": _token()}


@functools.lru_cache(maxsize=1)
def list_repo_files() -> tuple[str, ...]:
    """Cached per-process: discover_image_ids() and every download_annotation()
    call in one training run share a single network round trip. Call
    list_repo_files.cache_clear() first if you need to pick up a file
    uploaded to the repo mid-process.
    """
    try:
        files = HfApi(token=_token()).list_repo_files(**_client_kwargs())
    except Exception as exc:
        raise HFConfigError(f"Could not list files in HF dataset repo '{_repo_id()}': {exc}") from exc
    return tuple(files)


def discover_image_ids() -> tuple[list[int], list[int]]:
    """(labeled_ids, unlabeled_ids), determined by listing the live HF repo --
    an image is "labeled" iff a matching artifact_annotation_image_{id}.geojson
    exists in the repo, never by a hardcoded list. New images (and their
    annotations, or lack thereof) are picked up automatically on the next
    call, no code change required.
    """
    files = list_repo_files()
    image_ids = sorted(
        int(f[len(_IMAGE_PREFIX) : -len(_IMAGE_SUFFIX)])
        for f in files
        if f.startswith(_IMAGE_PREFIX) and f.endswith(_IMAGE_SUFFIX)
    )
    annotated_ids = {
        int(f[len(_ANNOTATION_PREFIX) : -len(_ANNOTATION_SUFFIX)])
        for f in files
        if f.startswith(_ANNOTATION_PREFIX) and f.endswith(_ANNOTATION_SUFFIX)
    }
    labeled = [i for i in image_ids if i in annotated_ids]
    unlabeled = [i for i in image_ids if i not in annotated_ids]
    return labeled, unlabeled


def download_image(image_id: int) -> Path:
    """Download (or reuse huggingface_hub's own cached copy of) the original
    .ome.tiff for image_id, returning its local path."""
    try:
        path = hf_hub_download(filename=f"{_IMAGE_PREFIX}{image_id}{_IMAGE_SUFFIX}", **_client_kwargs())
    except Exception as exc:
        raise HFConfigError(f"Could not fetch Image_{image_id}{_IMAGE_SUFFIX} from '{_repo_id()}': {exc}") from exc
    return Path(path)


def download_annotation(image_id: int) -> Path | None:
    """Download (or reuse the cached copy of) image_id's GeoJSON annotation,
    or None if this image has no annotation in the repo -- distinguishing
    "no annotation" (a normal, expected state for unlabeled images) from a
    real fetch failure.
    """
    filename = f"{_ANNOTATION_PREFIX}{image_id}{_ANNOTATION_SUFFIX}"
    if image_id not in discover_image_ids()[0]:
        return None
    try:
        path = hf_hub_download(filename=filename, **_client_kwargs())
    except Exception as exc:
        raise HFConfigError(f"Could not fetch {filename} from '{_repo_id()}': {exc}") from exc
    return Path(path)
