"""Storage abstraction: where display PNGs and original OME-TIFFs come from.

Two backends behind one interface (`Storage`), selected by
`settings.storage_backend`. Everything else in the app (routes, inference)
only ever calls these three methods and never knows which backend is active.
"""

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile

from .config import settings


class ImageNotFoundError(Exception):
    pass


class Storage(ABC):
    @abstractmethod
    def get_manifest(self) -> dict:
        """Returns {"images": {image_id: {"level0_dims": [W, H], "display_dims": [w, h], "labeled": bool}}}."""

    @abstractmethod
    def get_display_png_bytes(self, image_id: str) -> bytes:
        """Raw PNG bytes for the downsampled display image."""

    @abstractmethod
    def get_original_array(self, image_id: str) -> np.ndarray:
        """Full-resolution (level 0) single-channel DAPI array for inference."""


def _image_id_to_index(image_id: str) -> int:
    # "image_7" -> 7
    try:
        return int(image_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError) as exc:
        raise ImageNotFoundError(f"Malformed image_id: {image_id!r}") from exc


class LocalStorage(Storage):
    """Reads display PNGs from bucket_staging/ (see scripts/prepare_display_images.py)
    and original OME-TIFFs directly from the sibling Checkpoint-1 DAPI_images/ folder,
    so nothing needs to be duplicated on disk for local development.
    """

    def __init__(self, bucket_dir: Path, originals_dir: Path):
        self.bucket_dir = bucket_dir
        self.originals_dir = originals_dir

    def get_manifest(self) -> dict:
        manifest_path = self.bucket_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest at {manifest_path}. Run scripts/prepare_display_images.py first."
            )
        with open(manifest_path) as f:
            return json.load(f)

    def get_display_png_bytes(self, image_id: str) -> bytes:
        png_path = self.bucket_dir / image_id / "display.png"
        if not png_path.exists():
            raise ImageNotFoundError(f"No display PNG for {image_id} at {png_path}")
        return png_path.read_bytes()

    def get_original_array(self, image_id: str) -> np.ndarray:
        idx = _image_id_to_index(image_id)
        tiff_path = self.originals_dir / f"Image_{idx}.ome.tiff"
        if not tiff_path.exists():
            raise ImageNotFoundError(f"No original TIFF for {image_id} at {tiff_path}")
        with tifffile.TiffFile(tiff_path) as tif:
            return tif.series[0].levels[0].asarray()


class HFStorage(Storage):
    """Reads everything from a *private* Hugging Face Hub dataset repo.

    Requires HF_TOKEN with read access to HF_DATASET_REPO. Files are cached
    locally by huggingface_hub after first download (hf_cache_dir), so a
    second /detect call for the same image doesn't re-download the original.
    """

    def __init__(self, repo_id: str, token: str, cache_dir: Path):
        self.repo_id = repo_id
        self.token = token
        self.cache_dir = cache_dir
        self._manifest: dict | None = None

    def _download(self, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=filename,
            token=self.token,
            cache_dir=str(self.cache_dir),
        )
        return Path(path)

    def get_manifest(self) -> dict:
        if self._manifest is None:
            path = self._download("manifest.json")
            with open(path) as f:
                self._manifest = json.load(f)
        return self._manifest

    def get_display_png_bytes(self, image_id: str) -> bytes:
        try:
            path = self._download(f"{image_id}/display.png")
        except Exception as exc:  # huggingface_hub raises its own HfHubHTTPError etc.
            raise ImageNotFoundError(f"No display PNG for {image_id} in {self.repo_id}") from exc
        return path.read_bytes()

    def get_original_array(self, image_id: str) -> np.ndarray:
        try:
            path = self._download(f"{image_id}/original.ome.tiff")
        except Exception as exc:
            raise ImageNotFoundError(f"No original TIFF for {image_id} in {self.repo_id}") from exc
        with tifffile.TiffFile(path) as tif:
            return tif.series[0].levels[0].asarray()


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    if settings.storage_backend == "hf":
        if not settings.hf_token or not settings.hf_dataset_repo:
            raise RuntimeError(
                "STORAGE_BACKEND=hf requires HF_TOKEN and HF_DATASET_REPO to be set."
            )
        return HFStorage(settings.hf_dataset_repo, settings.hf_token, settings.hf_cache_dir)
    return LocalStorage(settings.local_bucket_dir, settings.local_originals_dir)
