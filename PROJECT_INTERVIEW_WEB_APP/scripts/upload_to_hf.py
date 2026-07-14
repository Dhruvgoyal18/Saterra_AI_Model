"""Upload one or more files to the HF dataset repo -- originals
(Image_N.ome.tiff) or GeoJSON annotations (artifact_annotation_image_N.geojson).

NOT part of the required deploy flow for the web app anymore (the 8
originals are already there, and display PNGs are never uploaded at all --
the backend generates and caches them locally on demand, see
backend/app/storage/hf_storage.py). This script exists for:
  - adding a 9th+ image or replacing an existing original
  - uploading a GeoJSON annotation so src/'s training pipeline
    (src/hf_data.py) can discover the image as labeled -- an image counts
    as labeled iff a matching artifact_annotation_image_{id}.geojson exists
    in the repo, see src/hf_data.py::discover_image_ids()

Every file is uploaded to the repo root using its own local filename,
matching the repo's existing flat layout (no image_N/ subfolders) -- so the
filename itself is what determines both the path in the repo and, for
annotations, which image the training pipeline associates it with. Only
.ome.tiff and .geojson files are accepted, to avoid an accidental wrong-file
upload to a shared repo.

Usage:
  export HF_TOKEN=hf_xxx                       # a token with write access
  export HF_DATASET_REPO=Dhruvgoyal18/saterra-project
  python scripts/upload_to_hf.py path/to/Image_9.ome.tiff
  python scripts/upload_to_hf.py path/to/artifact_annotation_image_2.geojson
  python scripts/upload_to_hf.py path/to/file1.geojson path/to/file2.geojson  # multiple at once
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

_ALLOWED_SUFFIXES = (".ome.tiff", ".geojson")


def _validate(local_path: Path) -> None:
    if not local_path.exists():
        raise SystemExit(f"No such file: {local_path}")
    if not any(local_path.name.endswith(suffix) for suffix in _ALLOWED_SUFFIXES):
        raise SystemExit(
            f"Refusing to upload '{local_path.name}' -- only {_ALLOWED_SUFFIXES} files are accepted "
            "(a whole-slide original or a GeoJSON annotation), to avoid an accidental wrong-file "
            "upload to a shared repo."
        )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} path/to/file1 [path/to/file2 ...]")

    local_paths = [Path(p) for p in sys.argv[1:]]
    for local_path in local_paths:
        _validate(local_path)

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_DATASET_REPO")
    if not token or not repo_id:
        raise SystemExit("Set HF_TOKEN and HF_DATASET_REPO environment variables first.")

    api = HfApi(token=token)

    for local_path in local_paths:
        path_in_repo = local_path.name
        print(f"Uploading {local_path} -> {repo_id}/{path_in_repo} ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(
        f"Done ({len(local_paths)} file(s)). GET /images on the backend and "
        "src.hf_data.discover_image_ids() will both pick this up on their next call -- no restart needed."
    )


if __name__ == "__main__":
    main()
