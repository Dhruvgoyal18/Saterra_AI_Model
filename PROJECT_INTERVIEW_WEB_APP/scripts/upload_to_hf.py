"""Re-upload a changed or additional original .ome.tiff to the HF dataset
repo.

NOT part of the required deploy flow anymore. The 8 originals
(Image_1.ome.tiff ... Image_8.ome.tiff) are already uploaded to the private
repo, and display PNGs are never uploaded at all -- the backend generates
and caches them locally on demand (see backend/app/storage/hf_storage.py).
This script exists only for the case where you need to add a 9th image or
replace an existing original later.

Usage:
  export HF_TOKEN=hf_xxx                       # a token with write access
  export HF_DATASET_REPO=Dhruvgoyal18/saterra-project
  python scripts/upload_to_hf.py path/to/Image_9.ome.tiff
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/upload_to_hf.py path/to/Image_N.ome.tiff")

    local_path = Path(sys.argv[1])
    if not local_path.exists():
        raise SystemExit(f"No such file: {local_path}")

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_DATASET_REPO")
    if not token or not repo_id:
        raise SystemExit("Set HF_TOKEN and HF_DATASET_REPO environment variables first.")

    api = HfApi(token=token)
    path_in_repo = local_path.name  # e.g. "Image_9.ome.tiff", matching the repo's existing flat layout

    print(f"Uploading {local_path} -> {repo_id}/{path_in_repo} ...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("Done. GET /images on the backend will pick this up on its next call (no restart needed).")


if __name__ == "__main__":
    main()
