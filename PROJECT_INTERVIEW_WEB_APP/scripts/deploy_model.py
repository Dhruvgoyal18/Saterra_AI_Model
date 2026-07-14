"""Deploy freshly-trained Checkpoint 1 weights to the live backend.

Formalizes what has, until now, only ever been done as a hand-typed
HfApi().upload_folder(...) one-liner every time this project's backend was
deployed. Two steps, in order:

  1. Copy artifacts/model_final.pt + artifacts/preprocessing_config.json
     (src/train.py's output -- see src/config.py's MODEL_WEIGHTS_PATH /
     PREPROCESSING_CONFIG_PATH) into backend/app/ml/weights/ -- what the
     deployed backend actually loads (backend/app/config.py's
     model_weights_path / preprocessing_config_path). Not imported from
     src/config.py -- this project's "vendor, don't import across project
     trees" precedent applies here too, so the paths are just replicated as
     plain constants below.
  2. Push the whole backend/ folder to the live HF Space (repo_type="space"),
     the same ignore_patterns already used for every manual deploy this
     project has done (excludes .env, .cache/, a stray local venv, logs,
     __pycache__), then poll until the Space is RUNNING again.

Usage:
  export HF_TOKEN=hf_xxx                            # write access to the Space
  export HF_SPACE_REPO=Dhruvgoyal18/saterra-ml-model
  python scripts/deploy_model.py                     # copy + push + wait for rebuild
  python scripts/deploy_model.py --dry-run             # show what would happen, change nothing
  python scripts/deploy_model.py --skip-wait            # push but don't block on the rebuild finishing
"""

import argparse
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
_WEIGHTS_DIR = _BACKEND_DIR / "app" / "ml" / "weights"

_WEIGHT_FILES = ["model_final.pt", "preprocessing_config.json"]

_IGNORE_PATTERNS = [
    ".env",
    ".env.*",
    ".cache/**",
    "mod/**",
    "*.log",
    "__pycache__/**",
    "**/__pycache__/**",
    ".git/**",
]

_TERMINAL_FAILURE_STAGES = {"RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "DELETED"}


def _copy_weights(dry_run: bool) -> None:
    for name in _WEIGHT_FILES:
        src = _ARTIFACTS_DIR / name
        dst = _WEIGHTS_DIR / name
        if not src.exists():
            raise SystemExit(f"Missing training output: {src} -- run `python -m src.train` first.")
        print(f"{'[dry-run] would copy' if dry_run else 'Copying'} {src} -> {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _push_to_space(token: str | None, repo_id: str | None, dry_run: bool) -> None:
    print(f"{'[dry-run] would upload' if dry_run else 'Uploading'} {_BACKEND_DIR} -> {repo_id} (space)")
    if dry_run:
        return
    api = HfApi(token=token)
    commit = api.upload_folder(
        folder_path=str(_BACKEND_DIR),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=_IGNORE_PATTERNS,
        commit_message="Deploy updated model weights (scripts/deploy_model.py)",
    )
    print(f"Pushed: {commit}")


def _wait_for_running(token: str, repo_id: str, timeout_s: int = 600, poll_s: int = 15) -> None:
    api = HfApi(token=token)
    print("Waiting for the Space to rebuild...")
    start = time.time()
    while time.time() - start < timeout_s:
        stage = api.get_space_runtime(repo_id).stage
        print(f"  stage: {stage}")
        if stage == "RUNNING":
            print("Space is running.")
            return
        if stage in _TERMINAL_FAILURE_STAGES:
            raise SystemExit(f"Space failed to come up: {stage}")
        time.sleep(poll_s)
    raise SystemExit(f"Timed out after {timeout_s}s waiting for the Space to come up.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, change nothing.")
    parser.add_argument("--skip-wait", action="store_true", help="Push but don't block on the rebuild finishing.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_SPACE_REPO")
    if not args.dry_run and (not token or not repo_id):
        raise SystemExit("Set HF_TOKEN and HF_SPACE_REPO environment variables first.")

    _copy_weights(args.dry_run)
    _push_to_space(token, repo_id, args.dry_run)

    if args.dry_run:
        print("Dry run complete -- nothing was changed.")
        return
    if args.skip_wait:
        print("Skipping wait for rebuild (--skip-wait). Check the Space dashboard for status.")
        return
    _wait_for_running(token, repo_id)


if __name__ == "__main__":
    main()
