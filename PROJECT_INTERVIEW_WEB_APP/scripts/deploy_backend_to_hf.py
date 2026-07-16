"""Sync backend/.env secrets/variables to the HF Space, then push backend/
code and wait for it to rebuild -- formalizes the hand-typed HfApi() session
used for every manual backend redeploy so far (most recently: adding
INFERENCE_ENDPOINT_URL/INFERENCE_SERVICE_TOKEN for the Modal inference
split, then pushing the updated chat_storage.py/detection.py).

Unlike scripts/deploy_model.py (which assumes the Space's secrets/variables
are already correct and only copies fresh model weights + pushes code), this
script's job is keeping the Space's secrets/variables in sync with
backend/.env, so a new key added locally doesn't also have to be typed into
the Space's web UI by hand. Run this one when backend/.env gained a new key
or backend/ code changed; run deploy_model.py when only the trained weights
changed.

Reads backend/.env directly (via python-dotenv, already a transitive dep of
backend/requirements.txt's pydantic-settings) rather than the shell
environment, since that .env file is this project's one documented source of
truth for local secrets (see backend/README.md).

Secrets vs. variables classification matches backend/README.md's table:
HF_TOKEN, GROQ_API_KEY, INFERENCE_SERVICE_TOKEN are secrets (encrypted,
never re-displayed after saving); HF_DATASET_REPO, SUPABASE_URL,
SUPABASE_ANON_KEY, CORS_ORIGINS, INFERENCE_ENDPOINT_URL are variables
(plain, visible) -- the anon key is safe to expose per that table's own
reasoning. HF_DISPLAY_CACHE_DIR/HF_TILE_CACHE_DIR are deliberately not
synced -- both are optional, local-cache-path-only settings with fine
defaults in backend/app/config.py.

Note HF_TOKEN plays two different roles here, not one: the deploy token
(below, "write access to the Space") authenticates *this script's* API
calls, while backend/.env's HF_TOKEN value (a *separate* token, "read
access to the dataset repo" per backend/README.md) is just payload -- the
value that gets written into the Space as a secret for the running app to
use later. They may or may not be the same physical token depending on your
HF account's setup; don't assume they're interchangeable.

Usage:
  export HF_TOKEN=hf_xxx                       # write access to the Space (auth for this script's API calls)
  export HF_SPACE_REPO=Dhruvgoyal18/saterra-ml-model
  python scripts/deploy_backend_to_hf.py                  # sync secrets/variables + push code + wait for rebuild
  python scripts/deploy_backend_to_hf.py --skip-secrets      # push code only, don't touch secrets/variables
  python scripts/deploy_backend_to_hf.py --skip-wait           # push but don't block on the rebuild finishing
  python scripts/deploy_backend_to_hf.py --dry-run               # show what would happen, change nothing
"""

import argparse
import os
import time
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
_ENV_FILE = _BACKEND_DIR / ".env"

_SECRET_KEYS = ["HF_TOKEN", "GROQ_API_KEY", "INFERENCE_SERVICE_TOKEN"]
_VARIABLE_KEYS = ["HF_DATASET_REPO", "SUPABASE_URL", "SUPABASE_ANON_KEY", "CORS_ORIGINS", "INFERENCE_ENDPOINT_URL"]

_IGNORE_PATTERNS = [
    ".env",
    ".env.*",
    ".cache/**",
    "mod/**",
    "*.log",
    "sql/**",
    "__pycache__/**",
    "**/__pycache__/**",
    ".git/**",
]

_TERMINAL_FAILURE_STAGES = {"RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "DELETED"}


def _sync_secrets_and_variables(token: str, repo_id: str, dry_run: bool) -> None:
    if not _ENV_FILE.exists():
        raise SystemExit(f"Missing {_ENV_FILE} -- copy backend/.env.example to backend/.env and fill it in first.")
    env = dotenv_values(_ENV_FILE)
    api = HfApi(token=token)

    for key in _SECRET_KEYS:
        value = env.get(key)
        if not value:
            print(f"  [skip] {key} not set in backend/.env")
            continue
        print(f"{'[dry-run] would set' if dry_run else 'Setting'} secret {key}")
        if not dry_run:
            api.add_space_secret(repo_id=repo_id, key=key, value=value)

    for key in _VARIABLE_KEYS:
        value = env.get(key)
        if not value:
            print(f"  [skip] {key} not set in backend/.env")
            continue
        print(f"{'[dry-run] would set' if dry_run else 'Setting'} variable {key} = {value}")
        if not dry_run:
            api.add_space_variable(repo_id=repo_id, key=key, value=value)


def _push_to_space(token: str, repo_id: str, dry_run: bool) -> None:
    print(f"{'[dry-run] would upload' if dry_run else 'Uploading'} {_BACKEND_DIR} -> {repo_id} (space)")
    if dry_run:
        return
    api = HfApi(token=token)
    commit = api.upload_folder(
        folder_path=str(_BACKEND_DIR),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=_IGNORE_PATTERNS,
        commit_message="Deploy updated backend code (scripts/deploy_backend_to_hf.py)",
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
    parser.add_argument("--skip-secrets", action="store_true", help="Push code only, don't touch secrets/variables.")
    parser.add_argument("--skip-wait", action="store_true", help="Push but don't block on the rebuild finishing.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, change nothing.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_SPACE_REPO")
    if not args.dry_run and (not token or not repo_id):
        raise SystemExit("Set HF_TOKEN and HF_SPACE_REPO environment variables first.")

    if not args.skip_secrets:
        _sync_secrets_and_variables(token, repo_id, args.dry_run)
    else:
        print("Skipping secrets/variables sync (--skip-secrets).")

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
