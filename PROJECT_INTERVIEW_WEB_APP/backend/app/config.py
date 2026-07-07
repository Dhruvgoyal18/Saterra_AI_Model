"""Backend configuration, read from environment variables (.env in dev).

STORAGE_BACKEND switches between two storage implementations without any
other code change:
  - "local": reads display PNGs from a local staging folder and original
    OME-TIFFs directly from the sibling Checkpoint-1 DAPI_images/ folder.
    Used for local development and for the tests run in this session
    (there is no live HF token in this environment).
  - "hf": reads both from a *private* Hugging Face Hub dataset repo via
    huggingface_hub, using HF_TOKEN. This is the production configuration
    the user asked for ("make sure it's private").
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    storage_backend: str = "local"  # "local" | "hf"

    # Local backend
    local_bucket_dir: Path = Path(__file__).resolve().parent.parent / "bucket_staging"
    local_originals_dir: Path = Path(__file__).resolve().parents[3] / "DAPI_images"

    # Hugging Face Hub backend (private dataset repo)
    hf_token: str | None = None
    hf_dataset_repo: str | None = None  # e.g. "your-username/dapi-artifact-images"
    hf_cache_dir: Path = Path(__file__).resolve().parent.parent / ".hf_cache"

    # Model
    model_weights_path: Path = Path(__file__).resolve().parent / "ml" / "weights" / "model_final.pt"
    preprocessing_config_path: Path = Path(__file__).resolve().parent / "ml" / "weights" / "preprocessing_config.json"

    # CORS -- "*" is fine for a demo; restrict to the deployed frontend origin in production.
    cors_origins: str = "*"

    # Inference
    prob_threshold: float = 0.5


settings = Settings()
