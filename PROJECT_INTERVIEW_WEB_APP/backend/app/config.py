"""Backend configuration, read from environment variables (.env in dev).

The Hugging Face dataset repo is the single source of truth for image data
in every environment, including local development -- there is no
"STORAGE_BACKEND=local" switch and no local originals/bucket path anymore
(see steps_of_implementation_checkpoint2.md for why this changed from the
first version of this backend, which did have a local-file mode). Since the
repo is private, HF_TOKEN is required even for local dev.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Hugging Face Hub (private dataset repo) -- required, no default.
    # Left as str | None (rather than a bare `str`) so a missing value fails
    # with the friendly, actionable RuntimeError raised in main.py at
    # startup, instead of a generic pydantic ValidationError.
    hf_token: str | None = None
    hf_dataset_repo: str | None = None

    # Supabase Auth -- gates /images, /images/{id}/display, and /detect (see
    # main.py's require_user dependency). Same "fail fast at startup" pattern
    # as hf_token/hf_dataset_repo above: left as str | None, not a bare str,
    # so a missing value raises the friendly RuntimeError in main.py instead
    # of a generic pydantic ValidationError.
    supabase_url: str | None = None
    supabase_anon_key: str | None = None

    # Groq (OpenAI-compatible chat completions + tool calling) -- backend-only,
    # same reasoning as hf_token: never sent to the frontend. Powers /chat
    # (Checkpoint 3). Same fail-fast pattern as the settings above.
    groq_api_key: str | None = None

    # Derived display PNGs are cached here, one file (+ a small dims sidecar)
    # per image, generated the first time that image's display is requested.
    # NOT the same thing as huggingface_hub's own cache for originals, which
    # is managed entirely by huggingface_hub itself (see storage/hf_storage.py).
    hf_display_cache_dir: Path = Path(__file__).resolve().parent.parent / ".cache" / "display"

    # Generated deep-zoom (DZI) tile pyramid cache: one JPEG per
    # (image_id, level, col, row) plus a small per-image meta sidecar
    # (dims, max_level, normalization stats), generated the first time that
    # image's deep-zoom view is opened. Independent of hf_display_cache_dir
    # above -- deliberately a separate directory, not a shared sidecar
    # format, so this feature's entire on-disk footprint is one new directory
    # (see storage/tile_storage.py).
    hf_tile_cache_dir: Path = Path(__file__).resolve().parent.parent / ".cache" / "tiles"

    # Model
    model_weights_path: Path = Path(__file__).resolve().parent / "ml" / "weights" / "model_final.pt"
    preprocessing_config_path: Path = Path(__file__).resolve().parent / "ml" / "weights" / "preprocessing_config.json"

    # CORS -- "*" is fine for a demo; restrict to the deployed frontend origin in production.
    cors_origins: str = "*"

    # Inference
    prob_threshold: float = 0.5


settings = Settings()
