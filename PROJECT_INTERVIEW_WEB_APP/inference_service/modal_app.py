"""Modal deployment for the inference service -- image_id in, predictions
out, hosted on Modal instead of a Docker container on Hugging Face
Spaces/Render (see ../../steps_of_implementation_checkpoint2.md for why
those two were abandoned: HF's token lacked Space-restart permission, then a
fresh account hit a persistent quota wall; Render's free tier previously
OOM'd on the same large TIFFs this service decodes).

This service fetches its own images from the private HF dataset repo
(app/main.py) rather than receiving them as an upload -- Modal enforces a
hard 150s timeout per HTTP request, and uploading the ~85MB DAPI images
directly as a request body blew well past that window (measured).

Usage:
  modal secret create inference-secret INFERENCE_TOKEN=<a-shared-secret>
  modal secret create hf-bucket-secret HF_TOKEN=<hf-token> HF_DATASET_REPO=<owner>/<repo>
  modal serve modal_app.py     # ephemeral URL, for testing
  modal deploy modal_app.py    # permanent URL
"""

from pathlib import Path

import modal

_APP_DIR = Path(__file__).resolve().parent / "app"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "numpy",
        "scipy",
        "tifffile",
        "imagecodecs",
        "torch",
        "huggingface_hub",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_dir(str(_APP_DIR), remote_path="/root/app", copy=True)
)

modal_app = modal.App("saterra-model-inference", image=image)


@modal_app.function(
    secrets=[modal.Secret.from_name("inference-secret"), modal.Secret.from_name("hf-bucket-secret")]
)
@modal.asgi_app()
def fastapi_app():
    from app.main import app as fastapi_asgi_app

    return fastapi_asgi_app
