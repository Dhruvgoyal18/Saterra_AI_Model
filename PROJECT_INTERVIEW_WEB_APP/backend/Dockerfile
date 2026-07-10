# Deploy target: Hugging Face Spaces (Docker SDK). HF Spaces containers must
# listen on port 7860 and are given no persistent disk beyond the image layer
# and whatever the app writes at runtime -- used here for two on-disk caches
# that are both fine to lose on a restart: huggingface_hub's own download
# cache for original .ome.tiff files (~/.cache/huggingface/hub, unmanaged by
# this app) and this app's own derived-display-PNG cache
# (HF_DISPLAY_CACHE_DIR, see app/config.py).
#
# HF_TOKEN and HF_DATASET_REPO are NOT set here -- they must be provided as
# Space secrets/variables (Settings -> Variables and secrets), since the
# dataset repo is private and the token must never be baked into the image.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
