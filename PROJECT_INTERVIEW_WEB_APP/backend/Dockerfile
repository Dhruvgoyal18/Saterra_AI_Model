# Deploy target: Hugging Face Spaces (Docker SDK). HF Spaces containers must
# listen on port 7860 and are given no persistent disk beyond the image layer
# and whatever the app writes at runtime (used here as the huggingface_hub
# download cache, see app/config.py's hf_cache_dir).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV STORAGE_BACKEND=hf
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
