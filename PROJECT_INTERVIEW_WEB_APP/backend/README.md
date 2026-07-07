---
title: DAPI Artifact Detector
emoji: 🔬
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# DAPI Artifact Detector -- inference backend

FastAPI service wrapping the Checkpoint 1 `detect_artifacts` model. See
`../steps_of_implementation_checkpoint2.md` for the full design writeup.

## Required Space secrets (Settings -> Variables and secrets)

| Name | Value |
|---|---|
| `STORAGE_BACKEND` | `hf` |
| `HF_TOKEN` | a token with **read** access to the private dataset repo |
| `HF_DATASET_REPO` | e.g. `your-username/dapi-artifact-images` |
| `CORS_ORIGINS` | the deployed frontend's origin (or `*` for a quick demo) |

Set `HF_TOKEN` as a **secret**, not a plain variable -- Spaces encrypts secrets
and never displays them again after saving, which is the point: the private
bucket's credential lives only on the server, and the browser never sees it.

## Local development

```bash
pip install -r requirements.txt
python ../scripts/prepare_display_images.py   # generates backend/bucket_staging/
uvicorn app.main:app --reload --port 8010
```

Defaults to `STORAGE_BACKEND=local`, which reads display PNGs from
`bucket_staging/` and original TIFFs directly from the sibling
Checkpoint-1 `DAPI_images/` folder -- no HF token needed for local dev.
