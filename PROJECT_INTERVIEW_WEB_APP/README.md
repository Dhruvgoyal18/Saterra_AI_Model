# Checkpoint 2 -- DAPI Artifact Detector Web App

FastAPI backend + TypeScript/React frontend around the Checkpoint 1
`detect_artifacts` model. Full design writeup, decisions, and the Phase E
scaling discussion: `steps_of_implementation_checkpoint2.md`.

## Layout

```
backend/            FastAPI service (app/main.py) + vendored inference code (app/ml/)
frontend/            Vite + React + TypeScript app
scripts/               Phase A asset prep (prepare_display_images.py) + HF upload (upload_to_hf.py)
```

## Run locally (both pieces, no cloud/HF token needed)

```bash
# 1. Generate display PNGs + manifest from the Checkpoint 1 DAPI_images/ (once)
../mod/Scripts/python.exe scripts/prepare_display_images.py   # or your own venv with tifffile/pillow/numpy

# 2. Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8010

# 3. Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open the printed Vite URL (typically http://localhost:5173). The frontend
reads `VITE_API_BASE_URL` from `frontend/.env` (copy `.env.example`) to find
the backend -- defaults to `http://127.0.0.1:8010`.

This exact loop (backend on :8010, frontend on :5173) was run and tested end
to end in this build: `/health`, `/images`, `/images/{id}/display`, and
`/detect` all verified via curl, and the full picker -> Detect -> overlay ->
summary flow was driven in an actual browser (see
`steps_of_implementation_checkpoint2.md` for the walkthrough and what it
showed).

## Deploy for real

1. **Storage**: `python scripts/upload_to_hf.py` with `HF_TOKEN` (write
   access) and `HF_DATASET_REPO` set -- creates a **private** HF dataset repo
   and uploads `manifest.json` + each image's `display.png` +
   `original.ome.tiff`. Not run in this build (no network/token in that
   sandboxed environment) but ready to run.
2. **Inference**: push `backend/` to a Hugging Face Space (Docker SDK, see
   `backend/README.md` for the required secrets: `STORAGE_BACKEND=hf`,
   `HF_TOKEN`, `HF_DATASET_REPO`, `CORS_ORIGINS`).
3. **Frontend**: set `VITE_API_BASE_URL` to the deployed Space's URL, `npm
   run build`, host the static `dist/` anywhere (or just keep running it
   locally -- the spec only requires the model and images to be in the
   cloud, per the checkpoint plan).
