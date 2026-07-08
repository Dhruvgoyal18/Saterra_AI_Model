# Checkpoint 2 -- DAPI Artifact Detector Web App

FastAPI backend + TypeScript/React frontend around the Checkpoint 1
`detect_artifacts` model. Full design writeup, decisions, and the Phase E
scaling discussion: `steps_of_implementation_checkpoint2.md`.

**The private Hugging Face dataset repo (`Dhruvgoyal18/saterra-project`) is
the single source of truth for image data, in every environment including
local development.** There is no local-file mode anymore -- `HF_TOKEN` is
required even to run this on your own machine. See
`steps_of_implementation_checkpoint2.md` for the full reasoning behind that
change.

## Layout

```
backend/            FastAPI service (app/main.py), storage/hf_storage.py (all HF I/O), vendored inference code (app/ml/)
frontend/            Vite + React + TypeScript app
scripts/               upload_to_hf.py (add/replace an original image) + prepare_display_images.py (optional local debug tool only)
```

## Run locally

```bash
# 1. Backend -- set HF_TOKEN + HF_DATASET_REPO in backend/.env first
cd backend
pip install -r requirements.txt
cp .env.example .env   # then edit .env: fill in HF_TOKEN
uvicorn app.main:app --reload --port 8010

# 2. Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open the printed Vite URL (typically http://localhost:5173). The frontend
reads `VITE_API_BASE_URL` from `frontend/.env` (copy `.env.example`) to find
the backend -- defaults to `http://127.0.0.1:8010`.

**No pre-generation step.** The backend fetches originals from HF and
generates+caches display PNGs on demand, the first time each image is
requested -- not from a local `DAPI_images/` folder and not from a
pre-built `manifest.json`. Expect the first request for any given image to
be slower (HF download + PNG generation); repeat requests for the same
image are fast (both `huggingface_hub`'s own cache and this app's
`.cache/display/` are warm).

## Deploy for real

1. **Storage**: already done -- the private dataset repo
   `Dhruvgoyal18/saterra-project` contains all 8 originals
   (`Image_1.ome.tiff` ... `Image_8.ome.tiff`) at its root.
   `scripts/upload_to_hf.py` is kept around only for adding a 9th image or
   replacing an existing one later; it is not part of the required deploy
   flow.
2. **Inference**: push `backend/` to a Hugging Face Space (Docker SDK, see
   `backend/README.md` for the required secrets: `HF_TOKEN`,
   `HF_DATASET_REPO`, `CORS_ORIGINS`).
3. **Frontend**: set `VITE_API_BASE_URL` to the deployed Space's URL, `npm
   run build`, host the static `dist/` anywhere (or just keep running it
   locally -- the spec only requires the model and images to be in the
   cloud, per the checkpoint plan).
