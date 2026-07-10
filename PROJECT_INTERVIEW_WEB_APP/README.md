# Checkpoint 2/3 -- DAPI Artifact Detector Web App + Chat Interface

FastAPI backend + TypeScript/React frontend around the Checkpoint 1
`detect_artifacts` model, plus (Checkpoint 3) a chat panel that triggers the
same detection via natural language. Full design writeup, decisions, and the
Phase E scaling discussion: `steps_of_implementation_checkpoint2.md`.

**The private Hugging Face dataset repo (`Dhruvgoyal18/saterra-project`) is
the single source of truth for image data, in every environment including
local development.** There is no local-file mode anymore -- `HF_TOKEN` is
required even to run this on your own machine. See
`steps_of_implementation_checkpoint2.md` for the full reasoning behind that
change.

## Layout

```
backend/            FastAPI service (app/main.py), storage/hf_storage.py (all HF I/O), vendored inference code (app/ml/)
                     detection.py (shared run_detection -- used by both /detect and /chat's tool call)
                     chat.py (Checkpoint 3: Groq function-calling over the same detection.run_detection)
frontend/            Vite + React + TypeScript app (ChatPanel.tsx alongside the picker/canvas/summary)
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

**Currently deployed:** backend on a Hugging Face Space
(`Dhruvgoyal18/saterra-ml-model`, public, Supabase-gated), frontend on Vercel
(`https://frontend-opal-iota-55.vercel.app`).

---

## Checkpoint 3 -- Chat interface

A chat panel that triggers the exact same `detect_artifacts` path the
"Detect artifacts" button already uses, via natural language instead of only
a click.

### Architecture

One new endpoint, `POST /chat`, on the same FastAPI app, gated by the same
`require_user` Supabase dependency every other route already uses -- no
second auth model, no second server. `/chat` holds an LLM function-calling
loop server-side; its one tool, `run_artifact_detection(image_id)`, calls
`backend/app/detection.py`'s `run_detection()` **in-process** -- the exact
same function `POST /detect` calls, not a second implementation and not an
HTTP round-trip to `/detect`. See `backend/app/chat.py`.

### Provider: Groq

Chosen over Gemini for this checkpoint because its API is OpenAI-compatible,
so the tool-calling wire format needed essentially no translation, and its
free tier is fast enough that a chat turn that triggers a real detection
(TIFF download + ~20k-tile CPU inference) doesn't add noticeable LLM-side
latency on top of that. Model: `llama-3.3-70b-versatile`.

### Tool schema

```json
{
  "type": "function",
  "function": {
    "name": "run_artifact_detection",
    "description": "Run the trained artifact-detection model on a specific DAPI image and return tile-level predictions plus a summary (artifact area %, region count, largest region size). Only call this for one of the known image IDs; if the user doesn't specify one, or asks for something this tool can't do (e.g. deleting or modifying an image), ask a clarifying question or explain the limitation instead of guessing.",
    "parameters": {
      "type": "object",
      "properties": {
        "image_id": { "type": "string", "description": "e.g. 'Image_6'" }
      },
      "required": ["image_id"]
    }
  }
}
```

The system prompt scopes the assistant strictly to this dataset and this one
tool, and explicitly instructs it to decline or ask for clarification rather
than guess -- see `SYSTEM_PROMPT` in `chat.py`.

### Prerequisite hardened first: `image_id` allowlist validation

Checkpoint 2's own review had already flagged (SS13, finding #4) that
`/detect` and `/images/{id}/display` didn't validate `image_id` against the
real image list before using it to construct an HF filename -- a low-priority
gap when only a `<select>` dropdown could supply it, not once an LLM can
freely invent one from natural language. `storage/hf_storage.py`'s new
`validate_image_id()` checks against `list_images()` before any HF fetch is
attempted, for both `/detect`/`/display` and the chat tool call alike. This
also fixed a real pre-existing bug found in the process: a 404 for "this
filename doesn't exist in an otherwise-accessible repo" was previously
indistinguishable from "no access to this repo at all," both raising the
same misleading "check your `HF_TOKEN`" error for what should have been a
plain "no such image."

### Demo transcripts (real, captured against the running backend -- not hypothetical)

**Valid -- triggers the tool, matches Checkpoint 1's saved numbers exactly:**
```
User: Run artifact detection on image_7
Assistant: For Image_7, 39.38% of the tissue was flagged as artifact. The
           detection found 708 connected artifact regions, with the largest
           region being 12,132,352 pixels in size.
```
(`detection_result.summary`: `total_artifact_area_frac_of_tissue=0.3938`,
`num_connected_regions=708`, `largest_region_area_px=12132352` -- exactly
Checkpoint 1's saved Image_7 result. Note the model normalized the user's
lowercase "image_7" to the dataset's actual "Image_7" on its own.)

**Nice-to-have -- answers from the same conversation's cached result, confirmed via backend logs to make zero extra HF/model calls:**
```
User: How much of image_7 is artifact?
Assistant: For Image_7, 39.38% of the tissue was flagged as artifact.
```

**Declined -- nonexistent image, caught by the allowlist check:**
```
User: Run detection on image_99
Assistant: The image "Image_99" is not a known image. The known images are:
           Image_1, Image_2, Image_3, Image_4, Image_5, Image_6, Image_7,
           Image_8. Which of these images would you like to run detection on?
```

**Declined -- out-of-scope action, model recognizes this without even calling the tool:**
```
User: Please delete image_6 from the dataset
Assistant: I'm not able to delete or modify images. I can only run the
           trained artifact-detection model on a specific DAPI image and
           return tile-level predictions plus a summary. If you would like
           to run the artifact detection model on an image, please let me
           know the image ID and I can assist you with that.
```

All four turns were verified against backend request logs, not just the
chat replies: the valid turn shows a real Groq call, HF file listing, TIFF
`HEAD` request, and a `detect_artifacts` log line; the cached-answer turn
shows only one Groq call and zero HF traffic; the nonexistent-image decline
shows a Groq call *plus* an HF file-listing call (the allowlist check
firing) before the decline; the out-of-scope decline shows only one Groq
call, confirming the tool was never invoked at all.

Also re-verified live against the deployed HF Space (not just locally): "Run
artifact detection on Image_8" -> 41.14% artifact frac, 338 regions,
18,636,800px largest region -- exactly Checkpoint 1's saved Image_8 result.
