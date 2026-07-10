# Pipeline update — source everything from Hugging Face, remove local-file dependency

## Why this change

The current codebase treats the local `DAPI_images/` folder as the source of
truth: `scripts/prepare_display_images.py` reads originals from disk and
writes display PNGs + `manifest.json` to disk, the backend serves those local
files, and `scripts/upload_to_hf.py` is a separate, optional, one-way push to
Hugging Face that happens *after* local prep is done.

That's backwards for where this project is going. The Hugging Face dataset
repo (`Dhruvgoyal18/saterra-project`, private, containing `Image_1.ome.tiff`
… `Image_8.ome.tiff` at the repo root) is now **the single source of truth**
for image data — including for local development. Nothing in the app,
including the version running on `localhost`, should read an original image
from the local `DAPI_images/` folder ever again. Display PNGs stop being a
pre-generated, pre-uploaded asset and become a **derived, on-demand,
locally-cached artifact** — generated the first time an image is requested
from HF, and reused from a cache folder after that.

This document is the spec for that refactor. Implement it as described;
don't reintroduce a "read from local DAPI_images" code path anywhere,
including as a fallback.

## Target architecture

```
                     ┌─────────────────────────────┐
                     │  Hugging Face dataset repo   │
                     │  Dhruvgoyal18/saterra-project│
                     │  (private) — Image_1..8      │
                     │  .ome.tiff at repo root       │
                     └───────────┬─────────────────┘
                                 │ hf_hub_download (auth: HF_TOKEN)
                                 ▼
                     ┌─────────────────────────────┐
                     │   backend/app/storage/       │
                     │   hf_storage.py               │
                     │  - list available images      │
                     │  - fetch original (cached by   │
                     │    huggingface_hub's own       │
                     │    ~/.cache/huggingface/hub)   │
                     │  - get-or-create display PNG   │
                     │    (own cache folder)          │
                     └───────┬─────────────┬─────────┘
                             │             │
                 originals   │             │  display PNGs
              (full res,     │             │  (derived, cached in
               read for      │             │   backend/.cache/display/)
               inference)    ▼             ▼
                     detect_artifacts   /images/{id}/display
                     (existing CP1 code, unchanged)
```

Two different local caches, both gitignored, both disposable:

- **HF Hub's own cache** (`~/.cache/huggingface/hub` by default, or wherever
  `HF_HOME` points) — holds the downloaded original `.ome.tiff` files.
  `hf_hub_download` manages this automatically; don't build a custom cache
  for originals, just call it correctly and let it do its job.
- **A new local folder for generated display images** —
  `backend/.cache/display/{image_id}.png`. This is new code you're adding.
  It does not exist until the first time that image's display endpoint is
  hit. If the file is already there, serve it. If not, download the
  original via `hf_storage`, generate the PNG the same way
  `prepare_display_images.py` already does (percentile normalize, downsample
  longest side to ~1500–2500px), write it to this folder, then serve it.

## What to change, file by file

### 1. New module: `backend/app/storage/hf_storage.py`

Create this as the single place that talks to Hugging Face. Nothing else in
the codebase should import `huggingface_hub` directly — everything else goes
through this module.

Responsibilities:

- `list_images() -> list[str]` — return the available image ids (e.g.
  `["Image_1", "Image_2", ..., "Image_8"]`) by listing files in the HF repo
  via `HfApi().list_repo_files(repo_id=..., repo_type="dataset")` and
  filtering for `.ome.tiff`, rather than reading a static `manifest.json`
  that was generated once locally. This makes the image list always reflect
  what's actually in the HF repo — add a 9th image to the repo and it shows
  up with no local file to regenerate.
- `get_original_path(image_id: str) -> Path` — call `hf_hub_download(
  repo_id=HF_DATASET_REPO, filename=f"{image_id}.ome.tiff",
  repo_type="dataset", token=HF_TOKEN)` and return the local cached path it
  gives back. This is what `detect_artifacts` should be pointed at for
  inference — it already handles caching, so repeated calls for the same
  image don't re-download.
- `get_display_image_path(image_id: str) -> Path` — the get-or-create logic
  described above: check `backend/.cache/display/{image_id}.png` first;
  if missing, call `get_original_path`, read it with `tifffile`, run the
  existing percentile-normalize + downsample logic (reuse the function
  from `prepare_display_images.py` verbatim — don't reimplement it, move it
  into a shared module both this file and the old script can import), save
  the PNG to the cache path, and return it.

Read `HF_DATASET_REPO` and `HF_TOKEN` from environment variables (see config
section below). Since the repo is **private**, `HF_TOKEN` is required even
for local development now — there is no unauthenticated path anymore.

### 2. `backend/app/main.py` — update the three endpoints

- `GET /images` — call `hf_storage.list_images()` instead of reading
  `manifest.json` from disk. Delete the manifest-reading code path entirely.
- `GET /images/{id}/display` — call `hf_storage.get_display_image_path(id)`
  and serve that file. Remove any code that looks for a pre-existing PNG in
  a folder populated by the old script.
- `POST /detect` — call `hf_storage.get_original_path(image_id)` to get the
  local path to hand to `detect_artifacts`, instead of resolving a path
  under `DAPI_images/` or a local originals folder.

No response shape changes — the JSON contracts for `/images`, `/display`,
and `/detect` stay exactly as they are. This is purely a change in *where
the bytes come from*, not what the API returns.

### 3. `scripts/prepare_display_images.py` — retire as a standalone step

This script's *logic* (percentile normalize, downsample, save PNG) needs to
survive — move it into a shared module, e.g. `backend/app/ml/display_prep.py`,
that both `hf_storage.py` and this script can import. But running this
script manually against a local `DAPI_images/` folder is no longer part of
the workflow, and the README should stop instructing anyone to run it as
"step 1" of local dev. If you want to keep the file around for one-off local
debugging (e.g. visually checking normalization on a file someone has
locally), that's fine, but it must not be a required step, and the app must
work correctly with that folder absent entirely.

### 4. `scripts/upload_to_hf.py` — repurpose, don't delete

The originals are already uploaded (done manually via `hf upload` from the
CLI, per the earlier setup). Keep this script only for the case of
re-uploading a changed or additional original image to the HF repo — not
for display PNGs, since those are never uploaded anymore, only generated
locally and cached. Update its docstring/README description accordingly so
it's not described as part of the required deploy flow for images.

### 5. Config / environment variables

Add to `backend/.env.example` (and real `.env`, gitignored):

```
HF_TOKEN=hf_xxx              # required even for local dev — repo is private
HF_DATASET_REPO=Dhruvgoyal18/saterra-project
HF_DISPLAY_CACHE_DIR=.cache/display   # relative to backend/, gitignored
```

Update `backend/.gitignore` to include `.cache/`.

Remove any existing env var or config value that points at a local
`DAPI_images` directory or a local originals directory — that path should no
longer exist anywhere in config, code, or defaults.

### 6. README updates

Update the "Run locally" section to remove step 1 (`prepare_display_images.py`
as a prerequisite). The new local dev flow is:

```bash
# 1. Backend — set HF_TOKEN + HF_DATASET_REPO in backend/.env, then:
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8010

# 2. Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

First request for any image will be slower (downloads the original from HF
+ generates its display PNG); subsequent requests for the same image are
fast (both caches hit). Call this out explicitly in the README so it isn't
mistaken for a bug during testing.

Also update the "Deploy for real" section: storage is no longer a separate
manual step before deploy — the backend always talks to HF, in every
environment. The only thing that changes between local and deployed is
whether `HF_TOKEN` comes from a local `.env` or from the hosting platform's
secrets — the code path is identical in both places. Remove the framing that
implies local dev and "real" deploy have different storage behavior.

## Acceptance criteria — verify all of these before considering this done

1. Delete (or rename) the local `DAPI_images/` folder entirely, restart the
   backend from a clean checkout with only `HF_TOKEN` and `HF_DATASET_REPO`
   set, and confirm `/images`, `/images/{id}/display`, and `/detect` all work
   correctly for every image with zero local originals present beforehand.
2. Confirm `backend/.cache/display/` is empty on a clean checkout, and gets
   populated one file at a time as each image's display endpoint is hit for
   the first time — not all at once, not via a script.
3. Confirm hitting the same image's display endpoint twice does not
   re-download the original or regenerate the PNG the second time (check
   HF Hub's cache directory and the new display cache directory — timestamps
   shouldn't change on the second call).
4. Confirm the app fails with a clear, actionable error (not a stack trace
   about a missing local folder) if `HF_TOKEN` is missing or invalid —
   since the repo is private, an auth failure here is expected user error,
   not a bug, and should be reported as such.
5. Grep the codebase for any remaining reference to `DAPI_images`,
   `manifest.json`, or a hardcoded local originals path, and remove them.
