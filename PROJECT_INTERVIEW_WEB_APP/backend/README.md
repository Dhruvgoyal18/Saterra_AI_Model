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
`../steps_of_implementation_checkpoint2.md` for the full design writeup,
including why storage is HF-only in every environment (there is no
"local files" mode anymore).

## Required secrets/variables (Space Settings -> Variables and secrets, or `backend/.env` locally)

| Name | Value |
|---|---|
| `HF_TOKEN` | a token with **read** access to the private dataset repo -- required everywhere, including local dev, since the repo is private |
| `HF_DATASET_REPO` | `Dhruvgoyal18/saterra-project` |
| `HF_DISPLAY_CACHE_DIR` | optional, defaults to `.cache/display` (relative to `backend/`) |
| `CORS_ORIGINS` | the deployed frontend's origin (or `*` for a quick demo) |

Set `HF_TOKEN` as a **secret** on a Space, not a plain variable -- Spaces
encrypts secrets and never displays them again after saving, which is the
point: the private dataset's credential lives only on the server, and the
browser never sees it.

The app fails immediately at startup with a clear message if `HF_TOKEN` or
`HF_DATASET_REPO` is missing -- this is expected, not a bug, since there is
no fallback path that doesn't need them.

## Local development

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in HF_TOKEN
uvicorn app.main:app --reload --port 8010
```

That's it -- no separate asset-preparation step. The first request for any
given image (`GET /images/{id}/display` or `POST /detect`) downloads that
image's original from the private HF dataset repo (cached afterward by
`huggingface_hub` itself) and, for the display endpoint, generates and
caches a downsampled PNG under `.cache/display/`. Expect the *first* request
per image to be noticeably slower than the rest.
