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
| `SUPABASE_URL` | your Supabase project URL -- gates every `/images`, `/display`, `/detect`, `/chat` request (invite-only auth, see `steps_of_implementation_checkpoint2.md` SS11) |
| `SUPABASE_ANON_KEY` | Supabase project's anon/public key -- safe to expose (also goes in `frontend/.env` as `VITE_SUPABASE_ANON_KEY`), used here to verify session tokens against `/auth/v1/user` |
| `GROQ_API_KEY` | Checkpoint 3: powers `POST /chat`'s function-calling. `console.groq.com` -> API Keys. Backend-only, same reasoning as `HF_TOKEN` -- never sent to the frontend |

Set `HF_TOKEN` and `GROQ_API_KEY` as **secrets** on a Space, not plain
variables -- Spaces encrypts secrets and never displays them again after
saving, which is the point: real credentials live only on the server, and
the browser never sees them. `SUPABASE_URL`/`SUPABASE_ANON_KEY` are fine as
plain variables since the anon key is meant to be public by design.

The app fails immediately at startup with a clear message if any of
`HF_TOKEN`/`HF_DATASET_REPO`, `SUPABASE_URL`/`SUPABASE_ANON_KEY`, or
`GROQ_API_KEY` is missing -- this is expected, not a bug, since there is no
fallback path that doesn't need them.

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
