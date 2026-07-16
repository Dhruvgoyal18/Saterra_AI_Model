# Understanding Notes

Personal reference notes on how this project actually works, explained at
interview depth. Appended to over time — each entry is a self-contained
topic.

---

## Backend codebase walkthrough: every file, module by module

### Directory shape

```
backend/
  app/
    main.py            FastAPI app: routes, CORS, auth dependency
    config.py           Settings (env vars via pydantic-settings)
    schemas.py           Pydantic request/response models
    detection.py          run_detection(): the one "run the model" entrypoint
    chat.py                /chat LLM tool-calling orchestration (Groq)
    ml/
      model.py             SmallCNN architecture (vendored from src/model.py)
      preprocessing.py       percentile normalization (vendored from src/dataset.py)
      tissue.py               Otsu tissue-mask estimator (vendored from src/masks.py)
      inference.py             detect_artifacts(): tiling + batched inference + summary stats
      tiling.py                 Deep Zoom (DZI) pyramid + tile extraction
      display_prep.py           full-image -> downsampled 8-bit preview PNG
      weights/model_final.pt, preprocessing_config.json
    storage/
      hf_storage.py             the ONLY module that talks to huggingface_hub
      tile_storage.py            DZI tile cache/orchestration, sits on top of hf_storage
  requirements.txt, Dockerfile, .env / .env.example
```

Two package trees exist in this repo: `src/` (Checkpoint 1 training code) and
`PROJECT_INTERVIEW_WEB_APP/backend/app/ml/` (this section). The three files
`model.py`, `preprocessing.py`, `tissue.py` are **vendored copies**, not
imports, of their `src/` counterparts — deliberately duplicated rather than
sharing a package, because this backend is deployed standalone (e.g. as a
Hugging Face Space container) and only ships files inside
`PROJECT_INTERVIEW_WEB_APP/`; it cannot depend on a sibling `src/` package
that won't exist in that container. This is a real interview point: it's a
conscious "duplication over cross-tree dependency" tradeoff, justified by the
deployment boundary, not an oversight.

### The big picture: what problem is this backend solving

It serves a whole-slide DAPI (fluorescence microscopy) artifact detector as a
web app:
- Lists available slide images (`GET /images`)
- Serves a browsable low-res preview of a slide (`GET /images/{id}/display`)
- Serves a real pan/zoom deep-zoom tile pyramid of the full-resolution slide
  (`GET /images/{id}/dzi.dzi` + `GET /images/{id}/dzi_files/{level}/{tile}`)
- Runs a trained CNN over the slide, tile by tile, to flag "artifact" regions
  (`POST /detect`)
- Exposes the same detection capability through a natural-language chat
  interface backed by an LLM with function-calling (`POST /chat`)

All image data (`.ome.tiff` originals) lives in a **private Hugging Face
Hub dataset repo** — there is no local-disk fallback anywhere, in any
environment, including local dev (see `config.py` / `hf_storage.py`
docstrings). This was a deliberate migration away from an earlier
local-file-backed version of the backend (see `2af506d` on this branch,
`training/hf-backed-data`).

### `config.py` — Settings

A `pydantic-settings` `BaseSettings` subclass, reading from `.env`. Key
groups:
- `hf_token` / `hf_dataset_repo` — HF Hub credential + private repo id. Typed
  as `str | None` (not bare `str`) *on purpose*: if pydantic enforced them as
  required, a missing value would raise a generic `ValidationError` at import
  time with a confusing traceback. Instead they're left optional here and
  explicitly checked in `main.py` at startup, raising a friendly
  `RuntimeError` that says exactly what env var to set and where.
- `supabase_url` / `supabase_anon_key` — same fail-fast pattern, for auth.
- `groq_api_key` — same pattern again, checked in `chat.py` at import time.
- `hf_display_cache_dir` (`.cache/display/`) and `hf_tile_cache_dir`
  (`.cache/tiles/`) — two **independent**, purpose-specific on-disk caches
  this codebase owns itself (separate from huggingface_hub's own download
  cache). Kept as separate directories rather than a shared sidecar format so
  the deep-zoom feature's entire on-disk footprint is one new directory,
  isolated from the pre-existing display-PNG cache.
- `model_weights_path` / `preprocessing_config_path` — point at the vendored
  `ml/weights/` files.
- `prob_threshold` (default 0.5) — the sigmoid-output cutoff for calling a
  tile "artifact."

This same fail-fast-with-actionable-message pattern repeats three times in
this codebase (HF, Supabase, Groq) — a consistent house style: validate
external dependencies eagerly at process start, never let a missing
credential surface as a stack trace from deep inside request handling.

### `schemas.py` — the wire contract

Pydantic models FastAPI uses for request validation and response
serialization (and to auto-generate the OpenAPI schema):
- `ImageInfo` / `ImagesResponse` — for `GET /images`. `level0_dims` /
  `display_dims` are `None` until that image's display PNG has been
  generated at least once (listing is cheap — it doesn't force every image
  to be fetched just to report dimensions).
- `DetectRequest` / `DetectResponse` / `DetectSummary` — the `/detect`
  contract: a full confidence grid (`tile_confidence: list[list[float]]`,
  rows x cols) plus a `DetectSummary` (artifact area in px and as a fraction
  of tissue area, connected-region count, largest region size, tiles
  evaluated vs. total).
- `ChatRequest` / `ChatResponse` — `conversation_id` is `None` on a
  conversation's first turn (server mints a UUID and returns it); the client
  echoes it back on later turns to keep using the same in-memory history.
  `ChatResponse.detection_result` is the exact same `DetectResponse` shape,
  non-null only when that turn's tool call actually ran the model — this
  lets the frontend render an LLM-triggered detection through the identical
  `DetectionCanvas`/`SummaryPanel` components the manual "Detect" button uses.

Notable *removed* field: an old `labeled` field (from a prior
manifest.json-based schema, marking which images had Checkpoint-1
ground-truth annotations) was dropped entirely rather than defaulted away,
because that information was never uploaded to the HF repo and isn't
derivable from listing it — keeping it would mean hardcoding
Checkpoint-1-specific knowledge into a pipeline whose whole point is that the
HF repo is the single source of truth.

### `main.py` — the FastAPI app and routes

**Startup-time guards** (before the app object is even fully wired up): if
`HF_TOKEN`/`HF_DATASET_REPO` or `SUPABASE_URL`/`SUPABASE_ANON_KEY` are
missing, raise `RuntimeError` immediately with instructions. This is why the
uvicorn logs (`uvicorn_run*.log`) are worth checking first if the app won't
boot — it's very likely one of these three fail-fast checks (HF, Supabase,
or the Groq one in `chat.py`, since `chat.py` is imported by `main.py`).

**`require_user`** — the actual authorization boundary, used as a FastAPI
`Depends` on every data-bearing route (`/images`, `/images/{id}/display`,
`/images/{id}/dzi.dzi`, `/images/{id}/dzi_files/...`, `/detect`, `/chat`):
- Reads `Authorization: Bearer <token>` from the request header.
- Forwards that token to Supabase's `GET /auth/v1/user` endpoint (with the
  project's anon key as the `apikey` header) to verify the session is live.
- 401 if the header's missing/malformed, 401 if Supabase says the token's
  invalid/expired, **503** (not 500) if Supabase itself is unreachable
  (`httpx.TransportError`) — this distinguishes "your session is bad" from
  "the auth provider is having a moment," which used to surface as an
  unhandled 500 with a raw traceback before this was added.
- Explicitly documented as the *real* access boundary, not the frontend's
  login screen — anyone can curl this API directly, bypassing the SPA
  entirely, so this dependency is what actually enforces invite-only access.

**Routes:**
| Route | Purpose | Key errors mapped |
|---|---|---|
| `GET /health` | liveness + which HF repo is configured | — |
| `GET /images` | list image ids + cached dims (if known) | `HFAuthError` → 401 |
| `GET /images/{id}/display` | serve cached/generated preview PNG | `HFAuthError`→401, `ImageNotFoundError`→404 |
| `GET /images/{id}/dzi.dzi` | DZI XML descriptor for OpenSeadragon | same |
| `GET /images/{id}/dzi_files/{level}/{tile}` | one deep-zoom JPEG tile | same + `InvalidTileError`→404 |
| `POST /detect` | run the model, return grid + summary | same |
| `POST /chat` | LLM-driven natural-language detection trigger | only the LLM call itself → 502; tool-level errors are absorbed into the conversation (see chat.py) |

Two other things worth knowing:
- The **DZI tile filename convention** (`"{col}_{row}.jpeg"`) is parsed by
  hand in the route (`tile_name.removesuffix(".jpeg").split("_")`) because
  it isn't expressible as a FastAPI path-converter pattern.
- The backend **proxies the display PNG through itself** rather than having
  the browser fetch it straight from the HF bucket — because the bucket is
  private, so a direct browser fetch would 401/403. `HF_TOKEN` never leaves
  the server; the browser only ever talks to *this* FastAPI service.

### `detection.py` — the one canonical "run the model" path

`run_detection(image_id) -> DetectResponse` is deliberately the *only*
implementation of "fetch an image and run the CNN over it." Both `POST
/detect` and `chat.py`'s `run_artifact_detection` tool call this same
function directly (in-process, not over HTTP) — so there's no risk of the
chat path and the button path drifting into two different behaviors.

What it does, in order:
1. `storage.get_original_path(image_id)` — download/reuse-cached `.ome.tiff`.
2. Read the array with `tifffile`: `tif.series[0].levels[0].asarray()` (note
   `.levels[0]` — see tiling.py notes below on why these files have no
   native pyramid).
3. `estimate_tissue_mask(image)` — cheap Otsu-based tissue/background split.
4. `detect_artifacts(...)` — the actual CNN inference (see `ml/inference.py`).
5. Repack the numpy results into the `DetectResponse` pydantic shape
   (`tile_confidence.tolist()` for JSON-serializability, `grid_shape` from
   the confidence array's own `.shape`).

`TILE_SIZE` and `BACKGROUND_THRESHOLD` are read once at import time from
`preprocessing_config.json` (`{"tile_size": 64, "background_threshold": 0.5,
...}`) — this file is the single source of truth for preprocessing constants
shared between training (Checkpoint 1) and this inference service, so tile
size can't silently drift between train and serve.

### `ml/model.py` — the network

`SmallCNN`: 3 conv blocks (`Conv2d → BatchNorm2d → ReLU`, channels 1→16→32→64,
each followed by `MaxPool2d(2)` except the last which uses
`AdaptiveAvgPool2d(1)` instead), then a single `Linear(64, 1)` head. Input is
single-channel (`in_channels=1`, DAPI is a single fluorescence channel — no
RGB). Output is a single raw logit per tile (squeezed to shape `(batch,)`),
turned into a probability via `sigmoid` in `inference.py`, not inside the
model itself — keeps the model a pure logit producer, sigmoid applied once at
the call site.

### `ml/preprocessing.py` — normalization

Per-image 1st/99th-percentile normalization (`compute_percentile_norm_stats`
+ `normalize_tile`): clip each tile's raw uint16 intensities to `[lo, hi]`
computed from the *whole image's* percentiles, then rescale to `[0, 1]`
float32. Using whole-image percentiles (not per-tile) means every tile from
the same image is normalized consistently — a tile isn't independently
contrast-stretched relative to just its own local pixels. This exact logic
is reused in training, in `ml/tiling.py`'s DZI generation, and in
`ml/display_prep.py`'s preview generation — one normalization definition,
three consumers.

### `ml/tissue.py` — tissue/background mask

Approximates "is this tissue or background" without any ground truth, via:
1. Downsample by averaging non-overlapping `64x64` blocks (`_block_mean`).
2. Otsu-threshold (`_otsu_threshold_1d`, a from-scratch histogram-variance
   implementation, not cv2/skimage) on `log1p` of those block means — log
   because raw DAPI background/tissue intensity distributions are
   heavy-tailed and Otsu's between-class variance criterion works better in
   log space here.
3. Upsample the resulting block-level boolean mask back to full resolution
   by repeating each block's decision across all its pixels.

Applied uniformly to **all 8 images**, including the 5 that actually have
Checkpoint-1 ground-truth tissue polygons — because those GeoJSON
annotations were never uploaded to the HF repo (only the raw TIFFs were), so
this service has no ground truth available for *any* image at inference
time. Validated in Checkpoint 1 at IoU 0.92–0.96 against real ground truth
for images 1–5 — "good enough to gate obvious background out of inference,"
explicitly not a claim of pixel-perfect segmentation.

### `ml/inference.py` — `detect_artifacts`, the actual tiled-CNN pass

This is the core algorithm. Given a full-resolution single-channel image:
1. Compute global percentile norm stats once (`compute_percentile_norm_stats`).
2. Walk the image in a `tile_size` (64px) grid, `n_rows = height // tile_size`,
   `n_cols = width // tile_size` (a **truncating** grid — any partial tile at
   the bottom/right edge that doesn't fill a full 64px is simply not
   evaluated at all, not padded).
3. For each grid cell, if a `tissue_mask` was passed, skip it entirely
   (`continue`, never added to any batch) when that tile's tissue-mask mean
   coverage is below `background_threshold` (0.5) — background tiles never
   reach the model.
4. Otherwise normalize the tile and accumulate it into a batch; once the
   batch reaches `batch_size` (512), run one forward pass
   (`flush_batch`) — batching tiles rather than one-at-a-time inference calls
   is the main perf lever here (single `torch.stack` + one forward pass per
   512 tiles instead of thousands of individual calls).
5. Any tile scoring `prob >= prob_threshold` (0.5 default) gets its full
   64x64 pixel region marked `True` in a full-resolution boolean `mask`.
6. `mask &= tissue_mask` — a belt-and-suspenders re-intersection with the
   tissue mask (tiles were already gated at the tile-selection step, so this
   line is mostly a no-op safety net, not the primary filter).
7. `scipy.ndimage.label` finds connected components in `mask` →
   `num_connected_regions`, and their sizes → `largest_region_area_px`.
8. Returns `{"mask", "per_tile_confidence", "summary"}` — the route layer
   converts the numpy arrays to JSON-friendly lists; this function itself
   stays numpy-native.

`load_model` caches the loaded `SmallCNN` in a module-level dict keyed by
weights path (`_MODEL_CACHE`) — the model is loaded from disk and
`.eval()`'d once per process, not once per request.

Everything runs on CPU (`device="cpu"` default) — consistent with the plain
`python:3.11-slim` Dockerfile (no CUDA, no GPU driver layer).

### `ml/tiling.py` — Deep Zoom (DZI) pyramid generation

The `.ome.tiff` source files have **no native multi-resolution pyramid**
(`len(tif.series[0].levels) == 1`, confirmed by hand for all 8 files) — so
this module builds one server-side purely from the single full-res level,
entirely with libraries already in `requirements.txt` (numpy, cv2, Pillow).
Deliberately does **not** use `openslide-python`: it would only be
downsampling from that same single base level anyway (no free pyramid to
exploit), and it would require adding a system package
(`libopenslide0`) to what's currently a plain slim Docker image with zero
apt-get steps.

Key mechanics:
- **DZI level numbering** is the *inverse* of "how downsampled" — level 0 is
  a 1x1 image, `max_level` is full native resolution
  (`dzi_max_level = ceil(log2(max(w, h)))`). This is the opposite of the
  intuitive "level 0 = full res" and is called out explicitly in the
  docstring as a common source of off-by-one/direction bugs at the seam
  between this module and the route/descriptor code.
- `dzi_level_dims` uses the exact formula OpenSeadragon itself uses
  client-side (`ceil(dim / 2^(max_level - level))`) — not a naive recursive
  halving — because a halving recurrence would drift from OpenSeadragon's
  own expectation over several levels on odd dimensions, silently breaking
  tile/row/col counts at an edge.
- `build_pyramid` builds every level by resizing from the nearest
  **already-computed larger level** (not always re-resizing from the
  original full-res array) — cheap, and the whole pyramid's total memory is
  only ~1.33x the full-res array (geometric series of quarter-area steps).
  Every level stays in native `uint16` — no percentile clip/rescale happens
  in this function at all.
- `extract_tile` is where the actual normalization happens, and — this is
  the important bit — it slices out **one tile's worth of pixels first**,
  and only *then* upcasts that small slice to float64 for the
  clip/percentile-rescale math. This is a direct fix for a real production
  bug: `display_prep.py`'s original approach of clipping the *entire*
  full-resolution array against float percentile bounds allocated a ~637MB
  intermediate for a single 9000x9000 image, which OOM'd on low-memory
  hosts. Never clip/rescale a whole large array — always downsample or slice
  down to the working size first, upcast last.
- Normalization stats (`lo`/`hi`) are computed **once per image**, not once
  per tile, and reused for every tile at every level — otherwise adjacent
  tiles could render at visibly different brightness, breaking the illusion
  of a single continuous image as a viewer pans across a tile boundary.

### `ml/display_prep.py` — the flat preview PNG

`to_display_png`: percentile-normalize (same 1/99 percentiles as
everywhere else) and downsample a full-res array to a preview PNG whose
longer side is `TARGET_LONG_SIDE = 2000`px. Order of operations matters here
too, and matches the OOM lesson above: **downsample first** (`cv2.resize` on
the raw uint16 array, since `cv2.resize` handles uint16 correctly and
PIL's 16-bit support doesn't), **then** clip/rescale the now-small array to
uint8. This module's docstring explicitly says this fix's history is *why*
`tiling.py`'s `extract_tile` was designed the way it was — same bug class,
same fix pattern, applied at two different points in the codebase.

### `storage/hf_storage.py` — the only HF Hub client

Single rule enforced by convention (called out in the docstring): **nothing
else in the codebase imports `huggingface_hub` directly** — everything else
calls through this module's functions.

- `list_images()` — lists repo files via `HfApi.list_repo_files`, filters to
  `*.ome.tiff`, strips the suffix. Always a live repo listing, never a local
  manifest — a 9th image added to the repo just shows up, no code change.
- `validate_image_id()` — checks a given id against `list_images()` *before*
  it's used to build any HF filename/path. This exists specifically because
  Checkpoint 3's chat LLM can freely invent an `image_id` from natural
  language; a `<select>` dropdown in the old UI could never supply a bogus
  id, so this wasn't a real risk before. It also fixes a genuine
  pre-existing ambiguity: without this check, "file doesn't exist in an
  otherwise-accessible repo" and "no access to the repo at all" both landed
  in the same `status in (401, 403, 404)` branch in `get_original_path`,
  raising a misleading "check your HF_TOKEN" error for a simple typo'd image
  id. Checking the allowlist first makes the typo case short-circuit into a
  clean `ImageNotFoundError` before reaching that ambiguous branch.
- `get_original_path()` — `hf_hub_download(...)` with no custom `cache_dir`,
  deliberately: huggingface_hub already dedupes "don't re-download a file we
  have," so a second cache layer on top would just be redundant bookkeeping.
- `get_display_image_path()` — get-or-create for the preview PNG:
  cache hit = zero HF traffic, straight return; cache miss = fetch original,
  build via `ml/display_prep.to_display_png`, write PNG + a `.dims.json`
  sidecar (`level0_dims`, `display_dims`) next to it.
- `get_cached_dims()` — **read-only**, never triggers a fetch or generation;
  returns `None` if the sidecar doesn't exist yet. This is what lets
  `GET /images` stay cheap regardless of how many of the 8 images have
  actually been opened/generated so far.

Two *separate* caches are in play and this module only owns one: HF Hub's
own `~/.cache/huggingface/hub` (or `HF_HOME`) for original files (managed
entirely by huggingface_hub itself), and `settings.hf_display_cache_dir` for
derived display PNGs (owned/managed by this module).

### `storage/tile_storage.py` — DZI tile cache, layered on top

A deliberate **sibling** module to `hf_storage.py`, not a merge into it — so
the pre-existing display/detect paths have zero regression risk from adding
deep-zoom. It never touches `huggingface_hub` directly; it calls
`hf_storage.get_original_path`, preserving the "only one HF client" rule.

Two caches, both explicitly treated as ephemeral (lost on container restart
— accepted, same tradeoff as the display-PNG cache, since HF Spaces
containers have no persistent disk):
1. **On-disk**: one JPEG per `(image_id, level, col, row)` under
   `settings.hf_tile_cache_dir`, plus one `{image_id}.meta.json` sidecar
   (`width, height, max_level, lo, hi`) per image — this is the *only* place
   percentile stats get computed for the tile pyramid; every tile request
   after the first reads cached `lo`/`hi`, never recomputes.
2. **In-memory LRU** (`_pyramid_cache`, capped at
   `_PYRAMID_CACHE_MAX_IMAGES = 2`): the full decoded `dict[level ->
   ndarray]` pyramid for whichever image(s) were most recently viewed, so a
   burst of tile requests while panning/zooming one open image doesn't
   re-decode and re-resize the source TIFF per tile. Guarded by a
   `threading.Lock` because FastAPI runs sync routes in a thread pool — real
   concurrent OS threads can hit this, and "check cache, maybe evict, maybe
   insert" is not atomic across separate dict operations without the lock.

`get_or_build_tile()` order of operations matters: it validates
`level`/`col`/`row` against the image's real meta dimensions **before**
touching the pyramid cache at all — an out-of-range level would otherwise
raise a raw `KeyError` from inside the pyramid dict instead of a clean
`InvalidTileError` → 404.

### `chat.py` — Checkpoint 3's LLM function-calling

`POST /chat` lets a user trigger the *exact same* `detection.run_detection`
path via natural language, through Groq's OpenAI-compatible chat-completions
API (model: `llama-3.3-70b-versatile`) with function calling. No second
detection pipeline and no HTTP round-trip to the backend's own `/detect`
route — the tool executor calls `run_detection` in-process, same as the
route handler does.

Two tools are exposed to the model:
- `run_artifact_detection(image_id)` — runs the model, returns to the LLM
  *only the summary* (not the full `tile_confidence` grid — the LLM only
  needs the summary to describe results in words; sending the whole grid
  would just burn tokens). The frontend still gets the full `DetectResponse`
  separately, via the second element `_execute_tool_call` returns.
- `list_images()` — lets the model check which image ids actually exist
  instead of guessing/inventing one from user phrasing like "the second
  slide."

Conversation state (`_conversations: dict[str, list[dict]]`) is **in-memory
only**, keyed by `conversation_id`, no database — an explicit, accepted
tradeoff for a demo (same "no persistence beyond what's needed" judgment
already applied to Supabase auth): a restart wiping history is fine because
the point of this checkpoint is proving the tool-calling path works, not
building durable chat.

The system prompt is deliberately narrow and explicit about what the
assistant *cannot* do (delete/modify/upload images, change how results are
displayed, answer off-topic questions) and instructs it to ask a clarifying
question rather than guess an image id or fabricate numbers — and to answer
from an *earlier* turn's tool result if the same image was already detected
in this conversation, rather than re-running the model.

One notable piece of defensive engineering: `_looks_like_malformed_tool_call`
detects a real, observed failure mode of Groq's Llama-3.3-70B — it
occasionally hallucinates a fake function-call as plain text (e.g.
`<function.run_artifact_detection {"image_id": "Image_7"}></function>`)
instead of using the actual tool-calling protocol, specifically after a
prior successful tool call earlier in the same conversation (~1-in-4
reproduction rate against the deployed `list_images` →
`run_artifact_detection` sequence in testing). This was tried and
*rejected* as a schema fix (giving `list_images` a dummy non-empty parameter
made it worse — it started surfacing as an outright 400
`tool_use_failed`). Since it's inherent model flakiness rather than a bug in
this code's use of the API, the mitigation is a **one-time retry of the same
request** — reduces but does not guarantee eliminating the failure (a retry
can, in principle, fail the same way again).

Errors from tool execution (`storage.ImageNotFoundError`, `HFAuthError`)
are deliberately *not* raised as HTTP errors from `/chat` — `_execute_tool_call`
catches them and turns them into a `{"error": ...}` tool-result message fed
back to the LLM, which then phrases a natural-language decline itself. Only
a genuine failure of the Groq API call itself surfaces as an HTTP error
(502) from the route — the reasoning being that a bad tool argument is a
conversational turn, not an HTTP-layer failure.

### Cross-cutting patterns worth naming in an interview

- **Fail fast at import/startup, not at first request** — repeated 3x (HF,
  Supabase, Groq) with the same actionable-`RuntimeError` shape.
- **One canonical implementation per capability, multiple callers** — `/detect`
  and chat's tool both call `detection.run_detection`; `tile_storage.py` and
  `hf_storage.py` both build on the same `get_original_path`; `display_prep.py`,
  `preprocessing.py`, and `tiling.py` all share one normalization definition.
- **Downsample/slice before upcast-and-clip, never after** — the specific
  fix for a real OOM bug, applied independently in both `display_prep.py`
  and `tiling.py.extract_tile`.
- **Distinct, purpose-typed exceptions per failure mode** (`HFAuthError` vs.
  `ImageNotFoundError` vs. `InvalidTileError`) mapped to distinct HTTP status
  codes at the route layer, rather than one generic exception type.
- **Everything ephemeral is explicitly labeled as such** — both on-disk
  caches (display PNGs, DZI tiles) and the in-memory chat history are called
  out in docstrings as acceptable-to-lose-on-restart, matching the
  no-persistent-disk reality of the HF Spaces deployment target.

---

## Detection request lifecycle: `POST /detect`, browser click to overlay on screen

### Architecture in one line

A React/TypeScript SPA (Vercel) talks to a FastAPI backend (Hugging Face
Space, Docker) over a REST API secured by Supabase JWTs; the backend is
stateless compute in front of two external systems -- Hugging Face Hub
(private dataset storage) and a PyTorch model loaded in-process -- with no
database anywhere in the hot path.

### Request lifecycle

**1. Client-side auth attachment.** Before any data request, the frontend
calls `supabase.auth.getSession()` -- not a cached token, a fresh read each
time, since Supabase access tokens are short-lived (1hr) and the client SDK
handles silent refresh under the hood. That JWT goes into the
`Authorization: Bearer <token>` header. `POST /detect` ships `{image_id}` as
the body, wrapped in a 120s client-side timeout via `AbortController` (the
backend is a free-tier host that sleeps when idle, so a cold start is
expected latency, not a failure).

**2. Auth boundary at the API layer.** FastAPI resolves `require_user` as a
route dependency *before* the handler body executes -- this is the actual
access-control boundary, not the frontend's login screen, since anyone can
hit the API directly with curl and bypass the UI entirely. `require_user`
does a server-to-server call: `GET {SUPABASE_URL}/auth/v1/user` with the
bearer token forwarded plus the project's anon key, via a shared
`httpx.AsyncClient` (explicit 10s timeout -- this was a real bug hit and
fixed, since an unbounded/default-timeout call turned a transient Supabase
slowdown into an unhandled 500 instead of a clean, retryable error).
Non-200 -> 401. Transport failure -> 503.

**3. Input validation before touching external state.** The route delegates
to a shared `run_detection(image_id)` function (deliberately factored out so
`/detect` and the chat tool-call path share one implementation, not two).
First thing it does: `validate_image_id()` against `list_images()`, which
calls `HfApi.list_repo_files()` -- a metadata-only call against the Hub API
listing the private dataset repo. This is deliberate defense-in-depth: once
an LLM can generate `image_id` from freeform text (Checkpoint 3), you can't
trust it's one of the 8 real IDs without checking first, and this also
short-circuits a real pre-existing bug where a missing-file 404 and a
repo-access 404 were indistinguishable.

**4. Fetching the source data.** `hf_hub_download(filename=..., repo_id=...,
repo_type="dataset", token=HF_TOKEN)`. `huggingface_hub`'s own local cache
(content-addressed by revision/ETag) means a cache hit only costs a `HEAD`
request to confirm the ETag hasn't moved; a miss streams the actual
`.ome.tiff` (100-200MB) from Hub storage, authenticated since the repo is
private. Notably we don't build a second cache on top of this -- the
library's own cache already solves it correctly.

**5. Decode + preprocessing.** `tifffile.TiffFile(path).series[0].levels[0].asarray()`
reads pyramid level 0 explicitly (never the library's default
series-selection, which was validated against ground truth early in the
project to rule out a wrong-pyramid-level bug). This decodes to a
full-resolution `uint16` numpy array, roughly 9000x9000. `estimate_tissue_mask()`
then does a block-mean + log-transform + Otsu threshold to approximate
tissue-vs-background -- an engineering workaround for the fact this deployed
service has no ground-truth tissue polygon for any image (those GeoJSON
annotations were never uploaded to the Hub repo).

**6. Inference.** `detect_artifacts()`:
- `load_model()` checks a module-level `dict` cache keyed by weights path --
  the model loads from `model_final.pt` once per process, not once per
  request, which matters most right after a cold start where every other
  cost is already unavoidable.
- Percentile normalization stats (1st/99th) are computed once per image, not
  per tile -- per-image because absolute intensity drifts across scan
  batches, so a fixed normalization constant would bake in one batch's
  brightness.
- The image is walked as a grid of 64x64 tiles; any tile whose tissue-mask
  coverage is below 0.5 is skipped entirely before it ever reaches the model
  -- this is the main compute-saving lever, since background is most of the
  image.
- Surviving tiles are batched (up to 512) and forward-passed through a small
  CNN (3x conv-batchnorm-relu-maxpool + global average pool + linear head),
  `torch.sigmoid` producing a per-tile probability in one shot rather than
  per-tile inference calls.
- Tiles above the probability threshold get painted into a full-resolution
  boolean mask, which is then intersected with the tissue mask;
  `scipy.ndimage.label` does connected-component analysis for region count
  and largest-region size.

**7. Response shape.** The handler returns a `DetectResponse` Pydantic model
-- `grid_shape`, the full `tile_confidence` grid (serialized to nested
lists), a `summary` dict, `level0_dims`, `tile_size`. Deliberately the
*grid*, not the full-resolution boolean mask -- a 144x144 grid of floats is a
few hundred KB of JSON; the equivalent full-res mask would be tens of
megapixels. That's the actual design lever that keeps this viable over plain
HTTP instead of needing a streaming/binary protocol.

**8. Client-side rendering.** The frontend paints the already-independently-
fetched display PNG (a separate `GET /images/{id}/display` call, proxied
through the backend rather than fetched directly from the Hub, since the
bucket is private and a browser can't attach the download token itself) and
the confidence grid into *one* `<canvas>` -- image and overlay baked into a
single bitmap rather than an `<img>` plus an absolutely-positioned second
`<canvas>`, which removes an entire class of CSS-drift bugs where the two
layers could resize independently and misalign. The mapping from tile-grid
coordinates to display-pixel coordinates is a straight linear scale
(`display_dims / level0_dims`), exact because the display PNG is a pure
uniform downsample with no other resize step in between.

### Where `/chat` diverges

Only in how the request gets initiated. `POST /chat` goes through the same
`require_user` gate, then hands the conversation to Groq's OpenAI-compatible
chat-completions API with one tool definition (`run_artifact_detection`). If
the model emits a tool call, the executor calls the *same* `run_detection()`
function in-process -- no HTTP round-trip to `/detect`, no second copy of the
detection logic -- gets back a summary, and a follow-up completion call
turns that into natural language. The full `DetectResponse` still goes to
the frontend alongside the reply, rendered through the identical
canvas/summary components. The interesting engineering point here isn't the
LLM call itself, it's that the tool boundary was designed to reuse the
existing verified code path rather than becoming a second, divergent
implementation.

---

## Checkpoint 2 cloud-inference split: deploying the model on Modal

This section supersedes the in-process description above (`detect_artifacts`
running inside the same process as the web backend) with what's actually
deployed in production today. The web-serving/auth/storage-proxy layers
described above are all unchanged; only *where the model runs* moved.

### Why this exists at all

Checkpoint 2's rubric line is specific: *"Model inference hosted in the
cloud and image data stored in a cloud bucket; the local app reads images
from the bucket and calls the hosted inference endpoint."* Before this work,
image storage genuinely was a separate cloud bucket (the private HF dataset
repo, described above) -- but model inference was **not** separately
hosted. `detect_artifacts` ran `torch.load`/`model.eval()`/`model(batch)` in
the same FastAPI process that served the web API. That's "the whole app is
in the cloud," which is a different (weaker) claim than "the model is a
separately-callable hosted endpoint," which is what the rubric asks for.
The fix required standing up a second, independent service and having the
backend call it over HTTP -- mirroring how the bucket call already worked.

### The road here: three platforms, two failed outright

**Attempt 1 -- Hugging Face Spaces (`Dhruvgoyal18` account).** Built
`inference_service/` as a second FastAPI app (same `SmallCNN`,
`detect_artifacts`, `estimate_tissue_mask`, `preprocessing` code, vendored a
third time following this project's own "vendor across deploy boundaries"
precedent), Dockerized it, pushed it as a new private Space
(`saterra-model-inference`) via `HfApi().upload_folder(repo_type="space")`.
The code was fully correct -- verified locally to reproduce the exact
Checkpoint-1 baseline numbers for Image_6 (28.31% artifact frac, 1399
regions, 6,250,496px largest region at `prob_threshold=0.5`). The Space
itself never came up: it stayed `PAUSED` after the push, and calling
`HfApi().restart_space(...)` returned a bare `401 Unauthorized` --
the available token (an OAuth-derived `hf_oauth_...` token, not a
classic Access-Token-Settings-page token) could push repo content and set
Space secrets, but apparently lacked the separate permission needed to
manage a Space's *compute/runtime* state via the API.

**Attempt 2 -- Hugging Face Spaces, brand-new account (`mrcurious18`).**
Same code, pushed to a fresh account with a token generated the normal way
(Settings -> Access Tokens -> Write). This time `restart_space` didn't 401 --
progress -- but returned `403 Forbidden: You've reached your cpu-basic
quota limit`, on an account with **zero** other Spaces. Email verification
didn't clear it; neither did waiting. Concluded this is a platform-level
anti-abuse throttle on brand-new accounts that doesn't lift on any timeline
controllable from here. Abandoned per explicit instruction ("if something
breaks, be ready to reverse it back") -- all code changes reverted via `git
checkout`, new files deleted, nothing was ever committed.

**Render -- considered, not attempted.** This project's own deployment
history (see the Render section further down / `steps_of_implementation_checkpoint2.md`
S12.2) already answered this one: Render's free tier (~512MB RAM) OOM-killed
the *full* backend decoding these same ~9000x9000 TIFFs, even after a real
memory-usage fix landed. `inference_service` is leaner (no
`huggingface_hub`/`opencv`/`Pillow`/`groq` in that version's deps), so it
might have fared better, but Render also requires a git-based deploy (push
to a connected repo) rather than a direct file-upload API, which didn't fit
the "don't commit anything for this experiment" constraint in play at the
time. Never actually tried; documented here as "ruled out, not tested,"
which is an honest, different claim than "tried and failed."

**Attempt 3 -- Modal. This one worked.** `modal deploy` builds and starts a
function in one atomic call -- there's no separate "please wake up my
already-pushed app" step that can lack permission, which is exactly the
class of problem that killed both HF attempts. Deployment is CLI-driven
from a local Python file, not git-based, satisfying the "no commits" rule
Render couldn't. The user already had a Modal account with free credits.

### Modal-specific problems hit, and how each was actually solved

These were all real, not hypothetical -- each one blocked forward progress
until diagnosed:

1. **Windows console encoding crash.** `modal serve`/`deploy`'s Rich-based
   CLI output tries to print a checkmark (`✓`); Windows' default
   codepage (cp1252) can't encode it when stdout is redirected to a file
   (as it was here, running headless through a tool harness). Fixed by
   forcing `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` in the environment before
   invoking `modal`.

2. **`modal serve modal_app.py` couldn't find the app object.** Modal's CLI
   defaults to looking for a module-level variable literally named `app`.
   The Modal `App` object here was deliberately named `modal_app` (to avoid
   visual confusion with the *FastAPI* `app` object imported inside the
   same function), so the CLI needed the explicit target:
   `modal serve modal_app.py::modal_app`.

3. **The big one: Modal's hard 150-second per-request timeout.** The
   original design had the backend download the ~85MB DAPI TIFF from the HF
   bucket and re-upload those same bytes as the `/predict` request body
   (multipart file upload) to the Modal-hosted model. Modal enforces a hard
   150s timeout on any single HTTP request to a hosted function
   (`modal.asgi_app`/`fastapi_endpoint`/etc. all share this limit) --
   confirmed against Modal's own docs, not just inferred from symptoms.
   Measured directly: even a **10MB** synthetic upload didn't complete in
   90-120 seconds, and this wasn't a client artifact -- reproduced
   identically with `curl` (`Expect: 100-continue` disabled) and with a
   from-scratch Python `httpx.post(..., content=data)` call, which hit a
   `WriteTimeout`. Bisected with 1MB (30s), 10MB (>90s, no response) --
   clearly not linear with plain network bandwidth, and independent of
   whether the body was sent as multipart or a raw `application/octet-stream`
   body (`request.body()` directly). Root cause: Modal's web-endpoint
   ingress path just isn't built for large direct request-body uploads --
   it's meant for typical small JSON-style request/response traffic, per
   Modal's own recommended pattern for oversized payloads ("don't upload
   it directly -- have the function fetch/reference the data, or use an
   async job + polling pattern").
   **Fix:** flipped who initiates the image fetch. Instead of
   backend-fetches-then-forwards, the Modal-hosted `/predict` endpoint now
   fetches the image *itself*, directly from the same private HF dataset
   repo, via its own `hf_hub_download` call (needs its own `HF_TOKEN`/
   `HF_DATASET_REPO`, provisioned as a second Modal secret,
   `hf-bucket-secret`). The backend's request shrank from "85MB multipart
   upload" to `POST /predict?image_id=Image_6&prob_threshold=0.7` -- a
   few dozen bytes, regardless of how large the actual image is. This
   incidentally reads as an even more literal match for the Checkpoint 2
   wording: a genuinely separate, hosted service that reads from the bucket
   on its own.

### Final architecture: two Modal secrets, one function

`inference_service/modal_app.py` defines one Modal `App` (`modal_app`,
name `"saterra-model-inference"`) with one function
(`fastapi_app`), decorated `@modal_app.function(secrets=[...]) @modal.asgi_app()`,
that returns the existing `inference_service/app/main.py` FastAPI object
unmodified in shape (same route names, same response schema) -- only its
*internals* changed (fetch-by-id instead of receive-upload). The image is
built via `modal.Image.debian_slim().pip_install(...)` +
`.add_local_dir(app/, "/root/app", copy=True)`, which bakes the ~100KB model
weights and all vendored code directly into the image at build time (no
`modal.Volume` needed at this size). Two secrets:
- `inference-secret` -> `INFERENCE_TOKEN`: a generated shared bearer secret
  checked by `require_token()` in `inference_service/app/main.py` against
  every `Authorization: Bearer` header. This is a second, app-level auth
  layer independent of Modal's own platform auth.
- `hf-bucket-secret` -> `HF_TOKEN` + `HF_DATASET_REPO`: the same dataset-repo
  credentials the backend already holds, now also given to the Modal
  service so it can fetch images on its own behalf.

Deployed, permanent URL:
`https://dhruvgoyal990--saterra-model-inference-fastapi-app.modal.run`.

### Full request-flow map, end to end

```
Browser (React/Vite frontend)
   |
   | 1. GET /images/{id}/display  (Authorization: Bearer <Supabase JWT>)
   v
Backend (FastAPI, deployed as Dhruvgoyal18/saterra-ml-model HF Space,
         or localhost:8010 in local dev)
   |  - require_user(): validates the Supabase JWT against
   |    GET {SUPABASE_URL}/auth/v1/user
   |  - storage.get_display_image_path(): fetches/caches the original
   |    .ome.tiff from the private HF dataset repo (Dhruvgoyal18/
   |    saterra-project), downsamples it server-side, returns a PNG
   v
Browser renders the preview image on a <canvas>

Browser
   |
   | 2. POST /detect  {"image_id": "Image_6"}
   |    (Authorization: Bearer <Supabase JWT>)
   v
Backend  (app/detection.py: run_detection())
   |  - require_user() gate (same as above)
   |  - storage.validate_image_id(image_id): confirms it's a real file in
   |    the HF dataset repo listing (cheap 404 short-circuit for bad ids --
   |    preserves chat.py's graceful-decline behavior for LLM-invented ids)
   |  - httpx.post(f"{INFERENCE_ENDPOINT_URL}/predict",
   |               params={"image_id": ..., "prob_threshold": ...},
   |               headers={"Authorization": f"Bearer {INFERENCE_SERVICE_TOKEN}"})
   v
Modal-hosted inference service (fastapi_app, saterra-model-inference)
   |  - require_token(): checks the Authorization header against the
   |    INFERENCE_TOKEN Modal secret (independent of Supabase auth --
   |    this is service-to-service, the browser never sees this token)
   |  - hf_hub_download(filename=f"{image_id}.ome.tiff",
   |                     repo_id=HF_DATASET_REPO, token=HF_TOKEN)
   |    -- fetches the real ~85MB original directly from the HF bucket,
   |       using its OWN hf-bucket-secret credentials
   |  - tifffile.TiffFile(...).asarray() -- decodes the full-res image
   |  - estimate_tissue_mask(image) -- Otsu-on-log-block-mean, gates
   |    background tiles out of inference
   |  - detect_artifacts(image, weights_path, tissue_mask, ...):
   |      - loads SmallCNN + model_final.pt (baked into the Modal image)
   |      - tiles the image into 64x64 tiles, normalizes each via
   |        1st/99th-percentile stats computed on the full image
   |      - batches tiles through the model, torch.sigmoid() per tile
   |      - builds a boolean artifact mask, intersects with tissue mask
   |      - scipy.ndimage.label() for connected-component region stats
   |  - returns JSON: {grid_shape, tile_confidence (nested list),
   |                    summary {total_artifact_area_px,
   |                             total_artifact_area_frac_of_tissue,
   |                             num_connected_regions,
   |                             largest_region_area_px,
   |                             n_tiles_evaluated, n_tiles_total},
   |                    level0_dims, tile_size}
   v
Backend
   |  - wraps the JSON into DetectResponse(image_id=..., **payload)
   |    (image_id is the only field the backend adds back -- everything
   |    else is exactly what the Modal service computed)
   v
Browser
   |  - DetectionCanvas.tsx paints the per-tile confidence grid as a
   |    color-mapped overlay on top of the already-rendered display PNG
   |  - SummaryPanel.tsx renders the summary block directly
```

For the `/chat` path, step 2 above is identical except the *trigger* is an
LLM tool call (`run_artifact_detection`) instead of a button click --
`chat.py`'s tool executor calls the exact same `run_detection()` function,
so everything from `storage.validate_image_id` downward is byte-for-byte
the same code path, same Modal round-trip, same response shape.

### Verified, concretely, at every layer

- Modal `/health` (no auth): `200 {"status": "ok"}`.
- Modal `/predict` with a wrong/missing bearer token: `401` (fast, ~200ms --
  proves the token check runs before any image fetch is attempted).
- Modal `/predict?image_id=Image_6&prob_threshold=0.5` with the correct
  token: exact baseline match, 28.31% frac / 1399 regions / 6,250,496px
  largest region, ~29-31s per call (mix of cold-start + ~85MB HF download +
  actual tiled inference over ~19,317 candidate tiles -- warm calls weren't
  meaningfully faster, so the download+compute cost dominates over any
  container cold-start cost here).
- Backend `run_detection()` called in-process locally against the live
  Modal URL: same numbers, now at the backend's actual configured
  `prob_threshold=0.7` (12.51% frac / 917 regions / 802,816px largest).
- Backend `/detect` over real HTTP with a real Supabase JWT, both locally
  (`localhost:8010`) and against the redeployed **production** HF Space
  (`Dhruvgoyal18/saterra-ml-model`): identical numbers.
- Backend `/chat` with `"Please run artifact detection on Image_6..."`:
  the LLM's natural-language reply correctly cites 12.51%/917 regions,
  confirming the tool-calling path also round-trips through Modal correctly.

### What's still in place, deliberately, for now

The backend's old in-process `ml/{model,inference,tissue,preprocessing}.py`
+ `ml/weights/` files are still present and untouched -- kept as a rollback
safety net, not yet deleted, per the same "be ready to reverse it back"
instruction that governed the two abandoned HF attempts. Removing them
(plus dropping `torch`/`scipy` from the backend's own `requirements.txt`,
since only the Modal service needs them now) is the one remaining cleanup
step, deferred until the Modal-backed path has been live long enough to
trust fully.
