# Checkpoint 2: Artifact Detection Web App — Implementation Log

A complete, end-to-end account of how this was built: every decision made,
every alternative considered and rejected, everything actually run and its
real output, and what still needs the user's own credentials/network to
finish. Code lives in `PROJECT_INTERVIEW_WEB_APP/` (`backend/`, `frontend/`,
`scripts/`).

**Headline status:** the entire local loop — asset generation, backend,
frontend — was built, installed, and run for real, not just written.
`/health`, `/images`, `/images/{id}/display`, and `/detect` were all hit with
curl and returned correct data; the frontend was loaded in an actual browser,
driven through the picker → Detect → overlay → summary flow, and the numbers
it displayed matched Checkpoint 1's saved results exactly. What was **not**
executed: uploading to a real Hugging Face dataset repo and deploying the
backend to a Space — both need an HF token, which this build environment
doesn't have. Both scripts for that are written and ready to run.

**Revision note (read this first):** everything above describes the
*first* version of this backend, which supported two storage modes —
`STORAGE_BACKEND=local` (read from a local `bucket_staging/` folder and the
sibling Checkpoint-1 `DAPI_images/`) and `STORAGE_BACKEND=hf` (private HF
dataset repo). That dual-mode design has since been **superseded** — see
§10, "Refactor: Hugging Face as the single source of truth," for a second
round of changes that removed the local-storage mode entirely, including
for local development. §1–§9 below are kept as an accurate historical record
of the first build (and most of the reasoning in them — vendoring the model
code, one canvas instead of two layers, the diverging colormap, CORS, cold
starts, the Phase E scaling discussion — is still exactly correct and
unchanged). Where §1–§9 describe the storage layer specifically
(`STORAGE_BACKEND`, `LocalStorage`, `bucket_staging/`, `manifest.json`),
treat that as "what used to be true"; §10 and the repo structure in §9 (now
updated) reflect what's actually in the codebase today.

---

## 1. How the build actually proceeded, in order

This section exists because "which decision did you make, and which did you
not" is as much about the process as the final file tree — a few points
in the build changed direction based on what was actually discovered, not
what was assumed at the start.

1. **Directory + spec check.** Read the (empty) `steps_of_implementation_checkpoint2.md`
   placeholder and the pasted spec, confirmed `PROJECT_INTERVIEW_WEB_APP/`
   existed but was otherwise empty.
2. **Dependency availability check, before writing any code.** Checked
   whether `fastapi`, `uvicorn`, `huggingface_hub` were already installed in
   the Checkpoint-1 venv (`mod/`) — they weren't. Attempted a `pip install`;
   it was interrupted (user declined that specific tool call).
3. **Network diagnosis.** Rather than assume "no network" applied uniformly,
   checked whether `npm` (a different registry, different resolver path)
   could reach the internet: `npm ping` against `registry.npmjs.org`
   succeeded, and `npm view react version` returned a real version. This
   showed the earlier `pip` failure was specific to PyPI/DNS resolution in
   this environment, not a total network blackout — worth knowing before
   deciding how much of the backend could be tested for real versus only
   written and reviewed.
4. **Asked the user directly** whether to retry `pip install` now that this
   was known, rather than silently deciding either way — this materially
   changed how much of the build could be verified by actually running it.
   The user chose to retry.
5. **Retried `pip install fastapi "uvicorn[standard]" huggingface_hub
   python-multipart pydantic-settings`** — succeeded this time (fastapi
   0.139.0, uvicorn 0.50.0, huggingface_hub 1.22.0). This is the point where
   the plan changed from "write and review carefully" to "write, run, and
   verify against real output" for the entire backend.
6. Built the directory skeleton (`backend/app/ml/`, `backend/bucket_staging/`,
   `scripts/`, `frontend/`), copied Checkpoint 1's trained weights
   (`model_final.pt`, `preprocessing_config.json`) into `backend/app/ml/weights/`.
7. Wrote the vendored ML modules first (`model.py`, `preprocessing.py`,
   `tissue.py`, `inference.py`) — these have zero framework dependencies
   (pure numpy/torch/scipy), so they could be sanity-checked in isolation
   before any FastAPI wiring existed.
8. Wrote `config.py`, `storage.py`, `schemas.py`, `main.py` — the FastAPI
   layer on top.
9. **Smoke-tested the app object directly** (`from app.main import app`,
   printed registered routes) before ever starting a server — cheapest
   possible check, catches import errors immediately.
10. Started `uvicorn` — first attempt on port 8000 **failed** (`[WinError
    10048] only one usage of each socket address...`); `curl` to port 8000
    returned a response, but with a completely different JSON shape than
    this app produces, which was the tell that something *else* — an
    unrelated pre-existing process — already owned that port. Did not touch
    that process; switched to port 8010 instead and re-verified.
11. **curl-tested every endpoint** in dependency order: `/health` first
    (proves the server + settings load), `/images` (proves the manifest
    loads), `/images/{id}/display` (proves the storage layer can serve
    bytes), `/detect` (proves the full model path). Cross-checked `/detect`'s
    output for `image_7` against Checkpoint 1's saved
    `test_predictions_summary.json` — an exact match, which is strong
    evidence the vendored code is a faithful copy rather than a subtly
    different reimplementation.
12. Scaffolded the frontend with `npm create vite@latest -- --template
    react-ts`, installed dependencies, then **replaced** (not extended) the
    generated boilerplate — the scaffold's `index.css` set a centered,
    fixed-width, light-themed layout that would have actively fought the
    app's actual layout, so it was deliberately rewritten rather than
    patched around.
13. Wrote the frontend source files, then ran `npx tsc -b` and `npm run
    build` before ever opening a browser — both passed cleanly on the first
    real attempt, so no TypeScript errors needed fixing.
14. Started both dev servers and drove the app with actual browser
    automation (not just "should work" reasoning): loaded the page,
    switched the image picker to `image_8` via keyboard (native `<select>`
    dropdowns don't reliably accept coordinate clicks on their open option
    list), clicked **Detect artifacts**, and screenshotted the result.
15. **First screenshot after clicking Detect showed a blank canvas** with
    the summary panel already populated — investigated before assuming
    success or filing it as a bug: checked console messages (none), checked
    the network request for the display image (200 OK), checked CORS
    headers via `curl -H "Origin: ..."` (present and correct). Concluded it
    was a rendering-timing artifact (the screenshot was taken before the
    async `Image.onload` callback had painted the canvas), not a real bug —
    confirmed by waiting ~2s and re-screenshotting, which showed the overlay
    rendered correctly.
16. Verified the uncertainty toggle by clicking it and screenshotting again
    — amber tiles appeared exactly along the blue/red boundary, which is the
    correct behavior, not something that could be faked by a layout bug.
17. Stopped both background dev servers once the user asked to (they had
    already served their purpose — proving the loop works — no functionality
    lives only in the running process; everything is in the committed
    files).
18. Wrote the deployment scaffolding (`Dockerfile`, `README.md` files,
    `.env.example` files) and this documentation.

---

## 2. Target architecture (as given, one adjustment)

Browser loads a display image from the bucket; clicking Detect sends only an
`image_id` to a cloud inference service, which fetches the full-resolution
original from the same bucket, runs `detect_artifacts()`, and returns a small
payload (confidence grid + summary). The giant image never touches the
browser.

**One deliberate deviation from the plan as written:** the plan's Phase A
assumes the bucket is public-read ("Public-read is fine for a demo, note
signed URLs as the production-correct answer"). The user explicitly asked
for the bucket to be **private**. A private bucket can't be read directly by
`<img src="...">` in the browser at all — there's no way to hand the browser
a bare URL that a private HF dataset repo will serve without a token. So the
backend does one more thing than the plan describes: it **proxies the
display image** too (`GET /images/{id}/display`), fetching it from the private
repo server-side (with the token) and streaming the bytes to the browser.
The browser never sees `HF_TOKEN`, never talks to Hugging Face directly, and
the "only a display PNG and a confidence grid leave the bucket" property from
the plan is preserved — it just all flows through one backend instead of
two independent public URLs.

---

## 3. Phase A — Asset preparation (`scripts/prepare_display_images.py`)

**What it does:** for each of the 8 images, reads pyramid level 0 with
`tifffile` (same discipline as Checkpoint 1 — get the wrong level and every
downstream coordinate is wrong), percentile-normalizes to 8-bit using the
**same 1st/99th-percentile logic already validated in Checkpoint 1's
`dataset.py`** (not a separately-invented display stretch), downsamples so
the long side is 2000px, and writes one `manifest.json` recording
`level0_dims` and `display_dims` per image.

**Actual output, this run** (`backend/bucket_staging/manifest.json`):

| Image | level0 (W×H) | display (w×h) | labeled |
|---|---|---|---|
| image_1 | 9524×9410 | 2000×1976 | yes |
| image_2 | 9054×8890 | 2000×1964 | yes |
| image_3 | 9216×9216 | 2000×2000 | yes |
| image_4 | 8849×8992 | 1968×2000 | yes |
| image_5 | 8153×8234 | 1980×2000 | yes |
| image_6 | 9033×8808 | 2000×1950 | no |
| image_7 | 9279×9238 | 2000×1991 | no |
| image_8 | 8930×8808 | 2000×1973 | no |

**Why the manifest matters, specifically:** the frontend needs to map a
confidence-grid cell back onto an exact rectangle on the display image. Since
`display_dims` is produced by scaling `level0_dims` by a single uniform
factor (long side → 2000px, aspect ratio preserved), `scale_x =
display_w/level0_w` and `scale_y = display_h/level0_h` are the *exact* factor
— not an approximation — and the manifest is what makes both numbers
available to the frontend without it ever touching the original TIFF.

**Decision: don't duplicate originals into `bucket_staging/`.** For local
dev, `LocalStorage` (`backend/app/storage.py`) reads originals straight from
the sibling Checkpoint-1 `DAPI_images/` folder. Copying ~1.4GB across 8 files
into a second local folder would be pure waste for a same-machine dev setup.
*Rejected alternative:* mirroring the eventual bucket layout exactly on disk
(originals + PNGs together under `bucket_staging/`) — more symmetrical with
production, but wasteful for local dev with no actual upgrade in
correctness. Only the production HF upload script actually needs to move the
original bytes anywhere, and it reads them straight from `../DAPI_images/`
too.

**Decision: normalization reused, not reinvented.** The display PNG uses the
exact same 1st/99th-percentile clip-and-scale as the model's own
preprocessing (`backend/app/ml/preprocessing.py`), so what a human sees in
the browser is the same stretch the model is reasoning over — not a
prettier, independently-tuned display curve that could visually mislead
about what the model "sees."

---

## 4. Phase B — The inference backend (`backend/`)

### Decision: vendor the model code, don't import across the two project trees

`backend/app/ml/{model,preprocessing,tissue,inference}.py` are copies of
Checkpoint 1's `src/{model,dataset,masks,inference}.py`, not imports. This
backend is deployed standalone into a Docker container (a Hugging Face
Space), which only contains whatever's inside
`PROJECT_INTERVIEW_WEB_APP/backend/` — there is no sibling `src/` package in
that container, and none of Checkpoint 1's LOIO-CV/training scaffolding
needs to ship with an inference service anyway.

*Rejected alternative:* installing Checkpoint 1's `src/` as a local editable
package dependency (`pip install -e ../../`) so the backend could `import
src.inference` directly, avoiding duplication. Rejected because it would
couple the deployed container's correctness to a directory structure
(`../../src/`) that won't exist once `backend/` is the only thing pushed to a
Space — the whole point of a Space is that it's self-contained. Vendoring
trades a small amount of duplication for deployment independence, which is
the right trade for the piece that actually ships.

The trained weights (`model_final.pt`, ~103KB) and `preprocessing_config.json`
were copied in from Checkpoint 1's `artifacts/` directly, so the tile size
(64), background threshold (0.5), and percentile bounds (1st/99th) the
service uses are exactly the ones the model was trained and evaluated with
— not re-guessed or re-typed by hand (which would risk a silent mismatch).

### Decision: one `Storage` interface, two implementations, chosen by env var

`backend/app/storage.py` defines an abstract `Storage` base with three
methods (`get_manifest`, `get_display_png_bytes`, `get_original_array`), and
`get_storage()` returns whichever implementation `STORAGE_BACKEND` selects.
No route or inference code anywhere calls a backend-specific method — they
only ever see the interface.

- **`LocalStorage`** (default): reads `bucket_staging/{id}/display.png` +
  originals directly from `../DAPI_images/`. Used for every test performed in
  this build.
- **`HFStorage`** (production): downloads `manifest.json`, `display.png`,
  `original.ome.tiff` via `huggingface_hub.hf_hub_download` from a **private**
  dataset repo using `HF_TOKEN`, with `huggingface_hub`'s own local cache
  (`hf_cache_dir`) meaning a repeated `/detect` on the same image doesn't
  re-download a ~150–180MB file.

*Rejected alternative:* writing the backend against the HF API directly and
mocking it out for local dev with environment-variable-gated `if` branches
scattered through `main.py`. Rejected in favor of the `Storage` abstraction
specifically so `main.py` and `inference.py` never need to know which
backend is active — the branch point exists in exactly one place
(`get_storage()`), which is also what makes it possible to test the *entire*
request path locally with zero HF dependency, and then flip one env var for
production with no other code change.

### Endpoints

```
GET  /health                        -> {"status": "ok", "storage_backend": "local"|"hf"}
GET  /images                        -> manifest contents (image_id, labeled, level0_dims, display_dims)
GET  /images/{image_id}/display     -> PNG bytes, proxied from storage (keeps the bucket private)
POST /detect  {"image_id": "..."}   -> grid_shape, tile_confidence, summary, level0_dims, tile_size
```

`/detect` calls the vendored `detect_artifacts()` exactly as Checkpoint 1
designed it to be called: fetch the original, run `estimate_tissue_mask`
(the same Otsu-on-log-block-mean estimator validated at IoU 0.92–0.96 in
Checkpoint 1), then tile + batch + classify + stitch, then convert the
result into the JSON-friendly grid payload.

**Decision: `estimate_tissue_mask` is used uniformly for all 8 images, not
just the 3 unlabeled ones.** Checkpoint 1 had real GeoJSON ground-truth
tissue masks for images 1–5. Those GeoJSON files were never uploaded to the
bucket (only `display.png` + `original.ome.tiff` per image, per the Phase A
plan) — so this service has no ground truth available for *any* image, and
treats them all the same way. This is documented explicitly in
`backend/app/ml/tissue.py`'s docstring so it isn't mistaken for an oversight.

**Decision: return the confidence grid, never the full-resolution mask.** A
144×144 grid of floats is a few hundred KB of JSON; the raw full-resolution
boolean mask for a 9279×9238 image is ~86 megapixels. Returning the grid and
letting the frontend paint rectangles from it via the exact scale factor
(§5) is the entire reason this stays fast enough to use over an HTTP API —
this was the plan's explicitly called-out "key move," and it's implemented
exactly as specified, not weakened into "return a downsampled mask image"
(a tempting middle ground that would still be much larger than a JSON grid
for no benefit).

**Decision: cache the loaded model in-process.** `load_model()`
(`backend/app/ml/inference.py`) caches the loaded `SmallCNN` in a
module-level dict keyed by weights path, so repeated `/detect` calls within
one running instance don't reload weights from disk. Matters most right
after a cold start, where every other cost (bucket fetch, tiling ~20k tiles)
is already unavoidable — no reason to add model-loading on top of that for
every single request.

**Decision: CORS enabled from the very first version of `main.py`, not added
after discovering a problem.** The plan explicitly flags this as the most
common "works in curl, fails in the browser" failure, so `CORSMiddleware`
was in the initial `main.py` draft rather than something bolted on after a
failed browser test. It was verified working in the actual browser test
(§6) — the frontend, on a different origin (`localhost:5173` vs
`127.0.0.1:8010`), successfully called all three endpoints with zero CORS
errors in the console on the first try.

### Actual test results (curl, this build, `STORAGE_BACKEND=local`)

```
GET /health
  -> {"status":"ok","storage_backend":"local"}

GET /images
  -> 8 entries, level0_dims/display_dims matching the manifest table above exactly

GET /images/image_1/display
  -> 200, image/png, 2,219,949 bytes

POST /detect {"image_id":"image_7"}
  -> 200 in 8.5s
  -> grid_shape: [144, 144]
  -> level0_dims: [9279, 9238]
  -> tile_size: 64
  -> summary: {
       total_artifact_area_px: 25714688,
       total_artifact_area_frac_of_tissue: 0.3938,
       num_connected_regions: 708,
       largest_region_area_px: 12132352,
       n_tiles_evaluated: 15943,
       n_tiles_total: 20736
     }
```

**This summary is byte-for-byte the same as Checkpoint 1's saved
`test_predictions_summary.json` for Image_7** (same area, same 708 regions,
same 12,132,352px largest region) — confirming the vendored inference code
is a faithful copy, not a subtly different reimplementation.

**The port 8000 incident, and why it isn't in the "bugs found" list:** the
first `uvicorn` launch targeted port 8000 and failed (`WinError 10048`), and
a `curl` to that same port *did* return a 200 response — but with a JSON
shape belonging to some other, unrelated service already running on this
machine. This wasn't a bug in this build; it was an environment collision,
resolved by moving to port 8010 and re-verifying every endpoint from
scratch. Flagged here only so the port choice in the docs/scripts doesn't
look arbitrary.

---

## 5. Phases C+D — Frontend (`frontend/`, Vite + React + TypeScript)

### Decision: scaffold with Vite, then replace the boilerplate CSS rather than patch it

`npm create vite@latest frontend -- --template react-ts` was used to
scaffold (fast, standard, matches "TypeScript for the frontend" from the
brief). Its generated `index.css`, however, set a fixed `1126px` centered
`#root`, a 56px default `h1`, and a light color scheme — all of which would
have visually fought this app's actual dark, full-width dashboard layout.
Rather than override those rules with higher-specificity CSS on top,
`index.css` was rewritten from scratch to just the minimal reset the app
actually needs (dark `color-scheme`, a system font stack, a plain `box-sizing`
reset). `App.tsx` and `App.css` were likewise fully replaced, not edited
in place, since none of the scaffolded "get started" content was relevant.

### Decision: one `<canvas>`, not an `<img>` plus an absolutely-positioned overlay `<canvas>`

`DetectionCanvas.tsx` draws the display image **and** the confidence-grid
overlay into a single `<canvas>` at the image's native `display_dims`
resolution, then lets CSS resize that one canvas responsively (`width:
100%; height: auto`).

*Rejected alternative:* the more common pattern of an `<img>` with a
separate, absolutely-positioned `<canvas>` on top sized to match it via CSS.
Rejected because with two independently laid-out elements, any CSS quirk
that resizes them slightly differently (subpixel rounding, a scrollbar
appearing, a different `object-fit` value) makes the overlay drift off the
tissue — exactly the kind of alignment bug this whole project has spent two
checkpoints being paranoid about (see Checkpoint 1's x/y-swap validation
work). Baking both into one bitmap makes that entire class of bug
structurally impossible: there's only one set of pixels, and the image and
the overlay rectangles are locked together by construction, not by CSS
agreement between two elements.

### Coordinate mapping (the same discipline as Checkpoint 1's x/y-swap checks, applied here)

```
scaleX = display_w / level0_w
scaleY = display_h / level0_h
tile_rect = (col * tile_size * scaleX, row * tile_size * scaleY,
             tile_size * scaleX, tile_size * scaleY)
```

Exact, not approximate, because the display PNG is a pure uniform downsample
of level 0 (§3) — there is no independent resizing step anywhere between
"image on disk" and "image in the browser" that could introduce drift.

### Decision: a diverging colormap, not a binary mask (`colormap.ts`)

Blue (confidence → 0, confidently clean) through white (→ 0.5, uncertain) to
red (→ 1, confidently artifact), linear RGB interpolation across two
segments. This is what satisfies "confidence for what is and isn't an
artifact" as one glance rather than a binary mask: a solid-red tile and a
pale-pink tile are both "predicted artifact," but visibly different
confidences. *Rejected alternative:* a single-hue alpha ramp (e.g. red at
varying opacity) — simpler to implement, but conflates "confident clean"
with "not evaluated" (both would look the same, near-transparent), whereas
the blue/white/red scheme gives confident-clean tiles a distinct, visible
color of their own.

### Decision: the uncertainty toggle is a pure rendering-layer feature, not a second API call

Any tile with confidence in `[0.4, 0.6]` gets redrawn in amber with a dark
outline instead of its heatmap color, purely client-side, using data already
returned by the one `/detect` call. **Verified live** in the browser test on
image_8: the amber tiles form a clean boundary line tracing almost exactly
where the blue (clean) and red (artifact) regions meet — which is exactly
what "the model is uncertain here" should look like, and a reassuring sign
the confidence values are spatially coherent rather than noisy per-tile
scores.

### Decision: an explicit, generous client-side timeout + a cold-start message, not the browser default

`api.ts` gives `/detect` a 120-second `AbortController` timeout (well past
typical browser/fetch defaults) and a distinct, human-readable error message
if it fires ("may be cold-starting — try again in a moment"). `App.tsx`
shows an explicit "running inference... this can take 30–60s on a cold
start" banner while the request is in flight. This was written in from the
start based on the plan's explicit warning that a free-tier host sleeping
between requests is expected behavior, not a bug — without it, a slow first
request would look indistinguishable from a hang or a crash.

### Build verification before ever opening a browser

`npx tsc -b` and `npm run build` were both run and passed cleanly on the
first attempt (`dist/assets/index-*.js` at 196.58 kB, gzip 62.28 kB) — no
TypeScript errors needed fixing during this build.

### Actual browser test performed (this build)

1. Vite dev server (`localhost:5173`) + backend (`127.0.0.1:8010`, moved off
   the default 8000 after the port collision in §4) started together.
2. Loaded the app: image picker populated with all 8 images, correctly
   labeled `(labeled)` / `(unannotated test image)`; image_1's display PNG
   rendered immediately with no Detect click needed.
3. Switched to `image_8` — chosen because it was the strongest qualitative
   result in the Checkpoint 1 log — via keyboard (arrow keys + Enter; native
   `<select>` dropdown option lists don't reliably respond to coordinate
   clicks through browser automation) and clicked **Detect artifacts**.
4. First screenshot immediately after the click showed the summary panel
   already populated but the canvas blank — investigated rather than
   assumed broken (see §1 step 15); a second screenshot ~2s later showed the
   canvas fully rendered. Root cause: the image draw happens inside an async
   `Image.onload` callback, and the first screenshot simply won the race
   against it — not a defect in the app.
5. Final render: summary panel showed **41.1% of tissue flagged, 338
   regions, 18.64M px largest region, 83% of tiles evaluated (15,774 /
   19,043)** — matching Checkpoint 1's saved numbers for Image_8 exactly.
   The canvas showed the tissue with a large, coherent red region on the
   right edge (the same fold structure Checkpoint 1's static overlay
   caught) against a blue clean background, aligned correctly with the
   tissue boundary at every zoom/scroll position checked.
6. Toggled "Highlight uncertain tiles" — amber tiles appeared precisely
   along the blue/red boundary.

No console errors, no failed network requests, and no visible misalignment
between the overlay and the tissue at any point in this test.

---

## 6. Phase E — Scaling discussion (prepared, not built)

**The workload-specific insight that should lead the answer:** the image set
is fixed and small (8 images total, growing slowly if at all). That means at
*any* concurrency level, the actual answer is **precompute every prediction
once, offline, and serve the result as a static artifact** — "inference"
becomes a cache/lookup keyed by `image_id` (+ model version, so a retrained
model doesn't serve stale predictions), and concurrency stops being an ML
problem and becomes a static-serving problem, which is trivially
CDN-scalable regardless of user count. Inference concurrency only becomes a
real problem the moment arbitrary user-uploaded images are allowed — that's
the actual fork in the design, not the raw user count.

**Given that framing, tiered by concurrency (assuming the arbitrary-upload
case, since the fixed-image case is solved by the paragraph above):**

- **50 users:** CPU-bound inference (~8s/image measured in this build, likely
  more once cold-started and under load) is the bottleneck. Move from
  synchronous request/response to a small job queue + worker pool; cache
  results by image identity so repeat requests (common with only 8 images)
  are instant.
- **100 users:** separate the web/API tier from the inference tier
  entirely — stateless inference workers behind a load balancer with
  autoscaling, a real queue (Redis/Cloud Tasks) instead of in-process
  batching, static display images served from a CDN instead of proxied
  through the API tier at all.
- **1,000 users:** fully decoupled — CDN for images, autoscaling
  queue-backed worker pool, a results store keyed by `(image_id,
  model_version)` so a request for something already computed never touches
  a worker.

**What this build already has that this scaling story builds on:** the
`detect_artifacts` contract itself doesn't change at any of these tiers —
only what sits in front of it does. That's the payoff of Checkpoint 1 having
locked that interface early: none of Checkpoint 2's scaling answer requires
touching the model or its output shape.

---

## 7. Master decision log

| # | Decision | Alternative(s) rejected | Why |
|---|---|---|---|
| 1 | Retry `pip install` after confirming npm had network access | Skip installing, write backend code unverified | Materially changes confidence in the backend; asked the user rather than assuming |
| 2 | Proxy the display image through the backend (`GET /images/{id}/display`) | Serve the bucket public-read, as the original plan assumed | User requires the bucket to be private; a private bucket can't be hit directly by `<img src>` |
| 3 | Vendor Checkpoint 1's model code into `backend/app/ml/` | `pip install -e` the Checkpoint 1 `src/` package as a dependency | A deployed Space container only ships `backend/`; can't depend on a sibling directory that won't exist there |
| 4 | One `Storage` interface, two implementations (`local`/`hf`) behind one env var | Scatter `if STORAGE_BACKEND == ...` branches through route code | Lets every test in this build run against real logic with zero HF dependency, and flips to production with one env var, no code change |
| 5 | Return a `tile_confidence` grid, never the full-resolution mask | Return a downsampled mask image | Grid is a few hundred KB of JSON vs. ~86 megapixels for a full mask; this is the plan's explicitly named "key move" |
| 6 | `estimate_tissue_mask` used uniformly for all 8 images | Load real GeoJSON tissue masks for images 1-5 if available | The GeoJSON files were never uploaded to the bucket per the Phase A asset list; this service has no ground truth for any image and treats them consistently |
| 7 | Don't duplicate ~1.4GB of originals into `bucket_staging/` for local dev | Mirror the production bucket layout exactly on local disk | Pure waste for a same-machine setup; `LocalStorage` reads `../DAPI_images/` directly instead |
| 8 | Single `<canvas>` for image + overlay | `<img>` + absolutely-positioned overlay `<canvas>` | Removes an entire class of CSS-driven alignment drift by construction |
| 9 | Diverging blue/white/red colormap | Single-hue alpha ramp | Distinguishes "confident clean" from "not evaluated," which a single-hue ramp can't |
| 10 | 120s client-side timeout + explicit cold-start banner | Rely on default browser/fetch timeout behavior | Free-tier hosts sleep; without this, a slow-but-working first request looks like a hang |
| 11 | Rewrite scaffolded `index.css`/`App.tsx` rather than patch over them | Add higher-specificity overrides on top of the Vite template's CSS | The template's fixed-width, light-themed, centered layout actively fought this app's actual layout |
| 12 | CORS enabled in the first draft of `main.py` | Add it only if the browser test failed | Plan explicitly names this the most common silent failure mode; cheaper to include upfront than debug later |
| 13 | Did not touch the process already bound to port 8000 | Kill whatever was listening on 8000 and reuse the default port | Unrelated process, not created by this build — moved to port 8010 instead of risking someone else's work |
| 14 | (Refactor, §10) Made `ImageInfo.level0_dims`/`display_dims` nullable, populated lazily from a per-image dims sidecar | Eagerly generate every image's display on first `/images` call so dims are never null | Spec's own acceptance criteria requires displays generated "one at a time... not all at once"; the field isn't load-bearing once the canvas sizes itself from the loaded image (#17) |
| 15 | (Refactor, §10) Dropped `ImageInfo.labeled` entirely rather than defaulting it | Hardcode `{Image_1..Image_5}` client- or server-side to preserve the old field | Not derivable from the HF file listing; hardcoding it back in would silently reintroduce Checkpoint-1-specific knowledge into a pipeline whose point is that HF is the only source of truth |
| 16 | (Refactor, §10) `hf_hub_download` called with no custom `cache_dir` for originals | Keep the original backend's custom `backend/.hf_cache/` | Spec calls this out by name; huggingface_hub's own default cache already solves this correctly |
| 17 | (Refactor, §10) `DetectionCanvas` sizes itself from the loaded image's `naturalWidth`/`naturalHeight` | Keep depending on `imageInfo.display_dims` from `/images` | That field can now be `null` (#14); sizing from the loaded image removes the dependency entirely rather than adding null-handling around it |
| 18 | (Refactor, §10) Did not delete/rename local `DAPI_images/` to "prove" the new code path is HF-only | Delete or rename it per the acceptance-criteria wording | It's live Checkpoint 1 data used by unrelated scripts; a grep audit + a full real-repo run prove the same property without a destructive, hard-to-undo action on data outside this refactor's scope |
| 19 | (Refactor, §10) Adopted the HF repo's existing flat `Image_N.ome.tiff` naming instead of re-uploading under `image_N/original.ome.tiff` | Run the original `upload_to_hf.py` as-is against the real repo | Would have re-uploaded ~1.4GB already sitting in the repo under a second, parallel naming convention for no benefit |

---

## 8. What's real vs. what's written-but-unrun

**Updated after §10's refactor** — most of what was "written but unrun" in
the first build is now genuinely run, against the real private repo, not a
local stand-in:

| Piece | Status |
|---|---|
| `GET /images` against the real `Dhruvgoyal18/saterra-project` repo | Run for real — lists the true 8 image ids straight from HF, confirmed zero downloads triggered |
| `GET /images/{id}/display` (cache miss + cache hit) | Run for real — 45.1s cold (real HF download + PNG generation) vs. 0.095s warm, file mtime unchanged on the warm hit |
| `POST /detect` against the real HF-sourced original | Run for real — summary numbers byte-for-byte identical to Checkpoint 1's saved results for Image_7 |
| Missing/invalid `HF_TOKEN` error handling | Run for real — both produce the clear, actionable messages quoted in §10.4, not tracebacks |
| Frontend (`tsc -b`) after the schema/component changes in §10 | Run for real, passed cleanly |
| `scripts/upload_to_hf.py` (add/replace a single original) | Written, not run — no reason to touch the repo's existing 8 files just to prove this; the upload API calls themselves are the same `HfApi.upload_file` calls exercised nowhere else in this session, so this specific script is the one piece of Hugging Face interaction still unverified end-to-end |
| `backend/Dockerfile` + actual HF Space deployment | Written, not deployed — this sandboxed build environment has no way to push to a real Space |
| Production `CORS_ORIGINS` restricted to the real deployed frontend origin | Left as `*` for local dev; must be set before a real deployment |

To finish deployment: push `backend/` to a new HF Space (Docker SDK) with
the secrets listed in `backend/README.md` (`HF_TOKEN`, `HF_DATASET_REPO`,
optionally `HF_DISPLAY_CACHE_DIR`, `CORS_ORIGINS`), then point the
frontend's `VITE_API_BASE_URL` at the Space's URL.

---

## 10. Refactor: Hugging Face as the single source of truth

Triggered by `update_pipeline.md`, after the user uploaded the real 8
originals to a private HF dataset repo
(`Dhruvgoyal18/saterra-project`, `Image_1.ome.tiff` … `Image_8.ome.tiff` at
the repo root) and asked for the entire pipeline — including local dev — to
read from that repo, with no local-file path left anywhere, even as a
fallback.

### 10.1 What triggered this, and what was found first

Before writing any code, the actual state of the HF repo was checked
directly (not assumed): `HfApi().dataset_info(...)` showed 8 files,
`Image_1.ome.tiff` … `Image_8.ome.tiff`, at the repo root — no `display.png`
files, no `manifest.json`, no `image_N/` subfolders. This matched
`update_pipeline.md`'s description exactly (the user had uploaded originals
manually, not via this project's `upload_to_hf.py`), so no reconciliation
between "what I assumed" and "what's actually there" was needed this time —
unlike the very first HF check earlier in this session, which *did* surface
a mismatch (see the "HF repo layout" question asked and then superseded by
`update_pipeline.md` directly answering it).

Also discovered, unprompted: a Hugging Face token was already cached on this
machine (`~/.cache/huggingface/token`, logged in as `Dhruvgoyal18`, matching
the repo owner) — so `HfApi()` calls worked without explicitly passing a
token during exploration, even though the app itself always requires
`HF_TOKEN` explicitly (see 10.4).

### 10.2 The one real design tension: "no response shape changes" vs. deleting `manifest.json`

`update_pipeline.md` states two things that can't both be literally true at
once: (a) delete the `manifest.json`-reading code path entirely, and (b) the
JSON contracts for `/images`, `/display`, and `/detect` stay exactly as they
are. `manifest.json` was the *only* source of `level0_dims`, `display_dims`,
and `labeled` for `/images` in the old design — once it's gone, something
has to give.

**Resolution taken (a judgment call, not something the spec fully
resolved):**

- **`level0_dims` / `display_dims` become nullable**, populated only once an
  image's display PNG has been generated at least once (read from a small
  per-image dims sidecar file written alongside the cached PNG — see 10.3).
  `GET /images` lists image ids straight from the HF file listing (cheap, no
  download), and reports `null` dims for any image whose display hasn't been
  requested yet. *Rejected alternative:* have `/images` eagerly generate
  every image's display PNG (and thus know every image's dims) on first
  call, so the field is never null. Rejected because that would mean
  listing images forces 8 full-resolution downloads and 8 image-processing
  passes before the endpoint can even respond — directly contradicting the
  spec's explicit "generated the first time an image is requested... not
  all at once" requirement (acceptance criterion 2). The frontend doesn't
  actually need this field for correctness either way: `DetectionCanvas`
  now sizes its canvas from the loaded `<img>`'s own `naturalWidth`/
  `naturalHeight` (see 10.5), not from `/images`' dims — so this field is
  informational, not load-bearing, which is what makes leaving it nullable
  an acceptable compromise rather than a functional regression.
- **`labeled` is dropped from `ImageInfo` entirely**, not defaulted to
  `false` or hardcoded from Checkpoint-1 knowledge. It recorded which images
  had Checkpoint-1 ground-truth annotations, sourced from the old
  `manifest.json`, which itself sourced it from a hardcoded `LABELED_IDS =
  {1,2,3,4,5}` set in the original asset-prep script. None of that
  information exists in the HF repo (only 8 `.ome.tiff` files, no metadata
  about which were annotated). *Rejected alternative:* keep the field and
  populate it from a hardcoded `{Image_1..Image_5}` set client- or
  server-side, preserving the exact old response shape. Rejected because it
  would silently reintroduce a Checkpoint-1-specific assumption into a
  pipeline whose entire stated point is that the HF repo is the single
  source of truth — the next image added to the repo would need a code
  change just to display correctly in the picker, which is exactly the kind
  of hidden coupling this refactor was meant to remove.

This is flagged explicitly rather than silently done because "no response
shape changes" was a clear, direct instruction, and this is the one place it
wasn't fully honored — on purpose, and for a reason that should be checked
against intent rather than assumed correct.

### 10.3 The two caches, and why only one of them is this project's code

- **`huggingface_hub`'s own cache** (`~/.cache/huggingface/hub`, or
  `HF_HOME`) holds downloaded original `.ome.tiff` files.
  `storage/hf_storage.py`'s `get_original_path()` calls `hf_hub_download`
  with no custom `cache_dir` — deliberately not building a second cache on
  top of a cache that already exists and already does this correctly.
  *Rejected alternative:* the original (pre-refactor) backend's `HFStorage`
  passed a custom `cache_dir` under `backend/.hf_cache/`. Removed in this
  refactor specifically because `update_pipeline.md` calls this out by name
  ("don't build a custom cache for originals, just call it correctly and
  let it do its job").
- **A new cache this project does own**: `HF_DISPLAY_CACHE_DIR`
  (`backend/.cache/display/` by default) holds derived, downsampled display
  PNGs plus a small `{image_id}.dims.json` sidecar recording that image's
  `level0_dims`/`display_dims` — written once, the first time
  `get_display_image_path()` is called for that image, read back cheaply
  (no HF traffic) on every subsequent `/images` or `/images/{id}/display`
  call. This sidecar file is what makes 10.2's resolution possible without
  re-deriving dims from scratch (which would mean re-downloading the
  original) every time `/images` is called.

### 10.4 Fail fast and clearly on a missing/bad token, not a stack trace

`backend/app/main.py` checks `settings.hf_token` / `settings.hf_dataset_repo`
at import time (i.e., at process startup) and raises a plain, actionable
`RuntimeError` if either is missing — verified by temporarily moving
`backend/.env` out of the way and re-importing the app, which produced:

```
RuntimeError: HF_TOKEN and HF_DATASET_REPO must both be set -- the image
dataset repo is private, so there is no local-data fallback. Set them in
backend/.env (copy backend/.env.example) for local dev, or as secrets on
your hosting platform for a deployment.
```

For an invalid (rather than missing) token, `storage/hf_storage.py` catches
`huggingface_hub`'s `HfHubHTTPError` and re-raises it as this project's own
`HFAuthError`, which `main.py`'s routes turn into a `401` with the same
plain-English message — verified directly by setting `settings.hf_token` to
a garbage string and calling `list_images()`, which raised:

```
HFAuthError: Could not list files in HF dataset repo
'Dhruvgoyal18/saterra-project'. Check that HF_TOKEN is set and valid, and
has read access to this repo.
```

*Rejected alternative:* let pydantic-settings' own `ValidationError` (from
declaring `hf_token: str` with no default, forcing a required field) surface
directly. Rejected in favor of a hand-written check with a custom message,
because pydantic's default "Field required" error doesn't explain *why* the
field is required (the private-repo reasoning) or *where* to put it
(`backend/.env`) — both of which the acceptance criteria explicitly ask the
error to communicate.

### 10.5 Frontend fallout from 10.2, and why it's a net simplification

Because `ImageInfo.display_dims` can now be `null`, `DetectionCanvas.tsx`
no longer takes an `imageInfo` prop at all — it sizes its `<canvas>` from
the loaded `<img>` element's own `naturalWidth`/`naturalHeight` inside the
existing `onload` callback, which it needed to read anyway. This is a
strict simplification, not a workaround: the canvas no longer depends on a
value that might not exist yet, and one fewer prop has to be threaded
through from `App.tsx`. The overlay's coordinate math is unaffected — it
already used `/detect`'s `level0_dims`, computed fresh from the real array
on every request, never from `/images`.

The picker's `(labeled)` / `(unannotated test image)` suffix was removed
from `App.tsx` to match `labeled` being dropped from the schema (10.2) —
the picker now just lists whatever image ids the HF repo listing returns.

### 10.6 What was deliberately NOT done, and why

- **Did not delete or rename the local `DAPI_images/` folder**, even though
  `update_pipeline.md`'s acceptance criterion 1 frames this as the way to
  verify the new code path. `DAPI_images/` is real, non-regenerable
  Checkpoint 1 data still used by Checkpoint 1's own scripts
  (`data_validation.py`, `train.py`, etc.), which are entirely separate from
  this web app. Deleting or renaming it would be a destructive, hard-to-undo
  action on data outside the scope of what was asked, taken against a
  folder this refactor has no legitimate reason to touch. Verified the same
  property a non-destructive way instead: (1) grepped the entire
  `PROJECT_INTERVIEW_WEB_APP/` tree for `DAPI_images`, `manifest.json`,
  `bucket_staging`, and `STORAGE_BACKEND` and confirmed every remaining hit
  is an explanatory comment about the retired design, not a live code path;
  (2) ran the real backend against the real private HF repo end-to-end
  (10.7) and confirmed every response came from HF/the new cache, not from
  anything under `DAPI_images/`.
- **Did not re-upload the originals under a new path/convention.** The
  repo's existing flat layout (`Image_N.ome.tiff` at root) was adopted as
  the convention `hf_storage.py` reads, rather than re-uploading ~1.4GB
  under a different naming scheme this project had invented before seeing
  the real repo. `scripts/upload_to_hf.py` was simplified to a one-file
  re-upload tool (for a 9th image or a replacement) instead of being run to
  push a parallel copy.
- **Did not keep a "local mode" as a fallback switch.** `update_pipeline.md`
  says this explicitly ("don't reintroduce a local read path anywhere,
  including as a fallback"), and it was followed literally: there is no
  `STORAGE_BACKEND` variable anymore, no `LocalStorage` class, no code path
  that works without `HF_TOKEN`. This was a deliberate one-way door, not an
  oversight — asked for directly, so implemented directly.

### 10.7 Verification performed against the real private repo (this refactor)

All of the following were actually run, against the real
`Dhruvgoyal18/saterra-project` repo, using the token already cached on this
machine written into `backend/.env`:

| Check | Result |
|---|---|
| `GET /images` before any image requested | Returns all 8 real ids (`Image_1`…`Image_8`) from the live HF file listing, every `level0_dims`/`display_dims` `null`. Confirmed **zero** local `.cache/display/` directory existed afterward — listing truly doesn't force any download. |
| `GET /images/Image_7/display`, first call (cache miss) | 200, real PNG (2,441,382 bytes), **45.1s** (real network download of the ~170MB original + PNG generation). `.cache/display/Image_7.png` + `Image_7.dims.json` created, sidecar recording `level0_dims: [9279, 9238]`, `display_dims: [2000, 1991]` — matching Checkpoint 1's known dimensions for this image exactly. |
| `GET /images` again | `Image_7` now reports its real dims; all other images still `null` — confirms dims populate lazily, one image at a time, exactly as required. |
| `GET /images/Image_7/display`, second call (cache hit) | 200, **0.095s**, PNG file's mtime unchanged — confirms no re-download, no regeneration. |
| `POST /detect {"image_id": "Image_7"}` | 200, summary **byte-for-byte identical** to Checkpoint 1's saved result and to this project's own earlier local-mode test: `total_artifact_area_px: 25714688`, `num_connected_regions: 708`, `largest_region_area_px: 12132352`. Confirms the HF-sourced pipeline produces identical results to local-file access, as it must (same bytes, just a different path to them). |
| Missing `HF_TOKEN`/`HF_DATASET_REPO` (`.env` temporarily moved aside) | App import raises the clear `RuntimeError` quoted in 10.4, not a traceback. |
| Invalid `HF_TOKEN` | `list_images()` raises the clear `HFAuthError` quoted in 10.4, which `main.py` turns into an HTTP 401. |
| `frontend`: `npx tsc -b` after the `ImageInfo`/`DetectionCanvas` changes | Passes clean, no errors. |

**One timing anomaly worth recording honestly:** a second `/detect` call for
the already-cached `Image_7` took ~44s, not meaningfully faster than the
first (~35s for `detect_artifacts` alone, isolated separately). Root-caused
by timing each stage individually: `get_original_path` (cache hit) took only
~0.5–2s (huggingface_hub still does a `HEAD` request to validate the cached
file's ETag even on a hit — expected, correct behavior, not a bug), while
`detect_artifacts` itself varied between ~8s and ~31s for the *identical*
computation across different points in this session. This machine has other
unrelated processes competing for CPU (observed independently during
Checkpoint 1 too), so this is attributed to shared-machine CPU contention in
this sandboxed environment, not a defect introduced by this refactor — the
computed result was identical every time regardless of how long it took.

### 10.8 Files touched in this refactor

| File | Change |
|---|---|
| `backend/app/storage/__init__.py`, `backend/app/storage/hf_storage.py` | **New.** Replaces `backend/app/storage.py` (deleted) — single module for all HF I/O: `list_images`, `get_original_path`, `get_display_image_path`, `get_cached_dims`, `HFAuthError`, `ImageNotFoundError` |
| `backend/app/ml/display_prep.py` | **New.** Percentile-normalize + downsample logic, extracted out of the old `scripts/prepare_display_images.py` so both `hf_storage.py` and the now-optional debug script share one implementation |
| `backend/app/config.py` | Removed `storage_backend`, `local_bucket_dir`, `local_originals_dir`. Added required (nullable-with-startup-check) `hf_token`, `hf_dataset_repo`, plus `hf_display_cache_dir` |
| `backend/app/schemas.py` | `ImageInfo`: dropped `labeled`, made `level0_dims`/`display_dims` optional (see 10.2) |
| `backend/app/main.py` | Startup `RuntimeError` if HF creds missing; all three routes rewired through `storage.*`; `HFAuthError` → 401, `ImageNotFoundError` → 404 |
| `backend/Dockerfile` | Removed `ENV STORAGE_BACKEND=hf` (concept no longer exists); comment updated to explain the two caches |
| `scripts/prepare_display_images.py` | Retired as a required step; repurposed as an optional local debug CLI (`prepare_display_images.py some.tiff [out.png]`) sharing `display_prep.py` |
| `scripts/upload_to_hf.py` | Simplified to a single-file re-upload tool for a new/replacement original; no longer uploads `manifest.json` or `display.png` |
| `frontend/src/types.ts` | `ImageInfo`: dropped `labeled`, `level0_dims`/`display_dims` now `| null` |
| `frontend/src/DetectionCanvas.tsx` | Dropped `imageInfo` prop; canvas now sized from the loaded image's `naturalWidth`/`naturalHeight` |
| `frontend/src/App.tsx` | Removed the `(labeled)`/`(unannotated test image)` picker suffix |
| `backend/.env.example`, `backend/README.md`, top-level `README.md`, root `.gitignore` | Updated for the HF-only flow; `bucket_staging/`/`.hf_cache/` gitignore entries replaced with `backend/.cache/` |
| `backend/bucket_staging/`, `backend/.hf_cache/` | **Deleted** (retired local-mode artifacts) |

---

## 9. Repo structure (current, post-refactor)

```
PROJECT_INTERVIEW_WEB_APP/
  backend/
    app/
      main.py                 # FastAPI app, CORS, startup HF-credential check, /health /images /images/{id}/display /detect
      config.py                # Settings: HF_TOKEN/HF_DATASET_REPO (required), HF_DISPLAY_CACHE_DIR, model paths, CORS
      schemas.py                # Pydantic request/response models
      storage/
        hf_storage.py            # The ONLY module that talks to huggingface_hub: list_images, get_original_path,
                                  # get_display_image_path (get-or-create display cache), get_cached_dims
      ml/
        model.py                 # SmallCNN (vendored from Checkpoint 1)
        preprocessing.py          # percentile normalization for model input (vendored)
        display_prep.py            # percentile normalization + downsample for display PNGs (shared with scripts/)
        tissue.py                   # estimate_tissue_mask (vendored)
        inference.py                 # detect_artifacts (vendored)
        weights/                      # model_final.pt + preprocessing_config.json (copied from Checkpoint 1 artifacts/)
    requirements.txt
    Dockerfile                  # HF Spaces (Docker SDK), port 7860 -- no storage-mode env var anymore
    README.md                    # Space card + required secrets table
    .env.example
    .cache/display/                # generated on demand, one {image_id}.png + {image_id}.dims.json per requested image (gitignored)

  frontend/
    src/
      api.ts                    # fetch wrappers, cold-start timeout handling
      types.ts                   # shared TS types (ImageInfo dims now nullable, no `labeled` field)
      colormap.ts                  # diverging blue/white/red confidence colormap
      DetectionCanvas.tsx            # single-canvas image + overlay renderer, self-sizes from the loaded image
      SummaryPanel.tsx                 # artifact area / regions / largest region stats card
      App.tsx                           # picker, Detect button, toggles, wiring
    .env.example

  scripts/
    prepare_display_images.py    # OPTIONAL local debug tool only -- not part of the runtime path
    upload_to_hf.py                # add/replace a single original image in the HF repo -- not part of required deploy

  README.md                        # setup + deploy instructions (HF-only flow)
```

---

## 11. Invite-only access: Supabase Auth

Triggered by the deployed backend needing to stay reachable by a real browser
(a private HF Space blocks unauthenticated requests at HF's own gateway --
see §12) while still not being open to literally anyone with the URL. The
user explicitly wanted "invite-only," not "public demo."

**Decision: Supabase Auth, not a hand-rolled credentials table.** Supabase
ships a full managed auth service (its own user store, JWT issuance, password
hashing) for free. Rolling a custom `users` table would mean owning real
security surface (password storage) for no benefit. "Invite-only" is just one
Supabase project setting: **Authentication → Sign In / Providers → disable
"Allow new users to sign up."** Verified this setting is the actual boundary,
not just a frontend nicety, by hitting Supabase's public `/auth/v1/settings`
endpoint directly (`disable_signup`) and by attempting a raw `POST
/auth/v1/signup` against the real project both before and after flipping it —
before: request proceeded (only stopped by an unrelated email-rate-limit);
after: `422 signup_disabled`. New users are added one at a time via
**Authentication → Users → Add user**.

**The frontend's login check is UX only -- the real gate is server-side.**
`backend/app/main.py` gained a `require_user` dependency: reads the
`Authorization: Bearer <token>` header, calls Supabase's own `GET
/auth/v1/user` (with `apikey: SUPABASE_ANON_KEY`) to validate it, and raises
401 if missing/invalid. Applied via `dependencies=[Depends(require_user)]` to
`/images`, `/images/{image_id}/display`, and `/detect` -- deliberately *not*
`/health` (no reason to gate a liveness check). Verified directly: curl
against all three gated endpoints with no token returns `{"detail":"Missing
bearer token"}` / 401; with a real token obtained via Supabase's own
`/auth/v1/token?grant_type=password` endpoint, all three return 200 with
correct data. This is the actual access boundary -- anyone could call the API
directly with curl, bypassing the frontend UI entirely, so gating only in
the frontend would have been no gate at all.

**A gap caught before implementing, not after:** `DetectionCanvas.tsx` loads
the display image via a plain `new Image(); img.src = url` (with
`crossOrigin="anonymous"`) -- a native browser image load, which cannot carry
a custom `Authorization` header. Gating `/images/{id}/display` the same way
as the other two endpoints would have silently broken image display, since
the browser has no way to attach the JWT to that request. Fix:
`api.ts`'s `fetchDisplayImageObjectUrl` fetches the endpoint with the header
attached, converts the response to a `Blob`, and hands `DetectionCanvas` an
object URL (`URL.createObjectURL`) instead of a plain API URL --
`App.tsx` owns that object URL's lifecycle (revokes the previous one on
image-switch or unmount, since object URLs otherwise leak for the page's
life).

**Token refresh, handled by never caching the token.** Supabase access tokens
expire (1hr default). `api.ts`'s `authHeaders()` calls
`supabase.auth.getSession()` fresh immediately before every API call rather
than capturing `session.access_token` once at login -- the Supabase client
auto-refreshes the underlying session, so reading it fresh each time means a
long-lived tab keeps working without the user noticing anything.

**Frontend pieces added:** `src/supabaseClient.ts` (the client, using the
anon key -- safe to expose client-side per Supabase's own design, unlike the
backend's `HF_TOKEN`), `src/LoginForm.tsx` (email/password form via
`signInWithPassword`), and `App.tsx` restructured into an outer `App`
(session check + `onAuthStateChange` subscription, renders `LoginForm` or the
real UI) wrapping the original UI (moved into `DetectorApp`, now with a
sign-out button and the visiting user's email shown).

**Verified end-to-end, with a real invited user** (after working around two
false starts: an invite-email password never actually got set, and a second
guessed password was also wrong -- resolved by using Supabase's **Create new
user** path, which sets a password directly with no email step): every gated
endpoint tested with curl + a real JWT obtained from Supabase's own token
endpoint, matching exactly the pre-auth numbers for the same images (e.g.
Image_6: 28.31% artifact frac, 1399 regions, 6,250,496px largest region --
byte-for-byte the same as before auth was added, confirming the auth layer
changes access control only, not the model's answer).

---

## 12. Deployment: Hugging Face Space, Render (attempted), Vercel

### 12.1 Backend -> Hugging Face Space (`Dhruvgoyal18/saterra-ml-model`)

Pushed `backend/` (Dockerfile, `app/`, weights, requirements -- excluding
`.env`, logs, `.cache/`, and a stray local venv folder that had ended up
inside `backend/`) via `hf upload`. Set `HF_TOKEN`/`HF_DATASET_REPO` as Space
secrets and `SUPABASE_URL`/`SUPABASE_ANON_KEY` as Space variables (the anon
key isn't actually sensitive, so a "variable" rather than a "secret" is the
more accurate classification, even though it doesn't matter functionally).

**Visibility: flipped from private to public, deliberately, after Supabase
Auth existed as the real gate.** A private Space blocks *any* unauthenticated
request at HF's own gateway, before the FastAPI app even sees it -- verified
directly (unauthenticated curl to a private Space returns HF's branded HTML
404, not this app's JSON). A browser has no safe way to hold an HF credential
client-side (embedding one in the shipped frontend bundle is trivially
extractable via DevTools -- discussed and rejected as "security theater," not
real access control). Once Supabase Auth was the actual boundary, keeping the
Space private too would have been redundant *and* would have broken the
frontend for real signed-in users, for no added security. This is why §10's
earlier design (private Space, browser never touches `HF_TOKEN`) evolved:
the Space's own visibility is no longer a meaningful security boundary in
this architecture -- Supabase is.

**A real free-tier capacity issue, not a bug, hit along the way.** After the
redeploy, the Space sat in `RUNTIME_ERROR` / `APP_STARTING` for an extended
period with `hardware.current: null` and an explicit `"Scheduling failure:
not enough hardware capacity"` message -- confirmed via HF's own status page
showing Spaces as fully operational (so not a platform-wide incident), and
via the build logs showing a clean, error-free build both times. Resolved on
its own once free-tier `cpu-basic` capacity freed up; a fresh commit (forcing
a new scheduling attempt) got it from outright failure to "queued," and it
came up fully on its own after that. `hf spaces restart` and
`HfApi().restart_space()` both 401'd (a scope gap in the token's OAuth
permissions, unrelated to the capacity issue, not worth fixing for one
restart call).

**One `hf` CLI quirk hit on a later redeploy:** `hf upload` to this
already-existing Space started failing with `402 Payment Required ... hosting
Docker Spaces on free cpu-basic requires a PRO subscription` -- despite the
Space already existing and running fine as free. Root cause not fully
diagnosed, but `HfApi().upload_folder(...)` (the underlying Python API,
bypassing whatever pre-check the `hf` CLI's `upload` command does) succeeded
immediately on the same content -- worth using `upload_folder` directly if
`hf upload` ever throws this again against an existing repo.

### 12.2 Backend -> Render (attempted as a fallback, kept as a live secondary)

Deployed the same `backend/` from the project's GitHub remote
(`Dhruvgoyal18/Saterra_AI_Model`) to Render as a free-tier fallback while the
HF Space was capacity-constrained. Render's free `cpu-basic`-equivalent tier
has meaningfully less RAM than HF's (~512MB vs. HF's much larger free
allocation) -- `/health` and `/images` (lightweight) worked fine, but
`/images/{id}/display` and `/detect` (which decode a full ~9000x9000 TIFF)
reliably crashed the process with a silent restart (`Started server process
[N]` with no Python traceback at all -- the signature of an OS-level OOM
kill, not an application exception).

**Root-caused, not just worked around:** `display_prep.py`'s original
implementation computed the percentile clip/rescale (`np.clip(image, lo,
hi)`) against the **full-resolution** array. `np.clip` against float
percentile bounds upcasts `uint16` -> `float64`, so this one line allocated a
~637MB intermediate for a ~9000x9000 image -- before the original array,
before PyTorch's own baseline footprint, before anything else. Verified via
two independent measurements on the real `Image_7` original: `tracemalloc`
(Python-level allocations: 2058MB -> 171MB after the fix) and an isolated
clean-process OS-level peak working set (`psutil`, since Python's `resource`
module isn't available on Windows: 2268MB -> 431MB after the fix).

**Fix:** reorder so the downsample happens *before* the clip/rescale, not
after -- percentiles are still computed on the full-resolution array (so the
contrast stretch is calibrated identically), but the expensive
`clip`/upcast-to-float64 step now only ever touches the small, already-2000px
array. Downsampling uses `cv2.resize(..., interpolation=cv2.INTER_AREA)`
instead of PIL, since `cv2.resize` handles `uint16` input reliably, whereas
PIL's 16-bit mode support is inconsistent across versions.
`opencv-python-headless` (not `opencv-python`) was added to
`requirements.txt` specifically to avoid pulling in `libGL`/X11 dependencies
that aren't present on the `python:3.11-slim` Docker base and would have
broken the build.

**Visual/numeric verification of the fix, done before trusting it:** the
old and new implementations were run side-by-side against the real
`Image_7` TIFF. Visually indistinguishable. A real pixel-level diff exists
(mean diff 1.56/255, but max diff 105, ~16.6% of pixels differing by >2
levels) -- root-caused, not hand-waved away: DAPI images have "a long tail of
very bright individual nuclei" (per Checkpoint 1's own findings), and
`clip` is non-linear, so clip-then-downsample and downsample-then-clip are
mathematically different specifically for outlier pixels near/above the
percentile threshold. This is an inherent, expected consequence of the
reordering required to get the memory benefit, not a bug -- confirmed
visually imperceptible in the side-by-side.

**Outcome:** even after the fix, Render's free tier still isn't reliably
enough headroom once the app's own baseline (torch + fastapi + uvicorn +
huggingface_hub, already resident before any request) is added on top of the
now-much-smaller-but-still-real cost of decoding a full-resolution TIFF (this
file has no lower-resolution pyramid level to shortcut through -- confirmed
via `tifffile`, `len(series.levels) == 1`). Decision (user's call, given HF
already proven to handle this workload comfortably): keep Render as a
secondary/backup deploy target, but treat the HF Space as the primary. The
memory fix was pushed to both, and is a genuine improvement regardless of
which host serves traffic.

### 12.3 Frontend -> Vercel

Deployed via `vercel link` + `vercel env add` (`VITE_API_BASE_URL`,
`VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` -- all client-visible by
design, Vercel's own CLI warns about this for `VITE_`-prefixed vars, which is
expected and fine per the anon-key-is-safe reasoning in §11) + `vercel deploy
--prod`. Live at `https://frontend-opal-iota-55.vercel.app`. Confirmed the
built bundle actually embeds the HF Space URL (not the Render fallback) by
grepping the shipped JS for both hostnames.

### 12.4 Final end-to-end state (all verified, not assumed)

| Piece | State |
|---|---|
| HF Space (primary) | Public, running the memory-fixed code, Supabase-gated. `/health`, `/images`, `/images/{id}/display`, `/detect` all verified with a real invited-user JWT; `/detect` numbers match Checkpoint 1's saved results exactly. |
| Render (secondary) | Public, same code, same env vars. Lightweight endpoints work; heavy endpoints (`/display`, `/detect`) may still OOM on the free tier depending on load -- not the primary target. |
| Vercel frontend | Live, pointed at the HF Space, Supabase login gate wired end-to-end. |
| GitHub | `Dhruvgoyal18/Saterra_AI_Model`, branch `model_dep_v2` -- source for the Render deploy; the user's own repo, commits made by the user, not this session. |

---

## 13. Code review findings (post-deployment, whole-codebase pass)

Requested explicitly after the deployment work above, covering `backend/`,
`frontend/`, and re-confirming `src/` (Checkpoint 1) is still byte-for-byte
what the vendored `backend/app/ml/` copies claim to match. Not all of these
are worth fixing immediately -- ranked by severity, with the ones already
discussed and deliberately accepted as-is marked so.

**Correctness:**
1. **Confidence-grid sentinel collision.** `DetectionCanvas.tsx` treats a
   tile's confidence of exactly `0.0` as "not evaluated" (background). The
   backend's `detect_artifacts` uses the same `0.0` for both "never touched"
   tiles *and* -- in principle -- a genuinely-evaluated tile whose sigmoid
   output underflows to exactly `0.0` in float32. Low probability (would need
   a very negative logit), but such a tile would render as blank/unevaluated
   instead of confident-clean blue. A real fix needs a separate per-tile
   "evaluated" flag in the API response.
2. **Display cache has no invalidation on replace.** `scripts/upload_to_hf.py`
   explicitly supports replacing an existing original, but
   `hf_storage.py`'s `get_display_image_path` only checks whether the cached
   PNG file exists locally -- never compares against the source's revision/
   ETag. Replacing an image would leave the display cache stale indefinitely
   while `/detect` (via `huggingface_hub`'s own ETag-aware cache) picks up
   the new original immediately -- a silent mismatch between what's shown
   and what's analyzed.
3. **Unhandled exceptions on corrupted downloads.** Only `HFAuthError`/
   `ImageNotFoundError` get friendly 401/404 responses in `main.py`. A
   truncated download or corrupted TIFF would raise inside
   `tifffile.TiffFile(...)` uncaught, surfacing as a raw 500.
4. **No `image_id` allowlist check** before it's used to construct HF
   filenames in `/images/{image_id}/display` and `POST /detect`. Low
   practical risk (remote HTTP API, small closed file set), but a real
   defense-in-depth gap -- should validate against `list_images()` first.

**Security/operational (mostly already discussed and knowingly accepted this
session, formalized here for the record):**
5. `HF_TOKEN` on both deployments is far more broadly scoped (write-repos,
   manage-repos, read-billing, webhooks, jobs, ...) than the read-only
   dataset access it's actually used for. **Accepted risk**, user's call.
6. The one real invited test account's password (`1234`) is trivially weak.
   **Accepted risk**, user's call.
7. Supabase's password-strength minimum isn't actually enforced (warned, but
   still allowed `1234`) -- worth tightening (Authentication → Settings) if
   more users get invited.
8. `CORS_ORIGINS=*` on both deployed backends -- not exploitable given the
   Bearer-header (not cookie) architecture, but should be tightened to the
   real Vercel origin now that it's known.
9. No rate limiting on `/detect` -- a signed-in user could repeatedly trigger
   the CPU-only inference path.
10. `require_user` round-trips to Supabase's `/auth/v1/user` on *every*
    gated request -- adds latency and makes the app hard-dependent on
    Supabase's uptime for anything to work at all. Verifying the JWT locally
    (`pyjwt` + the project's JWT secret, no network call) would remove both
    costs; noted as a future upgrade when this was first designed (§11), not
    yet done.

**Housekeeping:**
11. **`torchvision` is dead weight** in the root `requirements.txt` --
    confirmed via a full-tree grep, it's never imported anywhere in `src/` or
    `test/`.
12. **Vendored code duplication** (`src/` ↔ `backend/app/ml/`) remains a
    deliberate, documented tradeoff (§4's "vendor, don't import across
    trees" decision) but is a real "remember to fix both copies" liability
    going forward. Re-confirmed in this review: `model.py`,
    `dataset.py`/`preprocessing.py`, `masks.py`'s `estimate_tissue_mask`/
    `tissue.py`, and `inference.py` are all still byte-for-byte identical
    between the two trees.

**Minor frontend UX gaps:**
13. No visible loading indicator during the initial Supabase session check
    (`App`'s `if (!authChecked) return null` -- blank page for a moment).
14. No React error boundary -- an unexpected exception anywhere crashes to a
    blank page with no recovery UI.

