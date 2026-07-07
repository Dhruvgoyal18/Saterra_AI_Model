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

---

## 8. What's real vs. what's written-but-unrun

| Piece | Status |
|---|---|
| Asset generation (`prepare_display_images.py`) | Run for real against all 8 Checkpoint 1 images |
| Backend (`/health`, `/images`, `/images/{id}/display`, `/detect`) | Run for real, curl-tested, results cross-checked against Checkpoint 1 |
| Frontend (picker, canvas overlay, uncertainty toggle, summary panel) | Run for real in an actual browser against the real backend |
| TypeScript build (`tsc -b`, `vite build`) | Run for real, passed cleanly |
| `scripts/upload_to_hf.py` (private HF dataset repo upload) | Written, not run — no HF token in this build environment |
| `backend/Dockerfile` + HF Space deployment | Written, not deployed — same reason |
| Production `CORS_ORIGINS` restricted to the real deployed frontend origin | Left as `*` for local dev; must be set before a real deployment |

To finish deployment: get an HF token, run `scripts/upload_to_hf.py`, push
`backend/` to a new HF Space (Docker SDK) with the secrets listed in
`backend/README.md`, then point the frontend's `VITE_API_BASE_URL` at the
Space's URL.

---

## 9. Repo structure

```
PROJECT_INTERVIEW_WEB_APP/
  backend/
    app/
      main.py              # FastAPI app, CORS, /health /images /images/{id}/display /detect
      config.py            # Settings (local vs hf storage, model paths, CORS)
      storage.py           # Storage ABC + LocalStorage + HFStorage
      schemas.py           # Pydantic request/response models
      ml/
        model.py           # SmallCNN (vendored from Checkpoint 1)
        preprocessing.py   # percentile normalization (vendored)
        tissue.py          # estimate_tissue_mask (vendored)
        inference.py       # detect_artifacts (vendored)
        weights/           # model_final.pt + preprocessing_config.json (copied from Checkpoint 1 artifacts/)
    requirements.txt
    Dockerfile             # HF Spaces (Docker SDK), port 7860
    README.md              # Space card + secrets table
    .env.example
    bucket_staging/        # generated locally by scripts/prepare_display_images.py (gitignored)

  frontend/
    src/
      api.ts               # fetch wrappers, cold-start timeout handling
      types.ts              # shared TS types matching backend schemas
      colormap.ts            # diverging blue/white/red confidence colormap
      DetectionCanvas.tsx      # single-canvas image + overlay renderer
      SummaryPanel.tsx           # artifact area / regions / largest region stats card
      App.tsx                     # picker, Detect button, toggles, wiring
    .env.example

  scripts/
    prepare_display_images.py   # Phase A: display PNGs + manifest.json
    upload_to_hf.py              # Phase A production: upload to private HF dataset repo

  README.md                      # setup + deploy instructions
```
