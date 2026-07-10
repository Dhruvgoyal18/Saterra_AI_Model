# Understanding Notes

Personal reference notes on how this project actually works, explained at
interview depth. Appended to over time — each entry is a self-contained
topic.

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
