# Deep Zoom Viewer — System Understanding & Quick Reference

Reference notes on how `DeepZoomCanvas` / OpenSeadragon deep-zoom streaming works in this app, why it's built this way, and the exact code that implements it. Written for interview review.

---

## 1. The big picture, in one paragraph

The source images are huge, single-resolution 16-bit microscope scans (`.ome.tiff`, ~9500×9400px) with no built-in zoom pyramid. The app serves them two completely different ways: a **flat overview** (one small, permanently-downsampled 8-bit PNG, capped at 2000px) for "see the whole slide at a glance," and a **deep-zoom view** (a server-generated pyramid of 256×256 JPEG tiles, following the standard Deep Zoom Image / DZI format) for "pan and zoom into real native-resolution pixels." OpenSeadragon is the browser-side library that consumes the DZI pyramid and only ever requests the tiles currently visible on screen, so bandwidth and memory scale with what you're looking at, not with the size of the whole image.

---

## 2. Two independent pipelines — don't mix these up

| | Flat overview (`DetectionCanvas`) | Deep-zoom (`DeepZoomCanvas`) |
|---|---|---|
| Endpoint | `GET /images/{id}/display` | `GET /images/{id}/dzi.dzi` + `GET /images/{id}/dzi_files/{level}/{col}_{row}.jpeg` |
| What's returned | ONE flat PNG, capped at 2000px long side | XML descriptor, then many 256×256 JPEG tiles, lazily |
| Resolution ceiling | 2000×1976 (Image_1) — fixed forever | Full native 9524×9410 (Image_1) — real detail |
| Generator | `ml/display_prep.py: to_display_png` | `ml/tiling.py: build_pyramid` + `extract_tile` |
| Cache | `hf_display_cache_dir` (1 PNG + dims json per image) | `hf_tile_cache_dir` (per-tile JPEGs + meta json) + in-memory LRU pyramid (max 2 images) |

Both are downstream of the same original file (`hf_storage.py: get_original_path`), fetched once from the private HF dataset repo.

**A third, unrelated grid also exists**: the ML classifier's 64×64 tile grid (`ml/inference.py`), used only to produce the confidence-overlay heatmap. Different problem entirely — do not conflate with the 256px DZI tile size.

---

## 3. Full request flow, start to max zoom

1. `DeepZoomCanvas` mounts → calls `authHeaders()` (`api.ts`) → fresh Supabase bearer token, fetched new every time (tokens expire hourly).
2. Constructs `OpenSeadragon({ tileSources: '.../dzi.dzi', loadTilesWithAjax: true, ajaxHeaders: headers })`.
3. `GET /images/Image_1/dzi.dzi` → backend returns XML: width, height (full native res), tile size (256), overlap (0), format (jpeg).
4. OSD computes, client-side, no network call: `max_level = ceil(log2(max(w,h)))` → 14 for Image_1. Also derives tile-grid dimensions per level using the same ceil-based formula the backend uses (`dzi_level_dims`) — these must match exactly or edge tiles break.
5. OSD requests a couple of low-res tiles first to paint the zoomed-out view.
6. As you zoom/pan, OSD fires `GET /images/Image_1/dzi_files/{level}/{col}_{row}.jpeg` only for tiles inside (plus a small margin around) the current viewport — every request carries the bearer token via `ajaxHeaders`.
7. Backend (`main.py: get_dzi_tile`) parses `col_row.jpeg` → calls `tile_storage.get_or_build_tile`.
8. `get_or_build_tile`: validate level/col/row are in range → disk cache hit? return immediately → miss: get-or-build this image's full decoded pyramid (in-memory LRU, max 2 images) → grab the array for the requested level → `extract_tile` slices out just that one 256×256 region, normalizes it, saves as JPEG (quality 90) → return path.
9. At max zoom you're requesting real level-14 (native-resolution) tiles — only the ~4-6 actually on screen, never all 1,406 possible tiles at that level.

---

## 4. Key files and what each one owns

| File | Owns |
|---|---|
| `frontend/src/api.ts` | Fetching a fresh auth token per call; all authed `fetch()` wrappers |
| `frontend/src/DeepZoomCanvas.tsx` | Constructing/tearing down the OSD viewer; the confidence-overlay canvas on top |
| `backend/app/main.py` | Routes: `/dzi.dzi`, `/dzi_files/...`, `/display`, `/detect` — all auth-gated |
| `backend/app/storage/hf_storage.py` | The ONLY module that talks to Hugging Face Hub; flat-PNG get-or-create cache |
| `backend/app/ml/display_prep.py` | Flat overview generation: percentile-normalize + downsample to ≤2000px, 8-bit |
| `backend/app/storage/tile_storage.py` | DZI cache/orchestration: meta sidecar, descriptor XML, get-or-build one tile, in-memory pyramid LRU + lock |
| `backend/app/ml/tiling.py` | DZI math constants + pyramid building + `extract_tile` (the memory-safe slice-then-convert step) |
| `backend/app/ml/inference.py` | Unrelated: the 64px classifier grid for the artifact-detection overlay |

**Reading order if reviewing the code cold:** `api.ts` → `DeepZoomCanvas.tsx` → `main.py` (the 3 image routes) → `tile_storage.py` → `tiling.py` → (contrast with) `hf_storage.py` + `display_prep.py`.

---

## 5. The memory-safety trick — analogy → code mapping

This is the one most likely to get a follow-up question, so map it explicitly:

| Analogy | Real concept | Code |
|---|---|---|
| Whole warehouse of small boxes | Full image kept as 16-bit (uint16) | `pyramid = _get_or_build_pyramid(image_id)` |
| Pulling one crate off the shelf | Slicing out just one tile's pixels | `tile = level_array[y0:y1, x0:x1]` |
| Repacking a crate into bigger boxes | uint16 → float64 upcast (needed for decimal math) | `scaled = ((clipped - lo) / (hi - lo) * 255)` |
| Putting the result back in small boxes | float64 → uint8, temp array discarded | `.astype(np.uint8)` |
| The old bug: repacking the whole warehouse at once | Upcasting the ENTIRE image to float64 in one shot | old `display_prep.py`, pre-fix — ~684 MB spike |
| The fix: repack one crate, use it, let it go | Only ever upcasting one 256×256 slice at a time | `extract_tile()` in `tiling.py` |

**Why 256 specifically, not smaller/bigger:** smaller (64/128) → tile count explodes ~4x per halving (area scales with the square), way more HTTP round trips and per-file JPEG overhead. Bigger (1024) → slower first paint, more wasted prefetch bytes, and it re-inflates the exact float64 slice size the fix was built to keep small (8 MB instead of 0.5 MB). No formal benchmark exists in this repo for 256 — it's the DZI/OpenSeadragon ecosystem's long-standing default.

---

## 6. Numbers worth memorizing (Image_1: 9524 × 9410, 16-bit)

| Quantity | Value |
|---|---|
| Flat overview cap | 2000px long side → 2000×1976, 8-bit |
| DZI descriptor `<Size>` | Full native res, 9524×9410 — never downsampled |
| DZI tile size / overlap / format | 256×256, 0, JPEG q90 |
| `max_level` | 14 (`ceil(log2(9524))`) |
| Full-res (level 14) tile grid | 38 cols × 37 rows = 1,406 tiles total (only visible ones ever get built) |
| Classifier grid (unrelated, 64px) | 147 rows × 148 cols ≈ 21,756 tiles |
| Percentile normalization | 1st/99th percentile, computed once per image, cached, reused for every tile at every level (no seams) |
| Old OOM bug | ~684 MB (whole image upcast to float64 at once) |
| One 256×256 tile upcast to float64 | ~0.5 MB |
| One 1024×1024 tile upcast to float64 | ~8 MB (16×) |
| Whole pyramid kept as uint16 in memory | ~171 MB per image (cheap; capped at 2 images via LRU) |

---

## 7. Likely cross-questions, one-liners

- **Why tile instead of sending one big image?** Bandwidth/memory don't scale to a 9500×9400 bitmap; tiling makes cost proportional to what's actually viewed.
- **Why 8-bit if the source is 16-bit?** Browsers can't render 16-bit; conversion is unavoidable — the design question is only *when* it happens (per-tile, from full-res data, vs. once on an already-shrunk overview).
- **Why not openslide-python?** No native pyramid to exploit in these files anyway, and it'd add a system package to a minimal Docker image for no real gain.
- **Why normalize once, not per tile?** Prevents visible brightness seams at tile boundaries.
- **What if the auth token expires mid-session?** Real gap — token is baked into the viewer at construction time, not refreshed until you switch images.
- **Why lock the in-memory pyramid cache?** FastAPI runs sync routes in a threadpool; concurrent evict+insert on the LRU dict isn't atomic without one.
- **Did you benchmark the 256px tile size?** No — inherited ecosystem default, single tunable constant, cheap to revisit if profiling ever justified it.

---

## 8. What is the DZI XML descriptor, exactly?

A small metadata file, not an image — it tells OpenSeadragon the *shape* of the pyramid so it can compute everything else itself (levels, tile grid per level, tile URLs). It does NOT list individual tiles or URLs. Real example for Image_1 (`get_dzi_descriptor_xml`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="jpeg"
       Overlap="0"
       TileSize="256">
  <Size Width="9524" Height="9410"/>
</Image>
```

OSD reads `Width`/`Height`/`TileSize`/`Overlap` and derives `max_level`, per-level dimensions, and tile-grid size using the exact same formulas the backend uses. Tiles themselves are generated later, lazily, only on request.

---

## 9. AJAX and `ajaxHeaders`

AJAX = fetching data from the server via JavaScript (`fetch`/`XMLHttpRequest`) without a full page reload. Matters here because a plain `<img src="...">` tag — OSD's default tile-loading method — **cannot carry a custom `Authorization` header**, and every tile endpoint here is auth-gated.

- `loadTilesWithAjax: true` → tells OSD to load tiles via `fetch` instead of `<img>`, which *can* carry custom headers.
- `ajaxHeaders: headers` → the actual header object (`{ Authorization: "Bearer <token>" }`) attached to every one of those tile fetches.

Together: what makes a private, login-gated image collection work with a viewer library that normally assumes tiles are publicly loadable.

---

## 10. `get_or_build_tile` — full walkthrough with a real example (level 12, col 2, row 5 → `12/2_5.jpeg`)

**Two independent caches — know which is which:**

1. **Disk, keyed by exact (image_id, level, col, row):** `_tile_path()` → `.cache/tiles/Image_1/12/2_5.jpeg`. First request generates it; every request after (anyone, anytime, until container restart) is an instant file read.
2. **In-memory, keyed by image_id ONLY (not level/col/row):** the full decoded pyramid (`{level: array}` for every level), LRU-capped to 2 images. Not "have we served this tile before" — it's "have we already decoded this image's TIFF recently," so a burst of tile requests during one zoom session doesn't re-decode from scratch each time.

**Step by step for `12/2_5.jpeg`:**

1. `meta = get_or_build_meta("Image_1")` → cached sidecar: `{width: 9524, height: 9410, max_level: 14, lo: 9.0, hi: 1460.0}`.
2. Validate level 12 is in range (0–14) ✓.
3. Compute level 12's dimensions: `2381 × 2353` (native downsampled 4× — two levels below max). Tile grid at level 12: `10 cols × 10 rows`. Is `col=2, row=5` inside that? ✓.
4. Check disk for `.cache/tiles/Image_1/12/2_5.jpeg` → hit: return immediately. Miss: continue.
5. `pyramid = _get_or_build_pyramid("Image_1")` → reuse cached decoded pyramid, or decode+cascade-resize the TIFF if this is the first request for this image.
6. `level_array = pyramid[12]` → just the 2381×2353 array for this level.
7. `extract_tile(level_array, col=2, row=5, tile_size=256, lo=9.0, hi=1460.0)` slices out pixel columns **512–768**, rows **1280–1536** of the level-12 array (256×256). Mapped back to the *original* 9524×9410 image (level 12 is 4× downsampled), that's columns **2048–3072**, rows **5120–6144**.
8. Normalize that slice (clip/rescale to 8-bit — the memory-safety trick from section 5) → save as `.cache/tiles/Image_1/12/2_5.jpeg`, quality 90.
9. Return the path → route wraps it as `FileResponse(tile_path, media_type="image/jpeg")`.

**How the request reaches the frontend, full loop:** OSD (in-browser) computes it needs tile 12/2_5 from current pan/zoom → builds the URL → `fetch()` with bearer token (`ajaxHeaders`) → `main.py: get_dzi_tile` → `get_or_build_tile` (steps above) → JPEG bytes returned over HTTP → browser receives → OSD decodes and draws it at the screen position matching level/col/row (already known client-side, since OSD computed it needed exactly this tile in the first place).

---

## 11. Where does the tile cache actually live? (local machine vs. deployed server)

Same code, same relative path (`config.py`: `Path(__file__).resolve().parent.parent / ".cache" / "tiles"`), very different durability depending on where it runs:

- **Local dev:** resolves to a real folder next to the backend code, e.g. `.../backend/.cache/tiles/Image_1/12/2_5.jpeg`. A normal folder on a normal disk — survives forever until manually deleted, even across reboots.
- **Deployed (Hugging Face Spaces, per the `Dockerfile`):** `WORKDIR` is `/app`, so the same code creates `/app/.cache/tiles/Image_1/12/2_5.jpeg` — but inside the **container's writable layer**, which the Dockerfile's own comment says is *not persistent*: "no persistent disk beyond the image layer and whatever the app writes at runtime."
- **Consequence:** the folder doesn't exist at container start — it's created live, on the first tile request (`mkdir(parents=True, exist_ok=True)` inside `get_or_build_tile`). It's wiped completely on every container restart (redeploys, or the Space sleeping-then-waking after being idle — the same cold-start behavior `api.ts` has a comment about). First tile request after a restart is a cold miss again, even for tiles served instantly the day before. Explicitly accepted, not a bug — the durable source of truth is the private HF dataset repo; everything derived from it is treated as cheap-to-regenerate scratch space.
- **Why it's invisible in the Space's "Files" tab (the git repo browser):** that view only shows git-tracked source code — what's actually pushed/committed (`ml/`, `storage/`, `main.py`, etc., copied in by `COPY app ./app` in the Dockerfile). `.cache/` is never committed; it only comes into existence inside a *running* container, which the Files tab has no visibility into. Free-tier Spaces don't expose a shell/terminal to browse it directly either — the only way to "see" it working is indirectly, via the second request for the same tile being faster than the first.
- **Scaling caveat (good follow-up to have ready):** this cache is local disk, per container. Fine for a single Space (one container) today. If ever scaled to multiple backend instances, each would have its own separate, unshared `.cache/tiles` — a tile cached on instance A wouldn't exist for a request landing on instance B. Would need shared storage (S3, CDN, Redis) to fix.
