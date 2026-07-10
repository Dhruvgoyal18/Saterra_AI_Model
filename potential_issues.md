# Potential Issues — Whole-Codebase Review

Findings from a full pass over `src/`, `test/`, `PROJECT_INTERVIEW_WEB_APP/backend/`,
and `PROJECT_INTERVIEW_WEB_APP/frontend/`, done after the Checkpoint 2 deployment
work (Supabase Auth, Hugging Face Space / Render / Vercel deploys). None of these
blocked deployment; they're recorded here for prioritization later. Full narrative
context lives in `PROJECT_INTERVIEW_WEB_APP/steps_of_implementation_checkpoint2.md`
§13 and `step_of_implementation_checkpoint1.md` §9 — this file is the flat,
scannable checklist version of the same findings.

---

## Correctness bugs

### 1. Confidence-grid sentinel collision
**Where:** `PROJECT_INTERVIEW_WEB_APP/frontend/src/DetectionCanvas.tsx` (the
`if (confidence === 0) continue` check) vs. `backend/app/ml/inference.py` /
`src/inference.py`'s `detect_artifacts`.

The frontend treats a tile's confidence of exactly `0.0` as "not evaluated"
(background, skipped by the tissue filter). But the backend's confidence grid
uses `0.0` as its initial/default value for *every* tile — both ones that were
genuinely never evaluated, and (in principle) a genuinely-evaluated tile whose
sigmoid output underflows to exactly `0.0` in float32 (possible for a very
negative logit). Low probability, but such a tile would incorrectly render as
blank/unevaluated instead of confident-clean blue.

**Fix would need:** a separate per-tile "evaluated" boolean in the API response
(e.g. alongside `tile_confidence`), rather than inferring it from the confidence
value itself.

### 2. Display-image cache has no invalidation on replace
**Where:** `backend/app/storage/hf_storage.py`'s `get_display_image_path`.

`scripts/upload_to_hf.py`'s own docstring explicitly supports replacing an
existing original image. But `get_display_image_path` only checks whether the
cached PNG file exists locally (`png_path.exists()`) — it never compares
against the source file's revision/ETag on the HF Hub. If an image is ever
replaced, the cached display PNG stays stale indefinitely, while `/detect`
(via `huggingface_hub`'s own ETag-aware download cache) picks up the new
original immediately. Result: the canvas shows the old image while the
detection numbers reflect the new one — a silent, confusing mismatch.

**Fix would need:** compare the cached PNG's source against the current HF
file's ETag/commit hash before serving it, or just invalidate the whole
`.cache/display/` entry whenever `upload_to_hf.py` is run for that image.

### 3. Unhandled exceptions on corrupted downloads
**Where:** `backend/app/main.py` — `/detect` and `/images/{image_id}/display`.

Only `storage.HFAuthError` and `storage.ImageNotFoundError` are caught and
turned into friendly 401/404 responses. A truncated download or corrupted
TIFF would raise inside `tifffile.TiffFile(...)` uncaught, surfacing as a raw,
unfriendly 500 with no actionable message.

### 4. No `image_id` allowlist check
**Where:** `backend/app/main.py` (`DetectRequest.image_id`), `backend/app/storage/hf_storage.py`.

`image_id` is used directly to construct HF filenames (`f"{image_id}.ome.tiff"`)
with no validation against the known set from `list_images()` first. Low
practical risk (this hits a remote HTTP API against a small, closed file set,
not a local filesystem), but it's a defense-in-depth gap worth closing —
validate `image_id` against `list_images()` before using it.

---

## Security / operational

### 5. `HF_TOKEN` is over-scoped — accepted risk
Both deployments (HF Space, Render) use an HF token whose OAuth scopes include
`manage-repos`, `write-repos`, `read-billing`, `webhooks`, `jobs`, etc., when
the app only ever needs read-only access to one private dataset repo. If the
container were ever compromised, this token's blast radius is much larger than
necessary. **Discussed with the user and knowingly kept as-is** given the
project's stakes; revisit with a fine-grained, read-only token if this ever
becomes more than a demo.

### 6. Test account password is trivially weak — accepted risk
The one real invited Supabase user's password is `1234`. **Discussed with the
user and knowingly kept as-is.**

### 7. Supabase's password-strength minimum isn't enforced
Supabase's signup API returned a "weak_password" warning for `1234` but still
created the account. Worth tightening under Authentication → Settings if more
users are ever invited.

### 8. `CORS_ORIGINS=*` on both deployed backends
Not exploitable today — the architecture uses a `Authorization: Bearer` header
(not cookies), so this isn't CSRF-able the way wildcard CORS + cookies would
be. Still, now that the real Vercel origin (`https://frontend-opal-iota-55.vercel.app`)
is known, tightening `CORS_ORIGINS` to that exact value is good hygiene.

### 9. No rate limiting on `/detect`
A signed-in user (malicious or just a buggy script/retry loop) could
repeatedly trigger the CPU-only inference path, which is the most expensive
operation in the app. No protection against this today beyond Supabase's
default auth-endpoint rate limits (which don't cover this app's own routes at
all).

### 10. `require_user` round-trips to Supabase on every gated request
**Where:** `backend/app/main.py`'s `require_user` dependency.

Every call to `/images`, `/images/{id}/display`, and `/detect` makes a fresh
network call to Supabase's `GET /auth/v1/user` to validate the bearer token.
This adds latency to every request and makes the entire app hard-dependent on
Supabase's uptime — if Supabase has an outage, none of the gated endpoints
work even though the actual backend logic is otherwise fine.

**Fix would need:** verify the JWT signature locally (`pyjwt` + the Supabase
project's JWT secret, from Settings → API) instead of a network call per
request. This was the planned "upgrade later" path when the auth layer was
first designed, not yet implemented.

---

## Housekeeping

### 11. `torchvision` is a dead dependency
**Where:** root `requirements.txt`.

Listed but never imported anywhere in `src/` or `test/` (confirmed via a
full-tree grep) — `src/model.py`'s `SmallCNN` is defined from scratch and
never touches `torchvision`'s pretrained models or transforms. Safe to drop.

### 12. Vendored code duplication between `src/` and `backend/app/ml/`
Deliberate and documented (Checkpoint 2's "vendor, don't import across
project trees" decision, for deployment independence — a Hugging Face Space
only ships `backend/`, not a sibling `src/` package). Re-confirmed in this
review that `model.py`, `preprocessing.py`/`dataset.py`'s normalization
functions, `tissue.py`/`masks.py`'s `estimate_tissue_mask`, and `inference.py`
are all still byte-for-byte identical between the two trees. Still a real
"remember to fix both copies" maintenance liability going forward if either
ever needs a bugfix.

---

## Minor frontend UX gaps

### 13. No visible loading indicator during initial session check
**Where:** `frontend/src/App.tsx` — `if (!authChecked) return null`.

Renders nothing (blank page) for a moment while `supabase.auth.getSession()`
resolves on first load. Cosmetic only.

### 14. No React error boundary
An unexpected exception anywhere in the component tree crashes the whole app
to a blank page with no recovery UI (no "something went wrong, reload"
fallback).
