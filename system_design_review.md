# System Design Review — DAPI Artifact Detector

Full-codebase pass over `src/` (Checkpoint 1 training pipeline) and `PROJECT_INTERVIEW_WEB_APP/` (FastAPI backend + React frontend, Checkpoints 2–3). Cross-checked against the project's own prior analysis docs (`potential_issues.md`, `model_decision_audit.md`, `understanding.md`, the `*_interview_notes.md` files) and the live code, to confirm what's still true, what's been fixed since, and what's new.

---

## 1. Architecture, one paragraph

A React/TypeScript SPA (Vercel) talks to a stateless FastAPI backend (Docker, deployed as a Hugging Face Space) over a REST API. Auth is Supabase (invite-only, JWT bearer tokens, validated by round-trip, not locally). Image data lives in a private Hugging Face Hub dataset repo — the single source of truth in every environment, no local-disk fallback. The backend serves three things: a flat downsampled preview PNG, a server-generated Deep Zoom (DZI) tile pyramid for real pan/zoom at native resolution, and a PyTorch CNN (`SmallCNN`) that classifies 64×64 tiles as artifact/clean. A `/chat` endpoint wraps the same detection function behind Groq LLM function-calling. No database anywhere in the hot path — two on-disk caches (display PNGs, DZI tiles) and one in-memory cache (decoded pyramids, chat history) do all the state-holding, and all three are explicitly ephemeral (wiped on every container restart, since HF Spaces gives no persistent disk).

---

## 2. Open system design problems

These are the real, unsolved design questions in this codebase — the ones worth being ready to discuss deeply, not just recite.

### 2.1 Single point of coupling: auth round-trips to Supabase on every request
`require_user` calls `GET /auth/v1/user` on Supabase for every gated request (`/images`, `/display`, `/dzi.dzi`, `/dzi_files/...`, `/detect`, `/chat`). This buys instant revocation but means: (a) added latency on literally every request, (b) the whole app is hard-down if Supabase has an outage, even though none of the actual backend logic depends on Supabase being up. The documented alternative — local JWT verification with the project's signing secret — trades revocation-on-demand for speed and independence. Neither is implemented as a hybrid (e.g., verify locally + revalidate periodically), which is itself a fair follow-up design question.

### 2.2 No rate limiting on the one expensive endpoint
`/detect` runs a full tiled CNN pass over a ~9000×9000px image — the single most CPU-expensive operation in the app — with zero rate limiting beyond whatever Supabase's own auth endpoints enforce (which don't cover this app's routes at all). A signed-in user, or a chat conversation that keeps re-triggering `run_artifact_detection`, can hammer this freely. Nothing budgets concurrent inference either — there's no queue, no semaphore, no backpressure if two users hit `/detect` at once on a single-worker container.

### 2.3 Per-instance local caches don't survive scale-out
Both the display-PNG cache and the DZI tile cache (`hf_display_cache_dir`, `hf_tile_cache_dir`) are plain local disk, and the in-memory pyramid LRU is a process-local dict. This is fine for exactly one container. The moment this scales to more than one backend instance (for throughput, or multi-region), each instance has its own cold, disjoint cache — a tile warmed on instance A is a cold miss on instance B, and the "second request is fast" guarantee silently stops holding depending on load-balancer routing. Fixing this needs shared storage (S3/CDN for tiles, Redis or similar for the pyramid cache) — a real, not-yet-designed piece.

### 2.4 Chat conversation state: unbounded, in-memory, single-instance
`chat.py`'s `_conversations: dict[str, list[dict]]` has no eviction policy, no TTL, no size cap — unlike the pyramid cache, which is deliberately LRU-capped at 2 images. Every new `conversation_id` is a permanent entry until the process restarts. On a long-running deployment with real traffic this is unbounded memory growth, and it also doesn't survive or share across multiple instances (a request landing on a different worker than the one that holds the conversation loses all history). This is a genuine gap in an otherwise-consistent "cap what's cached" design pattern used everywhere else in the codebase.

### 2.5 Evaluation/serving threshold mismatch
`prob_threshold` was tuned from 0.5 → 0.7 via a proper 5-fold LOIO sweep (`model_decision_audit.md`), and that 0.7 is what's actually deployed (`backend/app/config.py`). But `src/train.py`'s `run_loio_cv` still hardcodes `y_pred = (y_prob >= 0.5)` when computing the IoU metric reported in `loio_cv_results.json`. The evaluation artifact and the deployed decision boundary have quietly diverged — the reported IoU numbers don't describe the model that's actually running in production. Worth deciding: should `run_loio_cv` read `prob_threshold` from the same config the backend uses, or is IoU-at-0.5 intentionally kept as a fixed reference point independent of serving threshold? Neither is currently documented as a choice — it looks like drift, not a decision.

### 2.6 Small-data generalization is the real ML constraint, not architecture
5 labeled images total, LOIO cross-validation, no held-out 6th image to validate any hyperparameter choice against. `model_decision_audit.md` is unusually honest about this: 7 of 14 accuracy-relevant decisions are stated assumptions, never swept (tile size, tau, background threshold, purity filter, pos_weight, training budget). This isn't really a "fix it" item — with only 5 images, aggressive tuning risks overfitting the *design choices* to those 5 images even without fitting a single model to all of them. The open design question is what the next real lever is: more labeled data, heavier augmentation (see the U-Net plan), or accepting the current ceiling.

### 2.7 U-Net migration is scoped but has its own unresolved questions
`unet_colab_migration_plan.md` is a real, detailed design doc (patch size, sampling strategy, augmentation, architecture, GPU training loop) but explicitly punts on deployment: if training moves to GPU (Colab) but the serving host stays CPU-only, a U-Net running per-pixel inference over a 9000×9000 slide (versus the current sparse, tissue-gated 64×64 tile batches) is a materially different compute budget at serving time. That's flagged in the doc's own §9 as a separate decision, not yet made.

### 2.8 No observability
No metrics (request latency, cache hit rate, inference time, cold-start frequency), no structured/correlatable logging beyond basic `logging.info`, no alerting. On a free-tier host that sleeps when idle, cold-start behavior is a known-but-unmeasured cost (`api.ts` has a client-side comment about it, but there's no server-side signal of how often it happens or how long it takes).

### 2.9 No automated test suite beyond one data-validation script
`test/data_validation.py` checks annotation/mask correctness (the polygon-orientation pitfall it rules out is a good sign of care), but there's no unit/integration testing for the FastAPI routes, the DZI math (`dzi_level_dims`, `extract_tile` boundary slicing), the auth dependency's error paths, or the chat tool-calling flow. Several of the bugs below (the confidence-sentinel collision, the cache-invalidation gap) are exactly the kind of thing a route-level or property-based test would have caught.

---

## 3. Interview question bank

Organized by area. The three `*_interview_notes.md` files already in the repo (`supabase_auth_interview_notes.md`, `deep_zoom_viewer_interview_notes.md`) are excellent and worth re-reading directly — I'm not duplicating their full Q&A here, just indexing them and adding what they don't cover.

### 3.1 Auth & access control (full Q&A bank already in `supabase_auth_interview_notes.md`)
Highlights to be ready for: why round-trip validation over local JWT verification (§7/§11 of that doc); what the actual security boundary is (`require_user`, not the login screen); the `<img>`-can't-carry-headers problem and its fix (object URLs / `ajaxHeaders`); why 503 vs 401 vs 500 are used differently; why the anon key is safe to ship but `HF_TOKEN` never is.

Add-on questions not yet in that file:
- "Your JWT validation round-trips per request. What's the cheapest way to cut that cost without losing revocation entirely?" (Answer shape: short local-cache of "this token was valid as of T," re-validate against Supabase only after some TTL — a middle ground between the two extremes already documented.)
- "If you scaled to multiple backend instances, does anything about the auth design break?" (No — `require_user` is stateless per request, unlike the caches below.)

### 3.2 Deep zoom / tiling (full Q&A bank already in `deep_zoom_viewer_interview_notes.md`)
Highlights: why DZI tiling instead of shipping one huge image; the memory-safety "slice-then-upcast" fix and the OOM bug it replaced; why normalization stats are computed once per image, not per tile (seam avoidance); the two independent caches (disk vs. in-memory LRU) and why the LRU needs a lock under FastAPI's threadpool; why the tile cache doesn't survive a container restart, and the multi-instance scaling caveat it names explicitly (§11) — same issue as §2.3 above, already self-identified.

Add-on questions:
- "Why 256px tiles and not something else?" — answer honestly: ecosystem default, never benchmarked in this repo (the doc says so directly — good, since claiming a fake benchmark would be worse than admitting it wasn't measured).
- "What happens if two requests for the same never-cached tile arrive concurrently?" — worth reasoning through: the lock guards the in-memory pyramid dict, but does it also prevent two threads from both missing the disk cache and redundantly rebuilding + writing the same tile file? (Worth checking `tile_storage.py`'s exact lock scope — a good live-code-reading question.)

### 3.3 ML pipeline design (`model_decision_audit.md` is the primary reference)
- "Walk me through which of your modeling decisions are actually backed by an experiment, versus reasoned-but-untested, versus a bare assumption." — the scorecard in that file is exactly this answer, and the fact that it's disclosed rather than papered over is itself a strong signal to lead with.
- "Why leave-one-image-out CV instead of random tile-level splits?" — spatial autocorrelation / shared scan-batch leakage; note explicitly that this argument was never empirically demonstrated in-project (no side-by-side inflated-metrics run), which is a good honest caveat to include.
- "Why not just report accuracy?" — measured class imbalance (3.9–7.5% artifact tiles) makes "always predict clean" score >92%; this one *is* data-backed.
- "Your final deployed model isn't held out from anything — how do you know it's not just memorizing the 5 labeled images?" — good pressure-test question; honest answer is "we don't, beyond qualitative inspection on the 3 unlabeled images" — a real limitation, not a solved problem.
- "If you had to pick the next single experiment to run, which untested assumption would you test first, and why?" — per the audit, `pos_weight` damping and the `purity` filter are the cheapest (one-line config change, reuse the existing LOIO loop); tile size and architecture are the expensive ones.

### 3.4 LLM tool-calling / chat (`chat.py`, `steps_of_implementation_checkpoint3.md`)
- "Why function-calling instead of standing up a real MCP server?" — the checkpoint doc has a direct, well-reasoned answer: MCP is the right weight when multiple heterogeneous clients need the same tools; a single web app's own backend calling its own function doesn't need a second server + protocol layer for that.
- "How do you guarantee the chat path and the button path never diverge?" — both call `detection.run_detection` in-process; no HTTP round-trip, no second implementation.
- "The model occasionally emits a malformed pseudo-tool-call instead of a real one. How did you handle that, and why not fix it at the schema level?" — documented: retried once, schema tweaks tried and made it worse; genuinely a model-flakiness problem, not this code's bug — good example of knowing when *not* to keep patching.
- "Where does chat conversation state live, and what's wrong with that?" — leads straight into §2.4 above (unbounded in-memory growth, no multi-instance sharing).
- "The LLM can invent an `image_id` from free text. What stops it from causing a confusing error?" — `validate_image_id()` against the live HF repo listing, added specifically because Checkpoint 3 introduced this risk (a `<select>` dropdown couldn't have).

### 3.5 Storage & caching layers
- "Name every cache in this system, what it's keyed by, and what invalidates it." (HF Hub's own download cache — ETag-based, automatic; display-PNG cache — filename-only, no invalidation at all, see bug §4.2; DZI tile cache — per (image, level, col, row), also no invalidation; in-memory pyramid LRU — capped at 2 images; chat history — unbounded, see §2.4.) This is a strong single question that forces surfacing most of the caching gaps at once.
- "Why does the display-PNG cache check only `exists()` and not compare against the source file's revision?" — leads into bug §4.2.
- "Two independent caches exist for HF originals — the library's own and yours. Why not build one unified cache?" — answer: `hf_hub_download`'s own cache already solves "don't re-fetch a file we have" correctly; a second layer on top is redundant bookkeeping, not defense in depth.

### 3.6 Scaling & reliability, cross-cutting
- "If this had to handle 100x the traffic tomorrow, what's the first thing that breaks?" — good open-ended one; strong answer touches §2.2 (no rate limiting / no inference queue), §2.3 (per-instance caches), and the free-tier cold-start behavior together.
- "What's the blast radius if `HF_TOKEN` leaked?" — already assessed and *knowingly accepted* in `potential_issues.md`: the token is over-scoped (write access when only read is needed) — good to be honest that this was a discussed, not overlooked, tradeoff.
- "Everything in this system is stateless except three caches. Why no database at all?" — deliberate: the durable source of truth is the HF dataset repo; everything else is treated as cheap-to-regenerate scratch space, consistent with the free-tier, ephemeral-container deployment target.

---

## 4. Bugs found

Cross-checked against the live code as of this pass. Items unchanged from `potential_issues.md` are marked **(confirmed still open)**; items that have since been fixed are marked **(now fixed)**; everything under "New findings" wasn't in that file.

### Still open (confirmed against current code)

1. **Confidence-grid sentinel collision** — `DetectionCanvas.tsx:101`, `if (confidence === 0) continue`. A tile that was genuinely evaluated but scored a sigmoid output that underflows to exactly `0.0` in float32 would render as "not evaluated" instead of confident-clean. Low probability, real ambiguity. Needs a separate `evaluated: boolean` per tile in the response rather than overloading the confidence value.

2. **Display-PNG cache has no invalidation** — `hf_storage.py:get_display_image_path`, line 131: `if png_path.exists(): return png_path`. No comparison against the source file's HF revision/ETag. If an original is ever replaced on the Hub, the cached preview PNG goes stale indefinitely while `/detect` (via `hf_hub_download`'s own ETag-aware cache) picks up the new original immediately — silent mismatch between what the canvas shows and what the detection numbers describe.

3. **Unhandled exceptions on corrupted/truncated downloads** — confirmed still true in both `detection.py:37` (`tifffile.TiffFile(original_path)`) and `hf_storage.py:140` (same call, inside `get_display_image_path`). Neither is wrapped in a try/except; a truncated download or corrupted TIFF surfaces as a raw, unfriendly 500.

4. **`CORS_ORIGINS=*`** on both deployed backends — still the default; low-risk given Bearer-header (not cookie) auth, but not yet tightened to the real frontend origin.

5. **No rate limiting on `/detect`** or `/chat`'s detection tool — still true, expanded scope in this pass since `/chat` can trigger the same expensive path conversationally, not just via the button.

6. **`require_user` round-trips to Supabase on every gated request** — unchanged, see §2.1.

### Now fixed (were open in `potential_issues.md`, resolved since)

7. **No `image_id` allowlist check** — this was flagged as open in `potential_issues.md` (#4) but is now fixed: `hf_storage.validate_image_id()` checks against a live `list_images()` call before any HF filename is constructed, called first thing inside `get_original_path`. Added specifically because Checkpoint 3's chat feature lets an LLM invent an `image_id` from free text. Worth noting explicitly since the older doc still lists this as unresolved.

### New findings (not in prior docs)

8. **Chat conversation memory has no cap** — `chat.py:104`, `_conversations: dict[str, list[dict]] = {}`. No TTL, no LRU eviction, no max-size check — every distinct `conversation_id` lives forever until the process restarts. Contrasts with the pyramid cache elsewhere in the same codebase, which is deliberately LRU-capped at 2 images. On sustained traffic this is unbounded memory growth. Cheapest fix: cap total conversations with LRU eviction, or add a TTL sweep.

9. **Evaluation/serving threshold drift** — `src/train.py`'s `run_loio_cv` (line 135) hardcodes `y_pred = (y_prob >= 0.5)` for the IoU metric saved to `loio_cv_results.json`, but the actually-deployed `prob_threshold` was later tuned to 0.7 (`backend/app/config.py`). The saved IoU numbers no longer describe the model that's running in production. Either re-run the sweep's IoU at 0.7, or explicitly document why 0.5 is kept as a fixed reference point — right now it reads as drift, not a decision.

10. **Orphaned dead code: `backend/mcp_server/`** — the directory contains only compiled `__pycache__/*.cpython-313.pyc` files (`server`, `backend_client`, `login_routes`, `login_session`, `oauth_provider`, `session`) with **no corresponding `.py` source files present**, and `git log` shows this path was **never committed** to the repository. `steps_of_implementation_checkpoint3.md` explicitly and correctly documents the decision *not* to build a real MCP server (function-calling was the right-weight choice instead) — but these bytecode artifacts prove someone actually started implementing one, ran it at least once, then the source was deleted without ever landing in git. Not a functional bug (dead code, not reachable from the running app) but worth a `.gitignore`/cleanup pass so a future `git status` or `find` doesn't surface confusing phantom evidence of a feature that doesn't exist.

11. **Unhandled JSON parse in chat tool-call arguments** — `chat.py:194`, `args = json.loads(tool_call.function.arguments or "{}")`. If Groq ever returns a malformed (non-JSON) arguments string for a real tool call — a different failure mode than the already-handled "hallucinated pseudo-tool-call as plain text" case — this raises uncaught. It's still caught by `main.py`'s broad `except Exception` around the whole chat handler, so it doesn't crash the server, but it surfaces as a generic "Chat service temporarily unavailable" (502), which misdiagnoses a malformed-arguments problem as a Groq-service-availability problem.

12. **`torchvision` confirmed still a dead dependency** — listed in root `requirements.txt`, never imported anywhere in `src/`, `test/`, or the backend (re-confirmed via full-tree grep in this pass). Safe to drop; `SmallCNN` is defined from scratch.

### Accepted, not re-litigated
`potential_issues.md` also lists a weak test-account password, Supabase's password-strength minimum not being enforced, and the over-scoped `HF_TOKEN` — all explicitly discussed with you and knowingly kept as-is for this project's stakes. Not re-flagged here as "bugs" since they were conscious tradeoffs, not oversights; listed for completeness only.

---

## 5. Where to go next, if useful

Two things would raise the ceiling on interview-readiness fastest:
- Turn item 9 (threshold drift) into either a one-line fix or an explicit documented decision — it's the cheapest of the new findings and closes a real "your own numbers don't match your own deployment" gap.
- Pick one of §2.1–2.3 (auth round-trip cost, rate limiting, or cache scale-out) and actually sketch the fix, even without implementing it — being able to go one level deeper than "yes I know this is a gap" on at least one item is usually what separates a good system-design answer from a great one.
