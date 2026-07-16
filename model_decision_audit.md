# Model Decision Audit: Is Every Choice Backed by Data?

Scope: the shipped Checkpoint-1 tile-classification pipeline (`src/config.py`,
`masks.py`, `tiling.py`, `dataset.py`, `model.py`, `train.py`, `metrics.py`,
`inference.py`) plus the web app's `prob_threshold` serving setting. The
U-Net exploration (`src/*_unet.py`, `losses.py`, `augment.py`) is out of
scope — it hasn't been trained or run in this repo yet, so there's no
evidence to audit for it.

## Method

Every hyperparameter, architecture choice, and design decision in scope was
checked against one question: **is there an actual number — a metric,
a comparison run, a measured ground-truth statistic — that justifies this
value over an alternative?** Three verdicts are used, and the difference
between them matters:

- **DATA-BACKED** — an experiment was actually run comparing this choice
  against alternative(s), with a metric deciding the winner.
- **REASONED, NOT TESTED** — grounded in sound domain/ML reasoning (or a
  real measurement of a *different* quantity), but the specific value chosen
  was never swept or compared against alternatives on this data.
- **ASSUMED / UNTESTED** — a default, convention, or time-budget choice with
  an explicit, named alternative that was never run.

A fourth category, **infra/non-accuracy**, covers decisions that don't
affect model quality at all (library choices, caching, batching for speed)
— these are noted for completeness but aren't scored.

**Headline answer: no, not every decision is data-backed — and the
project's own implementation log (`step_of_implementation_checkpoint1.md`)
says so explicitly and by name.** Of 14 accuracy-relevant decisions audited,
3 are genuinely data-backed, 1 is partially data-informed, 3 are reasoned
but untested in this project, and 7 are stated assumptions with a named,
untried alternative. That ratio is not a hidden gap — it's disclosed
up front in the project's own §7 ("What was **not** attempted, and why").

---

## Scorecard

| # | Decision | Shipped value | Verdict | Evidence |
|---|---|---|---|---|
| 1 | Tissue-mask estimator for unlabeled images | Otsu on log1p(block-mean), block=64 | **DATA-BACKED** | 4 methods compared by IoU vs. ground truth: 0.05 → 0.37–0.77 → brittle → 0.92–0.96 (shipped) |
| 2 | Imbalance-aware metrics over accuracy | Precision/Recall/F1/PR-AUC/IoU, no accuracy | **DATA-BACKED** | Measured artifact share 3.9–7.5% of tissue tiles; accuracy would reward a trivial "always clean" classifier at >92% |
| 3 | Decision threshold (`prob_threshold`) | **0.7** (this session's fix; was 0.5) | **DATA-BACKED** | 5-fold pooled out-of-fold sweep, `artifacts/threshold_sweep_results.json`: F1 0.269→0.342 |
| 4 | Per-image (not fixed) intensity normalization | 1st/99th percentile clip, per image | **DATA-INFORMED** | Real observed intensity drift (mean tissue ~230–405 across the 5 images) justifies *per-image*; the specific 1/99 cutoffs were never swept |
| 5 | Image-level (LOIO) CV split | Leave-one-image-out, 5 folds | **REASONED, NOT TESTED** | Standard leakage argument (shared scan batch, spatial autocorrelation); the claimed metric inflation from a random tile-level split was never actually run and measured in this project |
| 6 | Model architecture | Custom 3-block CNN, no pretraining | **REASONED, NOT TESTED** | Deployability/cost argument (CPU-only, single-channel); never run head-to-head against a pretrained backbone |
| 7 | Train final model on all 5 images (no holdout) | Combined-5 training for `model_final.pt` | **REASONED, NOT TESTED** | Data-scarcity argument (real held-out = 3 unlabeled images, evaluated qualitatively only) — a resource-allocation judgment, not a scored comparison |
| 8 | Tile size | 64px | **ASSUMED / UNTESTED** | 32px named as the explicit comparison the checkpoint plan asked for; never run ("not run in this pass, purely for CPU time budget") |
| 9 | Label rule `tau` (majority-vote artifact threshold) | 0.5 | **ASSUMED / UNTESTED** | Qualitative argument only; "hyperparameter search over tau... was skipped entirely" (project's own words) |
| 10 | Background-inclusion threshold | 0.5 | **ASSUMED / UNTESTED** | No sensitivity analysis recorded anywhere |
| 11 | `purity` boundary-tile filter | Implemented, default `None` (unused) | **ASSUMED / UNTESTED** | Alternative (`purity=0.95`) fully coded in `tiling.py:36,65-66`, never invoked or compared |
| 12 | Class-imbalance loss weighting | `pos_weight = n_neg/n_pos` (raw ratio) | **ASSUMED / UNTESTED** | Chosen as "simplest, most interpretable"; project's own next-steps list names `sqrt(pos_weight)` and focal loss as untried alternatives |
| 13 | Training budget | 3 epochs, batch 512, Adam lr=1e-3 | **ASSUMED / UNTESTED** | Explicit fixed ~15-min CPU budget; loss still decreasing at epoch 3 in every fold; no LR/epoch/batch sweep |
| 14 | Old `prob_threshold=0.5` default (pre-this-session) | 0.5 | **ASSUMED / UNTESTED** (now superseded — see #3) | Arbitrary sigmoid midpoint; never swept before this session |

**Infra / non-accuracy choices** (not scored — don't affect model quality):
GeoJSON classification by name-prefix (`masks.py:23-36`, an extensibility
choice), `PIL.ImageDraw` over `skimage`/`rasterio` for rasterization
(`masks.py:58-62`, forced by no network access to install alternatives),
`artifact_mask &= tissue_mask` clipping (`masks.py:94`, a correctness
invariant, not a tunable), `scipy.ndimage.label` for connected components
(dependency-availability choice), inference batch size 512 (throughput only).

---

## Detail: the data-backed decisions

### 1. Tissue-mask estimator (`src/masks.py:131-159`, `estimate_tissue_mask`)

This is the best-evidenced decision in the codebase — four candidate methods
were actually run and compared by IoU against real ground truth (the 5
labeled images' `Tissue` polygons):

| Method | IoU vs. ground truth |
|---|---|
| Plain Otsu on raw pixel intensity | 0.046–0.059 |
| Otsu on raw block-averaged intensity | 0.37–0.77 (unstable) |
| Histogram-valley heuristic | 0.0001–0.96 (too brittle to trust on unseen images) |
| **Otsu on log1p(block-mean), block=64 (shipped)** | **0.92–0.96** |

The failure mode of the naive approach was diagnosed, not guessed at: DAPI
images are a 3-population mixture (background / tissue / bright nuclei), and
plain Otsu latches onto the nuclei-vs-rest split instead of
background-vs-tissue. The log transform compresses the tissue population's
long tail, which is what fixes it. This is a textbook example of a decision
"backed by data or scores" — someone ran the alternatives, watched them fail,
diagnosed why, and picked the fix that measurably worked.

### 2. Metrics choice (`src/metrics.py`, `compute_classification_metrics`)

Accuracy is never reported, and the reason is a real number: artifact tiles
are 3.9–7.5% of the tile population per image (measured directly, `tiling.py`
output logged in `step_of_implementation_checkpoint1.md` §2). A model that
predicts "clean" unconditionally would score >92% accuracy while detecting
nothing. Precision/recall/F1/PR-AUC (tile-level) plus IoU (stitched-mask
level) were chosen instead specifically because the imbalance was measured,
not assumed.

### 3. Decision threshold, `prob_threshold` (this session's work)

Previously 0.5 — an arbitrary sigmoid midpoint, never checked. This session:
reran the 5-fold LOIO CV, pooled 81,756 out-of-fold `(y_true, y_prob)` pairs
(honest — no fold's held-out image was ever seen by the model that scored
it), and swept thresholds 0.05→0.95. Argmax-F1 lands at **0.7**
(precision 0.163→0.243, recall 0.762→0.578, F1 0.269→0.342 vs. the old
default). Verified qualitatively too: on the 3 unlabeled images, flagged
artifact area dropped 28–56% and region counts dropped 30–40% at 0.7 vs 0.5.
Full sweep table: `artifacts/threshold_sweep_results.json`. Applied in
`PROJECT_INTERVIEW_WEB_APP/backend/app/config.py:62`.

---

## Detail: reasoned-but-untested decisions

These have real, defensible arguments behind them — not arbitrary — but the
argument is a priori reasoning or a leakage/cost tradeoff, not a scored
head-to-head run in this project.

### 5. Leave-one-image-out CV split (`src/train.py:104-172`)

The reasoning is sound and standard: tiles from the same image share a scan
batch, staining run, and illumination, and neighboring tiles are spatially
autocorrelated, so a random tile-level split would train and validate on
near-duplicates and inflate every metric. This is correct ML methodology.
But note precisely what was *not* done: nobody in this project actually ran
the random-tile-split alternative and showed the inflated numbers side by
side with the LOIO numbers. The conclusion is almost certainly right, but
it's asserted, not measured here.

### 6. Model architecture, `SmallCNN` (`src/model.py:15-37`)

3 conv blocks (16→32→64 channels, Conv-BatchNorm-ReLU-MaxPool, final block
using `AdaptiveAvgPool2d(1)`), then `Linear(64,1)`. Justified by a
deployability argument (CPU-only Docker image, no CUDA, free-tier hosting)
and a scale argument (64×64 single-channel tiles don't obviously need
transfer learning). Nobody actually trained a pretrained ResNet-18 or
EfficientNet-B0 variant on this data to see whether it does better — the
comparison is named in `model.py`'s own docstring as "documented alternative
... for when more capacity is wanted," explicitly future work.

### 7. Final model trained on all 5 images combined (`src/train.py:175-201`)

`train_final_model()` deliberately doesn't hold out any of the 5 labeled
images — LOIO CV (above) is treated as purely an *evaluation* of expected
generalization, and the actually-shipped model then uses every labeled pixel
available, on the reasoning that the real held-out set is the 3 unlabeled
images (evaluated qualitatively only, never scored). This is a defensible
resource-allocation call given only 5 labeled images, not something that
needs (or could have) a comparative score — there's no ground truth to hold
that combined model against that isn't already "used."

---

## Detail: assumed / untested decisions

These are stated defaults with a named, uninvestigated alternative — the
project's own §7 lists most of these explicitly as "what to try next."

- **Tile size = 64px** (`config.py:22`) — 32px was the checkpoint plan's own
  explicitly requested comparison point and was never run, purely for CPU
  training-time budget (~4x the tiles, ~4x the training time).
- **`TAU = 0.5`** (`config.py:23`, the majority-vote artifact-label
  threshold) — reasoned qualitatively (low tau inflates the positive class
  with fuzzy boundary tiles) but never swept against LOIO metrics. The
  project's own document states plainly: "hyperparameter search over
  tau/tile size/loss weighting was skipped entirely."
- **`BACKGROUND_THRESHOLD = 0.5`** (`config.py:24`) — no sensitivity check
  recorded at all.
- **`purity` filter** (`tiling.py:36,65-66`) — a "pure tiles only" variant
  is fully implemented (drop ambiguous boundary tiles between
  `1-purity` and `purity`) but defaults to `None` and was never turned on
  for any reported run.
- **`pos_weight = n_neg/n_pos`** (`train.py:66-68`, the raw class-imbalance
  ratio fed to `BCEWithLogitsLoss`) — chosen as "the simplest, most directly
  interpretable option" over oversampling or focal loss. The project's own
  next-steps list names `sqrt(pos_weight)` as an untried, cheaper
  alternative — this is precisely the lever this project's earlier
  precision-focused work (this same session, prior to this audit) flagged
  as still open.
- **Training budget**: 3 epochs, batch 512, Adam @ lr=1e-3 (`config.py:37-39`)
  — an explicit fixed ~15-minute CPU-only budget, not a convergence point.
  Every fold's training loss was still decreasing at epoch 3. No epoch
  count, batch size, or learning rate was ever varied and compared.

---

## What this means in practice

The project is unusually transparent about this already — `step_of_implementation_checkpoint1.md` §7 ("Known limitations and what to try
next") and its closing note ("What was **not** attempted, and why:
hyperparameter search... was skipped entirely... to avoid overfitting the
*design choices* to these 5 images") pre-empt most of this audit's findings.
The explicit reasoning given there — with only 5 labeled images and no 6th
to validate against, extensive tuning risks overfitting hyperparameters to
this specific small sample even without ever fitting a single model to all
5 — is itself a legitimate, data-scarcity-aware argument for *why* certain
things weren't swept. It's a considered tradeoff (methodology over
leaderboard-chasing on a checkpoint explicitly graded on methodology), not
negligence.

The practical takeaway: of the untested assumptions, `pos_weight` damping
and the `purity` filter are the cheapest to actually test next (one-line
config changes, reuse the existing LOIO CV loop) — and `prob_threshold` has
now moved from "assumed" to "data-backed" this session. Tile size and
architecture changes are the expensive ones (full retraining runs / new
model code) and remain the least-tested part of the pipeline.
