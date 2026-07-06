# Checkpoint 1: Artifact Detection — Implementation Log

This documents exactly what was built, every decision made and why, the actual
numbers produced by running it against the real data in `DAPI_images/` and
`artifact_annotations/`, what's weak about it, and what to do next. Everything
below was actually run — no numbers are invented or extrapolated.

Code lives in `src/`. Data validation (the x/y-swap and pyramid-level checks
that had to be trusted before any of this was worth building) lives in
`test/data_validation.py` and is a separate, earlier piece of work — see that
file's docstring for those checks specifically.

---

## 0. What exists before this checkpoint

- `DAPI_images/Image_1..8.ome.tiff` — 8 whole-slide DAPI scans, single-channel
  `uint16`, roughly 8100-9500 px per side.
- `artifact_annotations/artifact_annotation_image_1..5.geojson` — GeoJSON
  annotations for **only the first 5 images**. Each contains exactly one
  `"Tissue"` polygon and a handful of `"Artifact - folds"` polygons.
  Images 6, 7, 8 have **no annotations** — they are the true, unlabeled test set.
- `test/data_validation.py` — already validated, before any of this was
  written, that: pyramid level 0 is the correct level to load, there is no
  row/col (x/y) axis swap, and no y-flip, for all 5 labeled images (bounding-box
  check + an intensity-contrast check + a visual overlay; see that file).

Geometry check done before writing the mask-rasterization code: every one of
the 31 features across the 5 GeoJSON files is a single-ring `"Polygon"` — no
holes, no `MultiPolygon`. The code still handles both defensively (see §1),
but this is what's actually in the data.

---

## 1. Step 1 — GeoJSON → pixel masks (`src/masks.py`)

**What it does:** for one image, builds four full-resolution boolean masks:
`tissue`, `artifact` (union of every `"Artifact - *"` class), `clean_tissue`
(`tissue & ~artifact`), `background` (`~tissue`).

**Decisions and why:**

- **Feature classification is by name prefix, not hard-coded.** A feature is
  "artifact" if `properties.classification.name` starts with `"Artifact"`,
  "tissue" if it equals `"Tissue"` exactly. This was a deliberate choice over
  hard-coding `"Artifact - folds"` specifically, so a new artifact subtype
  (e.g. `"Artifact - bubbles"`) is picked up automatically. Every artifact
  subtype is collapsed into one positive class, per the spec.
- **Rasterization uses `PIL.ImageDraw`, not `skimage.draw.polygon` or
  `rasterio.features.rasterize`.** `rasterio` and `scikit-image` are not
  installed in this project's venv and this machine has no network access to
  install them (`pip install scikit-image` failed with a DNS resolution
  error — verified, not assumed). `PIL.ImageDraw.polygon()` takes `(x, y)`
  points directly, which is what GeoJSON already provides — no manual
  row/col swap, and so no chance of reintroducing the exact bug
  `test/data_validation.py` was written to rule out.
- **Holes are handled by drawing the exterior ring `fill=255` then each hole
  ring `fill=0`** on top. Correct for simple, non-self-intersecting polygons
  (which is what QuPath/GeoJSON exports of this kind produce). Not present in
  the actual 5 files, but the code doesn't assume it away.
- **`artifact_mask &= tissue_mask`** — artifacts are clipped to be a subset of
  tissue defensively, in case an artifact polygon's boundary strays slightly
  outside the tissue polygon's boundary (would otherwise silently create
  "artifact but not tissue" pixels that the tiling step's tau calculation,
  which divides by tissue pixels, would then double-count).

**Measured on the real data** (artifact area as a fraction of tissue area,
computed by direct pixel counts before any tiling):

| Image | Tissue px | Artifact px | Artifact / Tissue |
|---|---|---|---|
| 1 | 70,553,666 | 3,086,678 | 4.37% |
| 2 | 69,448,017 | 3,522,897 | 5.07% |
| 3 | 71,461,116 | 4,163,166 | 5.83% |
| 4 | 70,525,010 | 2,770,307 | 3.93% |
| 5 | 52,910,868 | 3,957,580 | 7.48% |

This is the imbalance the rest of the pipeline has to deal with: artifacts are
a small minority class at the pixel level, before tiling even makes it worse.

### `estimate_tissue_mask` — a second function, for the *unlabeled* images only

Images 6-8 have no `Tissue` polygon, so `detect_artifacts` (§6) needs some way
to know where tissue is when filtering background tiles. This function is
**not** part of the labeled-data pipeline; it exists purely so the 3 unlabeled
images have *something* to restrict inference to, and its output is validated
against the 5 labeled images' real ground truth before being trusted.

**What was tried and failed, in order (this is worth recording because the
failure mode is non-obvious):**

1. **Plain Otsu threshold on raw pixel intensity** → IoU vs. ground truth
   tissue mask: **0.046-0.059** across the 5 labeled images. Essentially
   useless. Cause: DAPI images are a 3-population mixture — black slide
   background, mid-gray tissue, and a long tail of very bright individual
   nuclei. Global Otsu on raw pixel values latches onto the
   nuclei-vs-rest split (the highest-variance split in the histogram),
   not the background-vs-tissue split that's actually wanted.
2. **Otsu on raw block-averaged intensity** (average non-overlapping 64x64
   blocks first, to smooth out nuclei speckle, then Otsu on the block means)
   → IoU **0.37-0.77**, wildly unstable across images and highly sensitive to
   block size.
3. **A histogram-valley heuristic** (smooth the block-mean histogram, take
   the first local minimum after the first peak) → IoU as high as 0.96 on
   some images but as low as 0.0001-0.5 on others depending on bin count and
   smoothing window — far too brittle to trust on unseen images where there's
   no ground truth to tune against.
4. **Otsu on `log1p` of the block-averaged intensity** (block size = tile
   size = 64) → **the one that shipped.** The log transform compresses the
   long right tail from the tissue population, which is what was skewing
   plain Otsu's variance-maximizing split away from the true background/tissue
   boundary. Result, no hyperparameter tuning beyond the one block-size choice:

   | Image | IoU (Otsu-log-block estimate vs. ground truth) | Estimated tissue frac | True tissue frac |
   |---|---|---|---|
   | 1 | 0.9594 | 0.761 | 0.787 |
   | 2 | 0.9524 | 0.848 | 0.863 |
   | 3 | 0.9320 | 0.814 | 0.841 |
   | 4 | 0.9432 | 0.838 | 0.886 |
   | 5 | 0.9243 | 0.747 | 0.788 |

This is good enough to gate background tiles out of inference on the
unlabeled images, but it is explicitly **not** a substitute for a real
annotation, and the code's docstring says so. If tissue segmentation quality
ever becomes the bottleneck, this is the first thing to replace (e.g. with a
trained tissue/background segmenter), not the artifact detector itself.

---

## 2. Step 2 — Tiling and labeling (`src/tiling.py`)

Non-overlapping `tile_size x tile_size` windows (default 64px) slid across the
full image grid. Two independent decisions per tile:

**Decision 1 — include or exclude:** a tile is dropped entirely if less than
`background_threshold` (default 0.5) of its pixels are tissue. This is a
whole-tile fraction, computed against the *full* tile area.

**Decision 2 — clean or artifact:** for tiles that survive the background
filter, `artifact_frac = artifact_px / tissue_px` is computed using only the
tissue pixels in the tile (not all pixels) as the denominator — so a tile that
straddles the tissue/background edge isn't penalized in its artifact score
just for touching background. The tile is labeled `artifact` if
`artifact_frac >= tau` (default tau=0.5, i.e. majority vote).

**Why tau=0.5 as the baseline, and what the alternatives cost:**
- `tau -> 0` ("any artifact pixel makes it an artifact tile") inflates the
  positive class and teaches the model on fuzzy, mostly-clean tiles that
  happen to touch an artifact boundary — this pushes precision down further
  than it already is.
- `tau = 0.5` (shipped default) is the direct, defensible majority-vote rule.
- A **"pure tiles only"** variant is implemented (`purity` parameter in
  `generate_tile_records`) — set `purity=0.95` and any tile with
  `artifact_frac` strictly between 0.05 and 0.95 is dropped instead of forced
  into a class. This trades away boundary-adjacent training data for a
  cleaner label signal. **Not used for the results below** (tau=0.5, no
  purity filter, so every tile that passes the background filter gets a
  label) — this was a deliberate choice to keep the maximum amount of
  training data given only 5 images, but it's a one-line change
  (`generate_tile_records(..., purity=0.95)`) to try the alternative, and is
  worth trying if precision needs to improve (see §7).

**Actual tile counts at tile_size=64, tau=0.5, background_threshold=0.5** (this
is the real imbalance the model has to learn from, logged directly from the
training run):

| Image | Tiles kept | Clean | Artifact | Artifact share |
|---|---|---|---|---|
| 1 | 17,212 | 16,466 | 746 | 4.3% |
| 2 | 16,968 | 16,105 | 863 | 5.1% |
| 3 | 17,443 | 16,425 | 1,018 | 5.8% |
| 4 | 17,214 | 16,535 | 679 | 3.9% |
| 5 | 12,919 | 11,951 | 968 | 7.5% |

Tile-level imbalance closely tracks the pixel-level imbalance from §1 (as
expected, since tau=0.5 is a majority-vote rule on the same underlying
areas) — roughly 4-8% positive class, ~13-19x more clean tiles than artifact
tiles depending on the image.

**Tile size: 64px, not 32px or 16px.** Per the checkpoint plan, 64px gives
~17k tiles/image (matches the table above almost exactly), runs fast enough
for iterated experimentation on a CPU-only machine, and folds are large
enough features to be visible at that scale. 32px (~4x more tiles) or 16px
(~16x more tiles) were not run in this pass, purely for CPU training-time
budget reasons — see §7 for what a 32px comparison would cost and what it
might buy.

---

## 3. Step 3 — The train/validation split: Leave-One-Image-Out CV

**This is the part graded on methodology, not accuracy, and it's the one
place a wrong answer would be a real mistake, so the reasoning is spelled out
in full.**

A random split of *tiles* (ignoring which image they came from) would put
tiles from the same image on both sides of the split. Tiles from the same
image are not independent samples: they share the same scan batch, the same
staining run, the same illumination/flat-field conditions, and neighboring
tiles are spatially autocorrelated (a tile 64px from a training tile looks
almost identical to it). Training and validating on near-duplicates like that
inflates every metric — the model isn't being tested on anything it hasn't
effectively already seen.

With only 5 labeled images, the leak-free option is to split at the **image**
level: **leave-one-image-out (LOIO) cross-validation** — train on 4 images,
validate on the 1 held out, rotate through all 5, report **mean ± std**
across folds rather than a single number. The std across folds is itself
information: it's an honest estimate of how much the model's performance
would vary on a 6th labeled image, which a single train/val split could never
tell you.

The 3 unlabeled images (6, 7, 8) are **not** part of this CV — they have no
ground truth, so there's no metric to compute for them. They're evaluated
qualitatively only (§6), and that's stated plainly rather than inventing
numbers for them.

---

## 4. Step 4 — Model, training, imbalance handling, metrics

### Model (`src/model.py`)

A small CNN trained from scratch: 3 conv blocks (16→32→64 channels, each
`Conv-BatchNorm-ReLU-MaxPool`, except the last which uses
`AdaptiveAvgPool2d(1)` instead of a fixed-size pool so the same architecture
works at any tile size), then one linear layer to a single logit.

**Why not a pretrained ResNet-18/EfficientNet-B0:** both need the
single-channel DAPI tile replicated to 3 channels (or the first conv layer
edited), add real parameter count and CPU inference cost for no expected gain
at 64px tile resolution, and the checkpoint plan explicitly frames Checkpoint
1 as methodology-over-accuracy. A small custom CNN trains in minutes on CPU
and is entirely sufficient to establish the pipeline and the interface
contract (§6) that Checkpoints 2/3 depend on. Swapping in a bigger backbone
later is a model.py change only — nothing else in the pipeline (masks,
tiling, LOIO split, `detect_artifacts` contract) would need to change.

### Normalization (`src/dataset.py`)

**Per-image 1st/99th percentile clipping, scaled to [0, 1]** — not a fixed
range like `/65535`. Absolute DAPI intensity drifts between slides/scan runs
(visible even across the 5 labeled images: mean tissue intensity ranged from
~230 to ~405 in the intensity-contrast checks in `data_validation.py`). A
fixed normalization range would bake in whatever brightness this particular
batch of 5 slides happened to have. Per-image percentile scaling is
specifically what's expected to let a model trained on images 1-5 transfer to
images 6-8, which come from a different point in the same scan batch and are
not guaranteed to share an intensity distribution.

### Imbalance handling

**Class-weighted `BCEWithLogitsLoss`**, with `pos_weight = n_clean / n_artifact`
computed fresh per training fold from that fold's actual tile counts (so a
held-out-image-5 fold, which has less training data, gets a different weight
than a held-out-image-1 fold). This was chosen over oversampling or focal
loss as the simplest, most directly interpretable option — it's a single
scalar with a clear meaning ("penalize missing an artifact tile N times as
much as a false positive"), rather than a shape hyperparameter (focal loss's
gamma) that would need its own tuning pass this checkpoint's time budget
doesn't have room for. See §7 for what this choice costs in practice.

### Training budget

3 epochs, batch size 512, Adam @ lr=1e-3, CPU-only (no CUDA on this machine —
verified: `torch.cuda.is_available()` is `False`). This was a deliberate,
explicit cap, not a result of running out of time unexpectedly: the full
5-fold LOIO CV plus final-model training took **~15 minutes wall-clock** at
this budget. The checkpoint plan is explicit that accuracy is not scored here
and the point is to have a working, honestly-evaluated pipeline — so training
compute was capped rather than spent chasing a better number.

### Metrics (`src/metrics.py`)

**Accuracy is not reported anywhere in this pipeline, on purpose.** With
artifact tiles at 4-8% of the tile population (§2), a model that predicts
"clean" unconditionally scores >92% accuracy while detecting nothing.
Reported instead, per fold: precision, recall, F1, PR-AUC (all tile-level),
plus **IoU** between the stitched full-resolution predicted-artifact mask and
the ground-truth artifact mask — the metric that matches how an annotator
actually thinks (outlining a region), rather than a per-tile score.

### Results — actual LOIO CV output

```
Fold (held out)   precision   recall   f1      pr_auc   iou
Image_1             0.245     0.776    0.373   0.631    0.223
Image_2             0.124     0.787    0.215   0.555    0.119
Image_3             0.187     0.921    0.311   0.660    0.184
Image_4             0.151     0.536    0.236   0.324    0.131
Image_5             0.110     0.870    0.195   0.286    0.107

Aggregate (mean +/- std across 5 folds):
  precision: 0.163 +/- 0.049
  recall:    0.778 +/- 0.132
  f1:        0.266 +/- 0.066
  pr_auc:    0.491 +/- 0.156
  iou:       0.153 +/- 0.044
```

**Reading these honestly:** recall is consistently high (0.54-0.92) and
precision is consistently low (0.11-0.25) — the class-weighted loss is doing
exactly what it was told to do (penalize missed artifacts heavily), and the
result is a model that over-triggers: it rarely misses a real fold, but flags
roughly 4-9x more tiles as "artifact" than are actually artifacts. PR-AUC
(0.29-0.66) and IoU (0.11-0.22) confirm this isn't just a threshold-tuning
issue — the underlying ranking of tiles by predicted probability is only
moderately informative, not strongly separating clean from artifact.
Fold-to-fold variance is real (PR-AUC std of 0.156 on a mean of 0.491 is a
large spread for only 5 folds) — with 5 images, one image being unusually
hard (Image_4 and Image_5 both pull PR-AUC down) swings the aggregate
noticeably. This is exactly the kind of instability LOIO CV with std
reporting is meant to surface, rather than hide behind a single train/val
split's lucky or unlucky number.

**Training loss curves** (all folds, 3 epochs) all decrease monotonically
(e.g. fold held-out=Image_1: 0.961 → 0.889 → 0.854), confirming the model is
learning something, not stuck — the ceiling here is model capacity / training
budget / label quality, not a broken training loop.

### Final model

After LOIO CV (which is purely an *evaluation* of expected generalization),
a separate final model is trained on **all 5 labeled images combined** —
this is the model actually saved and shipped (`artifacts/model_final.pt`).
None of the 5 images is "held out" for this final model, since the real
held-out set is the 3 unlabeled images (evaluated qualitatively, §6), and
none of these 5 images should be wasted once training data is this scarce.
Final training loss: 0.912 → 0.831 → 0.803 (3 epochs, same config as each
CV fold).

Saved artifacts:
- `artifacts/model_final.pt` — model weights (~103KB).
- `artifacts/preprocessing_config.json` — tile_size, tau, background_threshold,
  intensity percentiles, model name. This is what a deployment environment
  (Checkpoint 2/3) needs to reproduce preprocessing exactly.
- `artifacts/loio_cv_results.json` — full per-fold + aggregate metrics.

---

## 5. The inference interface — `detect_artifacts` (`src/inference.py`)

Per the checkpoint plan, this is designed as the single seam Checkpoints 2/3
are meant to bolt onto — everything upstream of it (masks, tiling, LOIO CV) is
throwaway scaffolding once the weights exist; this function is not.

```python
detect_artifacts(image, tissue_mask=None, ...) -> {
    "mask": np.ndarray[bool],           # (H, W) full-resolution stitched prediction
    "per_tile_confidence": np.ndarray,  # (n_rows, n_cols) float32, sigmoid prob per tile
    "summary": {
        "total_artifact_area_px": int,
        "total_artifact_area_frac_of_tissue": float,
        "num_connected_regions": int,
        "largest_region_area_px": int,
        "n_tiles_evaluated": int,
        "n_tiles_total": int,
    },
}
```

**Design notes:**
- The model is loaded once and cached in a module-level dict keyed by weights
  path (`_MODEL_CACHE`) — repeated calls (e.g. from a web server handling
  multiple requests) don't reload weights from disk each time.
- Tiles are batched (default 512 at a time) before a forward pass, rather than
  one tile per forward call — meaningfully faster on CPU.
- `tissue_mask` is optional: pass the ground-truth mask when available (the 5
  labeled images), or `masks.estimate_tissue_mask(image)` when not (the 3
  unlabeled images, see §1). If omitted entirely, every tile in the grid is
  evaluated and the area fraction falls back to a fraction of total image
  area — background tiles will mostly self-classify as clean (background has
  no artifact signal), but this wastes compute over the always-black
  slide-background region.
- Connected-region counting uses `scipy.ndimage.label` (scipy is already a
  project dependency) rather than `scikit-image`, which isn't installed and
  couldn't be installed on this machine (no network access).
- `num_connected_regions` and `largest_region_area_px` are exactly the
  Checkpoint-2-required summary fields, produced here so Checkpoint 2 can be
  "call this function and render its output" rather than a rewrite.

---

## 6. Inference on the 3 unlabeled images (qualitative only)

Run via `python -m src.run_test_inference`. Since images 6-8 have no
annotation, **there is no metric to report for them** — this is stated
explicitly rather than inventing a number, per the checkpoint plan. Each
image gets a saved overlay PNG (`outputs/test_predictions/`) and a summary
JSON.

| Image | Est. tissue frac | Predicted artifact area (frac of est. tissue) | Connected regions | Largest region (px) |
|---|---|---|---|---|
| 6 | 0.879 | 28.3% | 1,399 | 6,250,496 |
| 7 | 0.762 | 39.4% | 708 | 12,132,352 |
| 8 | 0.821 | 41.1% | 338 | 18,636,800 |

**Visual read of the overlays** (this is the actual judgment call, not just
the numbers):
- **Image_8** is the most convincing result: there's an obvious, large,
  visually distinct fold structure in the raw DAPI channel itself (a bright
  swirled band cutting through the tissue), and the model's prediction
  mask tracks it closely and coherently, while the rest of the tissue (upper
  left) stays mostly clean with only scattered, isolated single-tile false
  positives.
- **Image_7** is a partial success: it catches a large contiguous blob
  (top-left) and correctly traces a thin, linear crease running diagonally
  through the tissue (visible as a faint bright line in the raw image) — but
  also has a substantial band of scattered false-positive tiles across the
  rest of the tissue.
- **Image_6** is the weakest of the three: one plausible coherent artifact
  region (bottom-right), but a large fraction of the predicted area is
  scattered, isolated single/few-tile false positives spread fairly
  uniformly across otherwise-clean tissue — consistent with the low
  precision (0.11-0.25) measured in LOIO CV.

**Net read:** the model has clearly learned *something* about what large,
coherent fold artifacts look like (it's not random — Image_8 in particular is
a genuinely good catch), but it substantially over-triggers on small,
scattered regions, which matches the LOIO CV numbers (high recall, low
precision) exactly rather than contradicting them. This consistency between
the quantitative CV result and the qualitative unlabeled-image behavior is
itself a useful sanity check that the two evaluations aren't telling
different stories.

---

## 7. Known limitations and what to try next

Ranked by expected leverage, given more time:

1. **Precision is the actual bottleneck, not recall.** The class-weighted
   loss was tuned to avoid missing artifacts, and it worked almost too well.
   Concrete next steps, cheapest first:
   - Raise the decision threshold above 0.5 (a free change — `prob_threshold`
     is already a parameter of `detect_artifacts`) and re-check the PR curve
     to pick an operating point with better precision/recall balance instead
     of the loss-only-implied 0.5 cutoff.
   - Try `purity=0.95` in `generate_tile_records` (the "pure tiles only"
     variant already implemented but not used, §2) — training only on
     unambiguous tiles may reduce noisy near-boundary label supervision.
   - Try a smaller `pos_weight` (e.g. `sqrt(n_clean/n_artifact)` instead of
     the raw ratio) — the current weighting may be over-correcting.
2. **The scattered single-tile false positives look like they could be
   partly driven by scan-tiling/flat-field brightness variation** (visible as
   a faint rectangular patchwork in the raw DAPI images at low zoom — a known
   property of stitched whole-slide scans) rather than genuine fold texture.
   Worth checking directly: do false-positive tiles cluster at those
   brightness-patch boundaries? If so, a per-tile local-contrast
   normalization (rather than only a global per-image percentile clip) might
   help precision more than any loss-function change.
3. **32px tiles**, as a documented comparison to the 64px baseline (the plan
   explicitly asks for this tradeoff to be shown): ~4x more tiles, finer
   localization of fold boundaries, ~4x training time. Not run here purely
   for CPU time budget; the code takes it as a one-line config change
   (`config.TILE_SIZE = 32`) with no other code changes required.
4. **More epochs / a validation-loss-based early stop.** 3 epochs was a fixed
   budget, not a convergence point — loss was still decreasing at epoch 3 in
   every fold. Whether more epochs actually helps *precision* (vs. just
   fitting the same recall-heavy objective harder) is unknown and worth
   checking before assuming it would.
5. **Segmentation instead of classification (U-Net)** — the plan's other
   explicitly-requested comparison. Would very likely give sharper artifact
   boundaries and probably a better IoU, at the cost of needing much heavier
   patch-based augmentation given only 5 labeled images, more memory, and a
   deployment footprint that doesn't fit a free CPU-tier host the way a small
   tile classifier does (this is the Checkpoint-2/3-facing argument the plan
   asks for — tile classification was chosen deliberately for
   deployability, not because it's the best possible model). Worth
   revisiting only if boundary precision becomes the actual bottleneck and
   GPU inference budget exists.
6. **A real tissue segmenter for the unlabeled images**, replacing
   `estimate_tissue_mask`'s Otsu-on-log-block-mean heuristic (§1), if tissue
   masking quality (not artifact detection) turns out to be the limiting
   factor for the 3 unlabeled images specifically.

What was **not** attempted, and why: hyperparameter search over tau/tile
size/loss weighting was skipped entirely — with only 5 labeled images and no
6th to check against, extensive tuning against LOIO CV numbers risks
overfitting the *design choices* to these 5 images even though no single
model was fit to all 5. The plan's own framing (Checkpoint 1 graded on
methodology, not accuracy) was taken at face value: get one honestly-evaluated,
working, documented pipeline with the right interface, rather than iterating
on this checkpoint's score at the expense of Checkpoints 2/3's time budget.

---

## 8. Repo structure

```
src/
  config.py              # every path + hyperparameter, one place
  masks.py               # GeoJSON -> tissue/artifact/clean/background masks
                          # + estimate_tissue_mask (unlabeled-image fallback)
  tiling.py               # TileRecord, generate_tile_records (background filter + tau labeling)
  dataset.py               # image loading, per-image percentile normalization, ArtifactTileDataset
  model.py                 # SmallCNN
  metrics.py                # precision/recall/F1/PR-AUC, IoU
  train.py                  # LOIO CV loop + final-model training (run: python -m src.train)
  inference.py               # detect_artifacts() -- the Checkpoint 2/3 contract
  run_test_inference.py       # applies detect_artifacts to images 6-8, saves overlays + summary

artifacts/
  model_final.pt              # trained weights (all 5 labeled images)
  preprocessing_config.json    # tile_size, tau, background_threshold, intensity percentiles
  loio_cv_results.json          # full per-fold + aggregate CV metrics

outputs/test_predictions/
  prediction_overlay_image_{6,7,8}.png
  test_predictions_summary.json

test/data_validation.py    # pre-existing: pyramid-level, x/y-swap, y-flip checks (not part of this checkpoint's build, but load-bearing for it)
```

Reproduce end-to-end: `python -m src.train` (~15 min on CPU, writes
`artifacts/`), then `python -m src.run_test_inference` (writes
`outputs/test_predictions/`).
