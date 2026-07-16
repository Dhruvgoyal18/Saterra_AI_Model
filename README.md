# DAPI Whole-Slide-Image Artifact Detector

Detects artifact regions (mainly tissue folds) in whole-slide DAPI
microscopy scans using a small CNN tile classifier, served through a web
app with a detection UI and an LLM chat interface.

**Live app:** https://frontend-opal-iota-55.vercel.app
(access is invite-only — sign in with an existing account, or contact the
project owner for one)

## What's in this repo

- **Root (`src/`, `artifacts/`, `test/`)** — the training pipeline: data
  tiling, the CNN model, leave-one-image-out cross-validation, threshold
  selection, and inference. Chosen operating point: `prob_threshold=0.7`.
- **`PROJECT_INTERVIEW_WEB_APP/`** — the FastAPI + React app that serves
  the trained model (deployed at the link above). See
  [`PROJECT_INTERVIEW_WEB_APP/README.md`](PROJECT_INTERVIEW_WEB_APP/README.md)
  for setup, architecture, and the chat interface details.

## Directory structure

```
├── src/                    training pipeline (tiling, model, train, threshold_sweep, inference)
├── test/                   data_validation.py -- image/annotation alignment QC
├── artifacts/               model_final.pt, preprocessing_config.json, CV + threshold-sweep results
├── outputs/                 qualitative predictions on the unlabeled test images
├── DAPI_images/             local working copies of the 8 scans (gitignored, pulled from HF Hub)
├── artifact_annotations/    local working copies of the 5 label GeoJSONs (gitignored)
├── requirements.txt
├── understanding.md         detailed engineering notes / full design rationale
└── PROJECT_INTERVIEW_WEB_APP/   FastAPI + React app that serves the model (see its own README)
```

## Scores at threshold 0.7

Pooled out-of-fold metrics (`artifacts/threshold_sweep_results.json`), the
argmax-F1 operating point:

| Metric | Score |
|---|---|
| Precision | 0.243 |
| Recall | 0.578 |
| F1 | 0.342 |
| PR-AUC | 0.225 |


## Run the training pipeline locally

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # private HF dataset repo access
export HF_DATASET_REPO=Dhruvgoyal18/saterra-project
python -m src.train             # trains the model, writes artifacts/
python -m src.threshold_sweep   # picks the operating threshold
python -m src.run_test_inference  # runs it on the unlabeled test images
```

For running the web app itself, see the web app's own README.
