# Repository Structure

```
Project_Interview_Model_Building/
├── .env
├── .gitignore
├── requirements.txt
├── step_of_implementation_checkpoint1.md
│
├── src/                              # Core ML model code (training/inference)
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── inference.py
│   ├── masks.py
│   ├── metrics.py
│   ├── model.py
│   ├── run_test_inference.py
│   ├── tiling.py
│   └── train.py
│
├── test/
│   └── data_validation.py
│
├── detect_response.json
│
└── PROJECT_INTERVIEW_WEB_APP/         # Full-stack web app wrapping the model
    ├── README.md
    ├── steps_of_implementation_checkpoint2.md
    ├── update_pipeline.md
    │
    ├── backend/                       # FastAPI backend
    │   ├── .env.example
    │   ├── Dockerfile
    │   ├── README.md
    │   ├── requirements.txt
    │   └── app/
    │       ├── __init__.py
    │       ├── config.py
    │       ├── main.py
    │       ├── schemas.py
    │       ├── ml/
    │       │   ├── __init__.py
    │       │   ├── display_prep.py
    │       │   ├── inference.py
    │       │   ├── model.py
    │       │   ├── preprocessing.py
    │       │   ├── tissue.py
    │       │   └── weights/
    │       │       ├── model_final.pt
    │       │       └── preprocessing_config.json
    │       └── storage/
    │           ├── __init__.py
    │           └── hf_storage.py
    │
    ├── frontend/                      # Vite + React + TS frontend
    │   ├── .env.example / .gitignore / .oxlintrc.json
    │   ├── README.md
    │   ├── index.html
    │   ├── package.json / package-lock.json
    │   ├── tsconfig*.json
    │   ├── vite.config.ts
    │   ├── public/
    │   │   ├── favicon.svg
    │   │   └── icons.svg
    │   └── src/
    │       ├── App.tsx / App.css
    │       ├── main.tsx
    │       ├── DetectionCanvas.tsx
    │       ├── SummaryPanel.tsx
    │       ├── api.ts
    │       ├── colormap.ts
    │       ├── types.ts
    │       ├── index.css
    │       └── assets/ (hero.png, react.svg, vite.svg)
    │
    └── scripts/
        ├── prepare_display_images.py
        └── upload_to_hf.py
```

Note: `backend/app/storage.py` was deleted and replaced by a `backend/app/storage/` package
(`__init__.py` + `hf_storage.py`) per the currently staged changes on the `model_dep` branch.
