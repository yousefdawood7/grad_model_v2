# Water Hyacinth Detection & Monitoring

Classification (EfficientNetV2-S) + detection (YOLOv8/YOLO11) pipeline for
water-hyacinth coverage estimation, extracted from the training notebook and
turned into a deployable FastAPI backend plus a decoupled Streamlit frontend.

## Layout

```text
api/                compatibility import that exposes the FastAPI app
backend/
  reference/        verbatim notebook-generated scripts used as the inference spec
  app/              FastAPI service implementation
  tests/            pytest coverage for the API
  requirements.txt  backend runtime dependencies
  Dockerfile        container image for the API
  README.md         backend runtime and weight-download configuration
streamlit_app.py    Streamlit frontend that talks to the API over HTTP
.streamlit/config.toml
requirements.txt    Streamlit frontend dependencies
models/             runtime cache for best_classifier.pth + best_detector.pt
.github/workflows/ci.yml
openapi.json        static OpenAPI schema for the API
DEPLOY.md           EC2 + Streamlit Community Cloud deployment steps
```

## API Base Path

The backend defaults to serving under `/api/v1/water-hyacinth`.

Examples:

- `GET /api/v1/water-hyacinth`
- `GET /api/v1/water-hyacinth/health`
- `POST /api/v1/water-hyacinth/predict`
- `POST /api/v1/water-hyacinth/predict/batch`
- `GET /api/v1/water-hyacinth/docs`
- `GET /api/v1/water-hyacinth/openapi.json`

If you want a different prefix, set `API_PREFIX`.

## Backend Quickstart

From the repo root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- `http://localhost:8000/api/v1/water-hyacinth/health`
- `http://localhost:8000/api/v1/water-hyacinth/docs`

## Frontend Quickstart

```bash
pip install -r requirements.txt
API_URL=http://localhost:8000/api/v1/water-hyacinth streamlit run streamlit_app.py
```

`API_URL` must be the backend base URL including `/api/v1/water-hyacinth`.

## Model Weights

Real weights are gitignored. The backend supports three runtime options:

1. Mount the files locally in `models/`.
2. Set `HF_MODEL_REPO` and let startup fetch from Hugging Face Hub.
3. Set `CLASSIFIER_WEIGHTS_URL` and `DETECTOR_WEIGHTS_URL` to download from direct asset URLs.

See `backend/README.md` for the full environment variable reference.

## CI

GitHub Actions now runs:

- backend import sanity check
- pytest for `backend/tests`
- ruff linting

## OpenAPI

- Static schema file: `openapi.json`
- Live schema endpoint: `/api/v1/water-hyacinth/openapi.json`

## Deployment

See `DEPLOY.md` for copy-pasteable EC2 Docker deployment steps and Streamlit Community Cloud setup.

