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
openapi.json        static OpenAPI schema for the HTTP API
DEPLOY.md           EC2 + Streamlit Community Cloud deployment steps
```

## API Base Path

The backend defaults to serving under `/api/v1/water-hyacinth`.

Examples:

- `GET /api/v1/water-hyacinth`
- `GET /api/v1/water-hyacinth/health`
- `POST /api/v1/water-hyacinth/predict`
- `POST /api/v1/water-hyacinth/predict/batch`
- `WS /api/v1/water-hyacinth/ws/live-detect`
- `GET /api/v1/water-hyacinth/docs`
- `GET /api/v1/water-hyacinth/openapi.json`

If you want a different prefix, set `API_PREFIX`.

## Mobile Detection Features

For a still photo:

- `POST /api/v1/water-hyacinth/predict`
- returns classification, confidences, coverage, image size, and YOLO `boxes`
- React Native can draw the returned `boxes` directly on top of the captured image

For live camera frames:

- connect to `WS /api/v1/water-hyacinth/ws/live-detect`
- send JSON like `{ "frame_id": "123", "image_base64": "..." }`
- the server responds with the same prediction fields plus `boxes`
- the mobile app draws the returned boxes per frame for a live overlay effect

## Backend Quickstart

From the repo root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 3000 --reload
```

Then open:

- `http://localhost:3000/api/v1/water-hyacinth/health`
- `http://localhost:3000/api/v1/water-hyacinth/docs`

## Frontend Quickstart

```bash
pip install -r requirements.txt
API_URL=http://localhost:3000/api/v1/water-hyacinth streamlit run streamlit_app.py
```

`API_URL` must be the backend base URL including `/api/v1/water-hyacinth`.

## Model Weights

Tracked runtime weights live in `models/` so an EC2 `git pull` brings them down with the repo.

## CI

GitHub Actions now runs:

- backend import sanity check
- pytest for `backend/tests`
- ruff linting

## OpenAPI

- Static schema file: `openapi.json`
- Live schema endpoint: `/api/v1/water-hyacinth/openapi.json`

OpenAPI covers the HTTP endpoints. The websocket live-detect route is documented in the READMEs because it is not part of standard OpenAPI path support.

## Deployment

See `DEPLOY.md` for copy-pasteable EC2 Docker deployment steps and Streamlit Community Cloud setup.
