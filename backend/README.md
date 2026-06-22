# Backend Deployment Guide

This backend serves the EfficientNetV2-S classifier and YOLO detector behind FastAPI.
It can load weights from a local `models/` directory or fetch them at startup.
By default the API is mounted under `/api/v1/water-hyacinth` so EC2 and Streamlit can both target a stable base path.

## Runtime Files

The API expects these filenames at runtime:

- `best_classifier.pth`
- `best_detector.pt`

By default the API looks for them in `models/` at the repo root. You can override the paths with:

- `CLASSIFIER_WEIGHTS`
- `DETECTOR_WEIGHTS`

## Base URL and OpenAPI

With the default configuration, the important routes are:

- `GET /api/v1/water-hyacinth`
- `GET /api/v1/water-hyacinth/health`
- `POST /api/v1/water-hyacinth/predict`
- `POST /api/v1/water-hyacinth/predict/batch`
- `WS /api/v1/water-hyacinth/ws/live-detect`
- `GET /api/v1/water-hyacinth/docs`
- `GET /api/v1/water-hyacinth/openapi.json`

If you need a different prefix, set:

- `API_PREFIX=/your-path`

Set `API_PREFIX=/` or an empty value if you want the API mounted at the host root instead.

A static OpenAPI file is also committed at the repo root as `openapi.json`.

## Mobile Client Integration

`POST /predict` now returns:

- classification label and confidence
- detected region count and mean detection confidence
- coverage percentage and risk level
- original image width and height
- `boxes` containing YOLO pixel coordinates for overlay rendering

`WS /ws/live-detect` accepts JSON frames like:

```json
{
  "frame_id": "frame-42",
  "image_base64": "<base64-encoded-jpeg-or-png>"
}
```

It responds with the same prediction payload plus the echoed `frame_id`, which is enough for a React Native camera view to draw live borders around detections.

## Weight Distribution Options

### Option A: Hugging Face Hub

Upload both weight files to a Hugging Face model repo first, then set:

- `HF_MODEL_REPO=<owner>/<repo>`
- `HF_CLASSIFIER_FILENAME=best_classifier.pth`
- `HF_DETECTOR_FILENAME=best_detector.pt`

Optional settings:

- `HF_MODEL_REVISION=<branch-or-tag>`
- `HF_TOKEN=<token>` if the repo is private

At startup the API will call `hf_hub_download(...)` for each missing file and cache the result under `models/`.
If the target file already exists locally, the download is skipped.

### Option B: Direct Download URLs

If you prefer GitHub Releases or another object store, upload the two files there and set:

- `CLASSIFIER_WEIGHTS_URL=https://.../best_classifier.pth`
- `DETECTOR_WEIGHTS_URL=https://.../best_detector.pt`

At startup the API downloads each missing file once into `models/` and reuses the local copy on later boots.

## Other Environment Variables

- `ALLOW_ORIGINS` comma-separated CORS origins. Default: `*`
- `API_PREFIX` path prefix for serving the API. Default: `/api/v1/water-hyacinth`
- `REQUEST_TIMEOUT_SECONDS` per-request timeout. Default: `60`
- `MAX_UPLOAD_BYTES` max size for one uploaded image in bytes. Default: `10485760`
- `MAX_BATCH_FILES` max files accepted by `/predict/batch`. Default: `16`
- `DOWNLOAD_TIMEOUT_SECONDS` timeout for direct weight downloads. Default: `120`
- `CLASSIFIER_ARCH` classifier architecture name. Default: `tf_efficientnetv2_s.in1k`
- `IMG_SIZE` classifier resize. Default: `224`
- `DET_CONF_THRESHOLD` detector confidence threshold. Default: `0.25`

Do not change `IMG_SIZE` or `DET_CONF_THRESHOLD` unless you intentionally want to diverge from the notebook-generated inference behavior.

## Local Run

From the repo root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 3000 --reload
```

Then open:

- `http://localhost:3000/api/v1/water-hyacinth/health`
- `http://localhost:3000/api/v1/water-hyacinth/docs`

## API Endpoints

- `GET /api/v1/water-hyacinth` base-path discovery
- `GET /api/v1/water-hyacinth/health` readiness and model-load status
- `POST /api/v1/water-hyacinth/predict` single-image prediction with boxes
- `POST /api/v1/water-hyacinth/predict/batch` multi-image prediction with boxes
- `WS /api/v1/water-hyacinth/ws/live-detect` live frame-by-frame detection
