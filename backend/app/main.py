"""
Water Hyacinth Detection API.

Production-shaped FastAPI service that preserves the exact inference behavior
from backend/reference/single_image_prediction.py while adding operational
hardening for deployment.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from starlette.concurrency import run_in_threadpool

from .schemas import BatchPredictionItem, HealthResponse, LiveDetectionResponse, PredictionResponse

LOGGER = logging.getLogger("water_hyacinth_api")
if not LOGGER.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")

CLASSES = ["non_water_hyacinth", "water_hyacinth"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

CLASSIFIER_ARCH = os.getenv("CLASSIFIER_ARCH", "tf_efficientnetv2_s.in1k")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
DET_CONF_THRESHOLD = float(os.getenv("DET_CONF_THRESHOLD", "0.25"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "16"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "120"))
ALLOW_ORIGINS = [origin.strip() for origin in os.getenv("ALLOW_ORIGINS", "*").split(",") if origin.strip()]
RAW_API_PREFIX = os.getenv("API_PREFIX", "/api/v1/water-hyacinth").strip()

if RAW_API_PREFIX in {"", "/"}:
    API_PREFIX = ""
else:
    API_PREFIX = f"/{RAW_API_PREFIX.strip('/')}"

CLASSIFIER_FILENAME = os.getenv("HF_CLASSIFIER_FILENAME", "best_classifier.pth")
DETECTOR_FILENAME = os.getenv("HF_DETECTOR_FILENAME", "best_detector.pt")
CLASSIFIER_PATH = Path(os.getenv("CLASSIFIER_WEIGHTS", str(MODELS_DIR / CLASSIFIER_FILENAME)))
DETECTOR_PATH = Path(os.getenv("DETECTOR_WEIGHTS", str(MODELS_DIR / DETECTOR_FILENAME)))

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION")
HF_TOKEN = os.getenv("HF_TOKEN")
CLASSIFIER_WEIGHTS_URL = os.getenv("CLASSIFIER_WEIGHTS_URL")
DETECTOR_WEIGHTS_URL = os.getenv("DETECTOR_WEIGHTS_URL")

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier: Any = None
detector: Any = None


def api_path(path: str) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    if API_PREFIX:
        return f"{API_PREFIX}{path}"
    return path


app = FastAPI(
    title="Water Hyacinth Detection API",
    summary="Classify water-hyacinth presence and estimate coverage risk from uploaded images.",
    description=(
        "This service runs the notebook-derived EfficientNetV2-S classifier and YOLO detector used "
        "for water-hyacinth monitoring. By default it is mounted under `/api/v1/water-hyacinth` so EC2 and "
        "Streamlit deployments can target a stable base path. `/predict` accepts a single JPG/PNG "
        "image and returns the classification label, confidence, detection boxes, coverage percentage, "
        "and derived risk level. `/predict/batch` applies the same inference pipeline to multiple images "
        "in one request, and `/ws/live-detect` lets mobile apps stream frames over a websocket and receive "
        "box coordinates for live overlays. Model weights can be mounted locally or downloaded at startup "
        "from Hugging Face Hub or direct asset URLs."
    ),
    version="0.4.0",
    docs_url=api_path("/docs"),
    redoc_url=api_path("/redoc"),
    openapi_url=api_path("/openapi.json"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    LOGGER.info(json.dumps(payload, default=str))


def _download_from_url(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _log_event("weights_cached", path=str(destination))
        return destination

    with requests.get(url, stream=True, timeout=(10, DOWNLOAD_TIMEOUT_SECONDS)) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
    tmp_path.replace(destination)
    _log_event("weights_downloaded", path=str(destination), source="url")
    return destination


def _prepare_weight_file(target_path: Path, filename: str, fallback_url: str | None) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path

    if HF_MODEL_REPO:
        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=filename,
            revision=HF_MODEL_REVISION,
            token=HF_TOKEN,
            local_dir=str(target_path.parent),
        )
        resolved = Path(downloaded)
        if resolved != target_path and resolved.exists():
            resolved.replace(target_path)
        _log_event("weights_downloaded", path=str(target_path), source="huggingface")
        return target_path

    if fallback_url:
        return _download_from_url(fallback_url, target_path)

    return target_path


def _ensure_weights_available() -> None:
    if HF_MODEL_REPO or CLASSIFIER_WEIGHTS_URL:
        _prepare_weight_file(CLASSIFIER_PATH, CLASSIFIER_FILENAME, CLASSIFIER_WEIGHTS_URL)
    if HF_MODEL_REPO or DETECTOR_WEIGHTS_URL:
        _prepare_weight_file(DETECTOR_PATH, DETECTOR_FILENAME, DETECTOR_WEIGHTS_URL)


def _load_models() -> None:
    global classifier, detector
    import timm
    from ultralytics import YOLO

    _ensure_weights_available()

    if CLASSIFIER_PATH.exists():
        classifier = timm.create_model(CLASSIFIER_ARCH, pretrained=False, num_classes=len(CLASSES))
        classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
        classifier.eval().to(device)
        _log_event("model_loaded", model="classifier", path=str(CLASSIFIER_PATH), device=device)
    else:
        LOGGER.warning("Classifier weights not found at %s; prediction routes will return 503 until available.", CLASSIFIER_PATH)

    if DETECTOR_PATH.exists():
        detector = YOLO(str(DETECTOR_PATH))
        _log_event("model_loaded", model="detector", path=str(DETECTOR_PATH), device=device)
    else:
        LOGGER.warning("Detector weights not found at %s; prediction routes will return 503 until available.", DETECTOR_PATH)


def _models_ready() -> bool:
    return classifier is not None and detector is not None


def _validate_image_bytes(image_bytes: bytes) -> None:
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Uploaded file exceeds the {MAX_UPLOAD_BYTES} byte limit.")


def _save_image_bytes(image_bytes: bytes, suffix: str = ".jpg") -> Path:
    _validate_image_bytes(image_bytes)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        return Path(tmp.name)


def _parse_data_url(data: str) -> bytes:
    payload = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from exc


def _run_inference(image_path: str) -> dict[str, Any]:
    import cv2

    raw = cv2.imread(image_path)
    if raw is None:
        raise ValueError("Could not decode image")

    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    norm = (resized - mean) / std
    tensor = torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)[0].cpu().numpy()
    cls_idx = int(probs.argmax())

    result = detector.predict(image_path, conf=DET_CONF_THRESHOLD, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4))
    confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.empty((0,))
    image_height, image_width = img.shape[:2]
    covered = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
    coverage_pct = min(100.0, 100 * covered / (image_height * image_width)) if image_height * image_width else 0.0
    risk = (
        "NONE"
        if coverage_pct == 0
        else "LOW"
        if coverage_pct < 15
        else "MEDIUM"
        if coverage_pct < 40
        else "HIGH"
        if coverage_pct < 70
        else "CRITICAL"
    )

    serialized_boxes = [
        {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(conf),
            "width": float(x2 - x1),
            "height": float(y2 - y1),
        }
        for (x1, y1, x2, y2), conf in zip(boxes, confs, strict=False)
    ]

    return {
        "classification": CLASSES[cls_idx],
        "classification_confidence": float(probs[cls_idx]),
        "detected_regions": int(len(boxes)),
        "coverage_percent": round(coverage_pct, 1),
        "detection_confidence": float(confs.mean()) if len(confs) else 0.0,
        "risk_level": risk,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "boxes": serialized_boxes,
    }


async def _save_upload(upload: UploadFile) -> Path:
    if upload.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported.")

    suffix = Path(upload.filename or "upload.jpg").suffix or ".jpg"
    chunks: list[bytes] = []
    total_bytes = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Uploaded file exceeds the {MAX_UPLOAD_BYTES} byte limit.")
        chunks.append(chunk)
    await upload.close()
    return _save_image_bytes(b"".join(chunks), suffix=suffix)


async def _predict_from_path(image_path: Path, request_id: str) -> dict[str, Any]:
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(_run_inference, str(image_path)),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Inference timed out.") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Inference failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        image_path.unlink(missing_ok=True)

    _log_event(
        "prediction_completed",
        request_id=request_id,
        classification=result["classification"],
        coverage_percent=result["coverage_percent"],
        detected_regions=result["detected_regions"],
        risk_level=result["risk_level"],
    )
    return result


async def _predict_file(upload: UploadFile, request_id: str) -> dict[str, Any]:
    tmp_path = await _save_upload(upload)
    return await _predict_from_path(tmp_path, request_id)


async def _predict_websocket_frame(image_base64: str, request_id: str, suffix: str = ".jpg") -> dict[str, Any]:
    image_bytes = _parse_data_url(image_base64)
    tmp_path = _save_image_bytes(image_bytes, suffix=suffix)
    return await _predict_from_path(tmp_path, request_id)


async def _predict_batch(files: list[UploadFile], request_id: str) -> list[BatchPredictionItem]:
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Batch requests are limited to {MAX_BATCH_FILES} files.",
        )

    results: list[BatchPredictionItem] = []
    for upload in files:
        try:
            prediction = await _predict_file(upload, request_id)
        except HTTPException as exc:
            _log_event(
                "batch_item_skipped",
                request_id=request_id,
                status_code=exc.status_code,
            )
            continue
        results.append(BatchPredictionItem(image=upload.filename or "upload", **prediction))
    return results


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
    request.state.request_id = request_id

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_UPLOAD_BYTES * max(MAX_BATCH_FILES, 1):
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body exceeds the configured size limit."},
            headers={"X-Request-ID": request_id},
        )

    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS + 5)
    except TimeoutError:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        _log_event(
            "request_timeout",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            latency_ms=latency_ms,
        )
        return JSONResponse(
            status_code=504,
            content={"detail": "Request timed out."},
            headers={"X-Request-ID": request_id},
        )

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    _log_event(
        "request_completed",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    return response


@app.on_event("startup")
def startup() -> None:
    _load_models()


@app.get(
    "/",
    summary="Describe the API entrypoint",
    description="Returns the configured API base path and documentation URLs.",
)
async def root() -> dict[str, str]:
    return {
        "message": "Water Hyacinth Detection API",
        "api_prefix": API_PREFIX or "/",
        "docs_url": api_path("/docs"),
        "openapi_url": api_path("/openapi.json"),
    }


@app.get(
    API_PREFIX or "/api/v1/water-hyacinth",
    summary="Describe the mounted API base path",
    description="Returns the base path clients should use for health, prediction, and websocket routes.",
)
async def api_index() -> dict[str, str]:
    return {
        "base_url_path": API_PREFIX or "/",
        "health": api_path("/health"),
        "predict": api_path("/predict"),
        "predict_batch": api_path("/predict/batch"),
        "live_detect_websocket": api_path("/ws/live-detect"),
        "openapi": api_path("/openapi.json"),
    }


@app.post(
    api_path("/predict"),
    response_model=PredictionResponse,
    summary="Predict water-hyacinth coverage and bounding boxes for one image",
    description=(
        "Runs the notebook-derived classifier and detector on one uploaded JPG/PNG image. "
        "The response includes YOLO bounding boxes that mobile or web clients can draw as overlays. "
        "The coverage percentage and risk level are computed with the same formula used in "
        "backend/reference/single_image_prediction.py."
    ),
)
async def predict_endpoint(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    if not _models_ready():
        raise HTTPException(status_code=503, detail="Models not loaded; check server logs and weight configuration.")

    result = await _predict_file(file, request.state.request_id)
    return PredictionResponse(**result)


@app.post(
    api_path("/predict/batch"),
    response_model=list[BatchPredictionItem],
    summary="Predict water-hyacinth coverage and bounding boxes for multiple images",
    description=(
        "Applies the same inference pipeline used by `/predict` to a batch of uploaded images. "
        "Each item includes the YOLO bounding boxes needed for frontend overlays."
    ),
)
async def predict_batch_endpoint(
    request: Request,
    files: list[UploadFile] = File(...),
) -> list[BatchPredictionItem]:
    if not _models_ready():
        raise HTTPException(status_code=503, detail="Models not loaded; check server logs and weight configuration.")

    results = await _predict_batch(files, request.state.request_id)
    if not results:
        raise HTTPException(status_code=400, detail="No valid predictions were produced for the uploaded files.")
    return results


@app.websocket(api_path("/ws/live-detect"))
async def live_detect_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    request_id = websocket.headers.get("x-request-id", uuid.uuid4().hex)

    if not _models_ready():
        await websocket.send_json({"detail": "Models not loaded; check server logs and weight configuration."})
        await websocket.close(code=1013)
        return

    try:
        while True:
            payload = await websocket.receive_json()
            frame_id = payload.get("frame_id")
            image_base64 = payload.get("image_base64")
            if not image_base64:
                await websocket.send_json({"frame_id": frame_id, "detail": "image_base64 is required."})
                continue

            try:
                prediction = await _predict_websocket_frame(image_base64, request_id)
            except HTTPException as exc:
                await websocket.send_json({"frame_id": frame_id, "detail": exc.detail, "status_code": exc.status_code})
                continue

            await websocket.send_json(LiveDetectionResponse(frame_id=frame_id, **prediction).model_dump())
    except WebSocketDisconnect:
        _log_event("websocket_disconnected", request_id=request_id, path=api_path("/ws/live-detect"))


@app.get(
    api_path("/health"),
    response_model=HealthResponse,
    summary="Check API readiness",
    description=(
        "Returns the serving device and whether the classifier and detector are loaded. "
        "Use this for container health checks, EC2 checks, and Streamlit connectivity checks."
    ),
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=device,
        classifier_loaded=classifier is not None,
        detector_loaded=detector is not None,
        classifier_weights=str(CLASSIFIER_PATH),
        detector_weights=str(DETECTOR_PATH),
        api_prefix=API_PREFIX or "/",
    )
