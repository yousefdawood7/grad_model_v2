import base64

import pytest
from fastapi.testclient import TestClient

from backend.app import main

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4//8/AwAI/AL+KDvJYQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def client():
    original_startup = list(main.app.router.on_startup)
    main.app.router.on_startup.clear()
    main.classifier = None
    main.detector = None

    with TestClient(main.app) as test_client:
        yield test_client

    main.app.router.on_startup[:] = original_startup
    main.classifier = None
    main.detector = None


def test_health_reports_unloaded_models(client: TestClient) -> None:
    response = client.get(main.api_path("/health"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["classifier_loaded"] is False
    assert payload["detector_loaded"] is False
    assert payload["classifier_weights"].endswith("best_classifier.pth")
    assert payload["detector_weights"].endswith("best_detector.pt")
    assert payload["api_prefix"] == (main.API_PREFIX or "/")


def test_predict_returns_mocked_inference_result_with_boxes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    main.classifier = object()
    main.detector = object()
    expected = {
        "classification": "water_hyacinth",
        "classification_confidence": 0.93,
        "detected_regions": 2,
        "coverage_percent": 41.6,
        "detection_confidence": 0.77,
        "risk_level": "HIGH",
        "image_width": 640,
        "image_height": 480,
        "boxes": [
            {
                "x1": 10.0,
                "y1": 20.0,
                "x2": 110.0,
                "y2": 120.0,
                "confidence": 0.87,
                "width": 100.0,
                "height": 100.0,
            }
        ],
    }
    monkeypatch.setattr(main, "_run_inference", lambda _: expected)

    response = client.post(
        main.api_path("/predict"),
        files={"file": ("sample.png", PNG_BYTES, "image/png")},
        headers={"x-request-id": "req-test-123"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-test-123"
    assert response.json() == expected


def test_predict_batch_returns_results_for_each_uploaded_file(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main.classifier = object()
    main.detector = object()
    expected = {
        "classification": "non_water_hyacinth",
        "classification_confidence": 0.88,
        "detected_regions": 0,
        "coverage_percent": 0.0,
        "detection_confidence": 0.0,
        "risk_level": "NONE",
        "image_width": 640,
        "image_height": 480,
        "boxes": [],
    }
    monkeypatch.setattr(main, "_run_inference", lambda _: expected)

    response = client.post(
        main.api_path("/predict/batch"),
        files=[
            ("files", ("one.png", PNG_BYTES, "image/png")),
            ("files", ("two.png", PNG_BYTES, "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.json() == [
        {"image": "one.png", **expected},
        {"image": "two.png", **expected},
    ]


def test_root_describes_prefixed_api(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_prefix"] == (main.API_PREFIX or "/")
    assert payload["openapi_url"] == main.api_path("/openapi.json")


def test_api_index_describes_live_detect_websocket(client: TestClient) -> None:
    response = client.get(main.API_PREFIX or "/api/v1/water-hyacinth")

    assert response.status_code == 200
    payload = response.json()
    assert payload["live_detect_websocket"] == main.api_path("/ws/live-detect")


def test_live_detect_websocket_returns_boxes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    main.classifier = object()
    main.detector = object()
    expected = {
        "classification": "water_hyacinth",
        "classification_confidence": 0.94,
        "detected_regions": 1,
        "coverage_percent": 22.9,
        "detection_confidence": 0.89,
        "risk_level": "MEDIUM",
        "image_width": 720,
        "image_height": 1280,
        "boxes": [
            {
                "x1": 55.0,
                "y1": 210.0,
                "x2": 410.0,
                "y2": 670.0,
                "confidence": 0.89,
                "width": 355.0,
                "height": 460.0,
            }
        ],
    }

    async def fake_predict_websocket_frame(image_base64: str, request_id: str, suffix: str = ".jpg"):
        assert image_base64
        assert request_id
        return expected

    monkeypatch.setattr(main, "_predict_websocket_frame", fake_predict_websocket_frame)

    with client.websocket_connect(main.api_path("/ws/live-detect")) as websocket:
        websocket.send_json({"frame_id": "frame-1", "image_base64": base64.b64encode(PNG_BYTES).decode("utf-8")})
        payload = websocket.receive_json()

    assert payload == {"frame_id": "frame-1", **expected}
