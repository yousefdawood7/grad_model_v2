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


def test_predict_returns_mocked_inference_result(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    main.classifier = object()
    main.detector = object()
    expected = {
        "classification": "water_hyacinth",
        "classification_confidence": 0.93,
        "detected_regions": 2,
        "coverage_percent": 41.6,
        "detection_confidence": 0.77,
        "risk_level": "HIGH",
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
