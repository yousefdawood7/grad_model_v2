from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(description="Left pixel coordinate.")
    y1: float = Field(description="Top pixel coordinate.")
    x2: float = Field(description="Right pixel coordinate.")
    y2: float = Field(description="Bottom pixel coordinate.")
    confidence: float = Field(description="Detector confidence for this box.")
    width: float = Field(description="Bounding box width in pixels.")
    height: float = Field(description="Bounding box height in pixels.")


class PredictionResponse(BaseModel):
    classification: str = Field(description="Predicted top-1 class label.")
    classification_confidence: float = Field(description="Softmax confidence for the predicted class.")
    detected_regions: int = Field(description="Number of detector bounding boxes returned at conf=0.25.")
    coverage_percent: float = Field(description="Estimated percentage of the image area covered by detections.")
    detection_confidence: float = Field(description="Mean confidence across detected regions, or 0 when none exist.")
    risk_level: str = Field(description="NONE | LOW | MEDIUM | HIGH | CRITICAL")
    image_width: int = Field(description="Original image width in pixels.")
    image_height: int = Field(description="Original image height in pixels.")
    boxes: list[BoundingBox] = Field(description="Detected YOLO bounding boxes for rendering overlays.")


class BatchPredictionItem(PredictionResponse):
    image: str = Field(description="Original uploaded filename associated with this prediction.")


class LiveDetectionResponse(PredictionResponse):
    frame_id: str | None = Field(default=None, description="Client-provided frame identifier echoed back by the websocket.")


class HealthResponse(BaseModel):
    status: str
    device: str
    classifier_loaded: bool
    detector_loaded: bool
    classifier_weights: str
    detector_weights: str
    api_prefix: str
