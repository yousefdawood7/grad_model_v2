from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    classification: str = Field(description="Predicted top-1 class label.")
    classification_confidence: float = Field(description="Softmax confidence for the predicted class.")
    detected_regions: int = Field(description="Number of detector bounding boxes returned at conf=0.25.")
    coverage_percent: float = Field(description="Estimated percentage of the image area covered by detections.")
    detection_confidence: float = Field(description="Mean confidence across detected regions, or 0 when none exist.")
    risk_level: str = Field(description="NONE | LOW | MEDIUM | HIGH | CRITICAL")


class BatchPredictionItem(PredictionResponse):
    image: str = Field(description="Original uploaded filename associated with this prediction.")


class HealthResponse(BaseModel):
    status: str
    device: str
    classifier_loaded: bool
    detector_loaded: bool
    classifier_weights: str
    detector_weights: str
    api_prefix: str
