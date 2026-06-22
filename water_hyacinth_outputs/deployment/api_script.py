# FastAPI inference service.
# Run with: uvicorn api_script:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, UploadFile, File
import torch, shutil, uuid
from single_image_prediction import load_classifier, load_detector, predict

app = FastAPI(title="Water Hyacinth Detection API")
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = load_classifier("models/best_classifier.pth", device=device)
detector = load_detector("models/best_detector.pt")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    tmp_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = predict(tmp_path, classifier, detector, device)
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "device": device}
