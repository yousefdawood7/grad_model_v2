# Single-image prediction: classification + detection + coverage estimate.
import sys, json
import torch, cv2, numpy as np
from torchvision import transforms

CLASSES = ["non_water_hyacinth", "water_hyacinth"]
IMG_SIZE = 224

def load_classifier(weights_path, arch="tf_efficientnetv2_s.in1k", device="cpu"):
    import timm
    model = timm.create_model(arch, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval().to(device)
    return model

def load_detector(weights_path):
    from ultralytics import YOLO
    return YOLO(weights_path)

def predict(image_path, classifier, detector, device="cpu"):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    norm = (resized - mean) / std
    tensor = torch.tensor(norm.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)[0].cpu().numpy()
    cls_idx = int(probs.argmax())

    result = detector.predict(image_path, conf=0.25, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0,4))
    confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.empty((0,))
    H, W = img.shape[:2]
    covered = sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes)
    coverage_pct = min(100.0, 100*covered/(H*W)) if H*W else 0.0
    risk = "NONE" if coverage_pct==0 else "LOW" if coverage_pct<15 else "MEDIUM" if coverage_pct<40 else "HIGH" if coverage_pct<70 else "CRITICAL"

    return {
        "classification": CLASSES[cls_idx],
        "classification_confidence": float(probs[cls_idx]),
        "detected_regions": int(len(boxes)),
        "coverage_percent": round(coverage_pct, 1),
        "detection_confidence": float(confs.mean()) if len(confs) else 0.0,
        "risk_level": risk,
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = load_classifier("models/best_classifier.pth", device=device)
    detector = load_detector("models/best_detector.pt")
    out = predict(sys.argv[1], classifier, detector, device)
    print(json.dumps(out, indent=2))
