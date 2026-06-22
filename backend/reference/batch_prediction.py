# Batch prediction over a folder of images -> CSV.
# Extracted verbatim from the training notebook (Section 29 — Deployment).
import sys, glob, csv
from single_image_prediction import load_classifier, load_detector, predict
import torch

def main(folder, out_csv="batch_predictions.csv"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = load_classifier("models/best_classifier.pth", device=device)
    detector = load_detector("models/best_detector.pt")
    rows = []
    for fp in glob.glob(f"{folder}/*"):
        try:
            r = predict(fp, classifier, detector, device)
            r["image"] = fp
            rows.append(r)
        except Exception as e:
            print("skip", fp, e)
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        print(f"Saved {len(rows)} predictions to {out_csv}")

if __name__ == "__main__":
    main(sys.argv[1])
