# Gradio demo app.
# Extracted verbatim from the training notebook (Section 29 — Deployment).
# Run with: python gradio_app.py
import gradio as gr
import torch
from single_image_prediction import load_classifier, load_detector, predict

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = load_classifier("models/best_classifier.pth", device=device)
detector = load_detector("models/best_detector.pt")

def infer(image_path):
    return predict(image_path, classifier, detector, device)

demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="filepath", label="Water body image"),
    outputs=gr.JSON(label="Detection & Coverage Result"),
    title="Water Hyacinth Detection & Monitoring",
    description="Upload an image to classify, detect, and estimate water hyacinth coverage + risk level.",
)

if __name__ == "__main__":
    demo.launch(share=True)
