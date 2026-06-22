# Streamlit demo app.
# Extracted verbatim from the training notebook (Section 29 — Deployment).
# Run with: streamlit run streamlit_app.py
import streamlit as st
import torch, tempfile, cv2
from single_image_prediction import load_classifier, load_detector, predict

st.set_page_config(page_title="Water Hyacinth Monitor", layout="centered")
st.title("🌊 Water Hyacinth Detection & Coverage Monitor")

@st.cache_resource
def get_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_classifier("models/best_classifier.pth", device=device), load_detector("models/best_detector.pt"), device

classifier, detector, device = get_models()

uploaded = st.file_uploader("Upload a water-body image", type=["jpg", "jpeg", "png"])
if uploaded:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    st.image(tmp_path, caption="Uploaded image", use_column_width=True)
    result = predict(tmp_path, classifier, detector, device)
    st.subheader("Results")
    st.json(result)
    risk_colors = {"NONE":"green","LOW":"green","MEDIUM":"orange","HIGH":"red","CRITICAL":"darkred"}
    st.markdown(f"### Risk Level: :{risk_colors.get(result['risk_level'],'gray')}[{result['risk_level']}]")
