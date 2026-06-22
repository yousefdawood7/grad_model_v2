"""
Water Hyacinth Monitor - Streamlit frontend.

Talks to the FastAPI backend over HTTP (set API_URL), so this can be deployed
independently on Streamlit Community Cloud while the backend runs wherever the
models and runtime resources live.

Run locally:
    streamlit run streamlit_app.py

Configure the backend location via Streamlit secrets or an env var:
    API_URL=https://your-backend-host/api/v1/water-hyacinth
"""
import os

import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000/api/v1/water-hyacinth"))
PREDICT_ENDPOINT = f"{API_URL.rstrip('/')}/predict"
HEALTH_ENDPOINT = f"{API_URL.rstrip('/')}/health"

st.set_page_config(page_title="Water Hyacinth Monitor", layout="centered")
st.title("Water Hyacinth Detection & Coverage Monitor")
st.caption(f"Backend: {API_URL}")

with st.sidebar:
    st.subheader("Backend status")
    try:
        health_response = requests.get(HEALTH_ENDPOINT, timeout=5)
        health_response.raise_for_status()
        st.json(health_response.json())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not reach backend: {exc}")

uploaded = st.file_uploader("Upload a water-body image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    if st.button("Run detection", type="primary"):
        with st.spinner("Calling the model..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = requests.post(PREDICT_ENDPOINT, files=files, timeout=60)
                resp.raise_for_status()
                result = resp.json()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
            else:
                st.subheader("Results")
                st.json(result)

                risk_colors = {
                    "NONE": "green",
                    "LOW": "green",
                    "MEDIUM": "orange",
                    "HIGH": "red",
                    "CRITICAL": "darkred",
                }
                risk = result.get("risk_level", "UNKNOWN")
                st.markdown(f"### Risk Level: :{risk_colors.get(risk, 'gray')}[{risk}]")
                st.metric("Coverage", f"{result.get('coverage_percent', 0)}%")
                st.metric("Detected regions", result.get("detected_regions", 0))

