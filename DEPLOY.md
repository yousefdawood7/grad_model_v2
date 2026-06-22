# Deploying the Water Hyacinth API and Streamlit Demo

This repo is set up for a split deployment:

- `backend/` runs the FastAPI inference service in Docker.
- `streamlit_app.py` runs on Streamlit Community Cloud and talks to the backend over HTTPS.

The backend defaults to the `/api/v1/water-hyacinth` base path. On EC2, your usable URLs look like:

- `http://<ec2-public-ip>:3000/api/v1/water-hyacinth/health`
- `http://<ec2-public-ip>:3000/api/v1/water-hyacinth/docs`
- `http://<ec2-public-ip>:3000/api/v1/water-hyacinth/openapi.json`

## 1. Prepare the weight files

You need these two files from training:

- `best_classifier.pth`
- `best_detector.pt`

Choose one distribution method before you deploy the backend.

### Option A: Hugging Face Hub

1. Create a Hugging Face model repo.
2. Upload `best_classifier.pth` and `best_detector.pt`.
3. Note the repo id, for example `your-name/water-hyacinth-models`.
4. If the repo is private, create a read token.

### Option B: Direct asset URLs

1. Create a GitHub Release or upload both files to cloud storage.
2. Copy the public download URLs for each file.

## 2. Deploy the backend on EC2 with Docker

1. Launch an Ubuntu EC2 instance.
2. Allow inbound traffic to port `3000` in the EC2 security group.
3. SSH into the instance.
4. Install Docker.
5. Clone this repo.
6. Build the image from the repo root.

```bash
docker build -t water-hyacinth-api .
```

7. Run the container and set your environment variables.

If you use Hugging Face Hub:

```bash
docker run -d \
  --name water-hyacinth-api \
  -p 3000:3000 \
  -e HF_MODEL_REPO=your-name/water-hyacinth-models \
  -e HF_CLASSIFIER_FILENAME=best_classifier.pth \
  -e HF_DETECTOR_FILENAME=best_detector.pt \
  -e API_PREFIX=/api/v1/water-hyacinth \
  -e ALLOW_ORIGINS=https://your-streamlit-app.streamlit.app \
  water-hyacinth-api
```

If you use direct asset URLs:

```bash
docker run -d \
  --name water-hyacinth-api \
  -p 3000:3000 \
  -e CLASSIFIER_WEIGHTS_URL=https://example.com/best_classifier.pth \
  -e DETECTOR_WEIGHTS_URL=https://example.com/best_detector.pt \
  -e API_PREFIX=/api/v1/water-hyacinth \
  -e ALLOW_ORIGINS=https://your-streamlit-app.streamlit.app \
  water-hyacinth-api
```

8. Verify the service.

```bash
curl http://localhost:3000/api/v1/water-hyacinth/health
curl http://localhost:3000/api/v1/water-hyacinth/openapi.json
```

From your machine, use:

```text
http://<ec2-public-ip>:3000/api/v1/water-hyacinth
```

## 3. Deploy the Streamlit frontend

1. Push the repo to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from this repo.
4. Set the main file path to `streamlit_app.py`.
5. Add the secret below.

```toml
API_URL = "http://<ec2-public-ip>:3000/api/v1/water-hyacinth"
```

Important: `API_URL` must include the `/api/v1/water-hyacinth` base path.

6. Deploy the app.
7. Once Streamlit gives you a public URL, set that exact origin in `ALLOW_ORIGINS` on EC2 if you initially left CORS open.

## 4. Local smoke test

Backend:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 3000 --reload
```

Frontend:

```bash
pip install -r requirements.txt
API_URL=http://localhost:3000/api/v1/water-hyacinth streamlit run streamlit_app.py
```

Then:

1. Open the local Streamlit page.
2. Confirm the sidebar can read `/api/v1/water-hyacinth/health`.
3. Upload a JPG or PNG image and confirm the result JSON appears.

## 5. Files relevant to deployment

- `Dockerfile`
- `backend/Dockerfile`
- `backend/README.md`
- `.streamlit/config.toml`
- `streamlit_app.py`
- `openapi.json`



