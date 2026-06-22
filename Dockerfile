FROM python:3.11-slim

WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CLASSIFIER_WEIGHTS=/srv/models/best_classifier.pth \
    DETECTOR_WEIGHTS=/srv/models/best_detector.pt \
    API_PREFIX=/api/v1/water-hyacinth \
    PORT=3000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY api ./api
COPY models ./models
COPY openapi.json ./openapi.json

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3000/api/v1/water-hyacinth/health')"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "3000"]


