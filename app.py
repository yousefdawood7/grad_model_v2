

import io
import time
import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


IMAGE_SIZE = (224, 224)

THRESHOLD = 0.5

MODEL_PATH = "model.keras"

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


app = FastAPI(
    title="Water Hyacinth Detection API",
    description="Detect water hyacinth from uploaded images 🌿",
    version="1.0.0"
)


def preprocess_image(image: Image.Image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype(np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


@app.get("/")
def home():

    return {
        "message": "Water Hyacinth Detection API 🌿",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    start_time = time.time()


    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents))


        processed = preprocess_image(image)

        prediction = model.predict(
            processed,
            verbose=0
        )[0][0]

        prediction = float(prediction)

        predicted_class = (
            "water_hyacinth"
            if prediction > THRESHOLD
            else "not_water_hyacinth"
        )

        confidence = (
            prediction
            if prediction > THRESHOLD
            else 1 - prediction
        )

        inference_time = round(
            time.time() - start_time,
            3
        )

        return {

            "prediction": predicted_class,

            "probability": round(prediction, 5),

            "confidence": round(float(confidence), 5),

            "threshold": THRESHOLD,

            "inference_time_seconds": inference_time
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )