import io
import time
import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5

app = FastAPI()

print("Loading model...")

model = tf.keras.models.load_model(
    "/app/model.keras",
    compile=False
)

print("Model loaded.")

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype(np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image

@app.get("/")
def root():

    return {
        "message": "Water Hyacinth API 🌿"
    }

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    start = time.time()

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="File must be an image"
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

        return {

            "prediction": predicted_class,

            "probability": round(prediction, 5),

            "confidence": round(confidence, 5),

            "inference_time": round(
                time.time() - start,
                3
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )