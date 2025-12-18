import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from PIL import Image
import io
import uvicorn
import os

MODEL_PATH = "model/flower_model_fixed.keras"
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        img_bytes = await file.read()
        processed = preprocess_image(img_bytes)

        preds = model.predict(processed)
        class_index = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))

        return {
            "class": CLASS_NAMES[class_index],
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Serve static files - MOUNT THIS LAST after all other routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Add a catch-all route to serve index.html for SPA
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)