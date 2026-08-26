#!/usr/bin/env python
"""
UVIP-AI Model Inference Service
FastAPI service untuk serve model predictions.

Usage:
    uvicorn model_service:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch

app = FastAPI(
    title="UVIP-AI Model Service",
    description="AI inference service for urban visual perception",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SegFormer model
print("Loading SegFormer-B0...")
seg_processor = SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512'
).to('cuda' if torch.cuda.is_available() else 'cpu').to(torch.float16).eval()
print("✓ SegFormer loaded")

# Load XGBoost models
print("Loading XGBoost models...")
models = {}
model_dir = Path('models/perception')

for name in ['beauty', 'safety', 'comfort', 'uvi']:
    model_path = model_dir / f'{name}_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, 'rb') as f:
        models[name] = pickle.load(f)
    print(f"✓ Loaded {name} model")

print(f"✓ All models loaded ({len(models)} models)")


def extract_segmentation_features(image: Image.Image) -> dict:
    """
    Extract segmentation features dari gambar.
    Returns dict dengan 14 features yang match backend schema.
    """
    # SegFormer inference
    inputs = seg_processor(images=image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

    with torch.no_grad():
        outputs = seg_model(**inputs)

    # Get segmentation map
    seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy()
    total_pixels = seg_map.size

    # Calculate percentages (simplified heuristic mapping)
    # ADE20K has 150 classes, kita map ke categories
    unique_classes, counts = np.unique(seg_map, return_counts=True)

    # Heuristic mapping based on typical ADE20K class indices
    vegetation_pct = sum(counts[unique_classes <= 20]) / total_pixels * 100
    building_pct = sum(counts[(unique_classes >= 21) & (unique_classes <= 40)]) / total_pixels * 100
    road_pct = sum(counts[(unique_classes >= 41) & (unique_classes <= 50)]) / total_pixels * 100
    sidewalk_pct = sum(counts[(unique_classes >= 51) & (unique_classes <= 55)]) / total_pixels * 100
    sky_pct = sum(counts[(unique_classes >= 56) & (unique_classes <= 60)]) / total_pixels * 100
    signage_pct = sum(counts[(unique_classes >= 61) & (unique_classes <= 70)]) / total_pixels * 100
    vehicle_pct = sum(counts[(unique_classes >= 71) & (unique_classes <= 80)]) / total_pixels * 100
    pedestrian_pct = sum(counts[(unique_classes >= 81) & (unique_classes <= 85)]) / total_pixels * 100
    street_furniture_pct = sum(counts[(unique_classes >= 86) & (unique_classes <= 100)]) / total_pixels * 100

    # Derived metrics
    green_coverage_pct = vegetation_pct
    building_coverage_pct = building_pct
    sky_visibility_pct = sky_pct

    # Walkability ratio: sidewalk / (road + sidewalk)
    walkability_ratio = sidewalk_pct / (road_pct + sidewalk_pct + 1e-6)

    # Visual clutter index: signage + vehicle + street_furniture
    visual_clutter_index = signage_pct + vehicle_pct + street_furniture_pct

    return {
        'vegetation_pct': round(float(vegetation_pct), 2),
        'building_pct': round(float(building_pct), 2),
        'road_pct': round(float(road_pct), 2),
        'sidewalk_pct': round(float(sidewalk_pct), 2),
        'sky_pct': round(float(sky_pct), 2),
        'signage_pct': round(float(signage_pct), 2),
        'vehicle_pct': round(float(vehicle_pct), 2),
        'pedestrian_pct': round(float(pedestrian_pct), 2),
        'street_furniture_pct': round(float(street_furniture_pct), 2),
        'green_coverage_pct': round(float(green_coverage_pct), 2),
        'building_coverage_pct': round(float(building_coverage_pct), 2),
        'sky_visibility_pct': round(float(sky_visibility_pct), 2),
        'walkability_ratio': round(float(walkability_ratio), 4),
        'visual_clutter_index': round(float(visual_clutter_index), 2),
    }


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "UVIP-AI Model Service",
        "version": "1.0.0",
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.get("/health")
async def health():
    """Health check for monitoring."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict perception scores dari gambar.

    Returns:
        - segmentation: 14 segmentation features
        - predictions: beauty, safety, comfort, uvi scores
        - inference_time_ms: processing time
    """
    start_time = time.time()

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Extract segmentation features
        segmentation = extract_segmentation_features(image)

        # Prepare feature vector for XGBoost
        feature_vector = np.array([[
            segmentation['vegetation_pct'],
            segmentation['building_pct'],
            segmentation['road_pct'],
            segmentation['sidewalk_pct'],
            segmentation['sky_pct'],
            segmentation['signage_pct'],
            segmentation['vehicle_pct'],
            segmentation['pedestrian_pct'],
            segmentation['street_furniture_pct'],
            segmentation['green_coverage_pct'],
            segmentation['building_coverage_pct'],
            segmentation['sky_visibility_pct'],
            segmentation['walkability_ratio'],
            segmentation['visual_clutter_index'],
        ]])

        # Predict with all models
        predictions = {}
        for name, model in models.items():
            pred = model.predict(feature_vector)[0]
            predictions[f"{name}_score"] = round(float(pred), 2)

        inference_time = (time.time() - start_time) * 1000

        return {
            "segmentation": segmentation,
            "predictions": predictions,
            "inference_time_ms": round(inference_time, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple images sekaligus.
    """
    results = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')

            segmentation = extract_segmentation_features(image)

            feature_vector = np.array([[
                segmentation['vegetation_pct'],
                segmentation['building_pct'],
                segmentation['road_pct'],
                segmentation['sidewalk_pct'],
                segmentation['sky_pct'],
                segmentation['signage_pct'],
                segmentation['vehicle_pct'],
                segmentation['pedestrian_pct'],
                segmentation['street_furniture_pct'],
                segmentation['green_coverage_pct'],
                segmentation['building_coverage_pct'],
                segmentation['sky_visibility_pct'],
                segmentation['walkability_ratio'],
                segmentation['visual_clutter_index'],
            ]])

            predictions = {}
            for name, model in models.items():
                pred = model.predict(feature_vector)[0]
                predictions[f"{name}_score"] = round(float(pred), 2)

            results.append({
                "filename": file.filename,
                "segmentation": segmentation,
                "predictions": predictions
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"results": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
