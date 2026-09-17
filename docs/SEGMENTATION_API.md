# Segmentation API Documentation - Excel-Aligned Mode

## Overview

Enhanced semantic segmentation module with Excel-compliant composite visual indices, moving object filtering, and road/sidewalk separation.

**Module:** `src/uvip_ai/segmentation.segformer.SegformerB5`

---

## Quick Start

```python
from uvip_ai.segmentation.segformer import SegformerB5

# Initialize model
model = SegformerB5(
    model_id="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    device="cuda",  # or "cpu"
    low_vram_mode=True  # Recommended for < 12GB VRAM
)

# Run inference in Excel mode
result = model.infer(
    image="path/to/image.jpg",
    max_resolution=512,
    excel_mode=True  # Enable Excel-compliant metrics
)

# Access results
indices = result["metrics"]
print(indices["building_visibility_index"])
```

---

## Class: SegformerB5

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `"nvidia/segformer-b5-finetuned-cityscapes-1024-1024"` | HuggingFace model identifier |
| `device` | `str` | Auto-detected | `"cuda"` or `"cpu"` |
| `low_vram_mode` | `bool` | `True` | Enable memory optimization (float16) |

### Initialization Example

```python
# Standard usage (recommended)
model = SegformerB5(low_vram_mode=True)

# Custom model variant
model = SegformerB5(
    model_id="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",  # Faster, less accurate
    device="cuda",
    low_vram_mode=False
)

# CPU-only fallback
model = SegformerB5(device="cpu")
```

---

## Method: infer()

### Signature

```python
def infer(
    self,
    image: Image.Image | np.ndarray | str,
    max_resolution: int = 512,
    excel_mode: bool = False
) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | `Image.Image \| np.ndarray \| str` | ✓ | Input image (file path, PIL, or numpy array) |
| `max_resolution` | `int` | ✗ | Maximum dimension for inference (default: 512px). Higher = more accurate but slower. |
| `excel_mode` | `bool` | ✗ | If True, return Excel-compliant composite indices. If False, return original urban metrics. |

### Returns

Returns dictionary with 3 keys:

```python
{
    "seg_map": np.ndarray,      # Semantic segmentation map (HxW, uint8)
    "metrics": dict,            # Metrics based on mode
    "pct_by_class": dict        # Percentage by category
}
```

### Usage Examples

#### Basic Inference (Original Metrics)

```python
result = model.infer("input_image.jpg")

# Output metrics (urban-focused):
metrics = result["metrics"]
{
    "green_coverage_pct": 45.23,
    "building_coverage_pct": 28.91,
    "walkability_ratio": 0.67,
    "visual_clutter_index": 0.12,
    "sky_visibility_pct": 15.44
}
```

#### Excel-Aligned Mode (Recommended for Analysis)

```python
result = model.infer(
    image="input_image.jpg",
    excel_mode=True,
    max_resolution=768  # Higher accuracy
)

# Output metrics (Excel-compliant):
metrics = result["metrics"]
{
    "building_visibility_index": 0.2891,
    "vegetation_coverage_index": 0.4523,
    "sky_openness_index": 0.1544,
    "ground_accessibility_index": 0.2345,
    "human_activity_index": 0.0089,
    "vehicle_intensity_index": 0.0958,
    "traffic_infrastructure_index": 0.0123,
    "heritage_dominance_index": 0.1567,
    "_static_normalization_factor": 1.1234,
    "_moving_object_percentage": 12.47,
    "_raw_percentages": {...}
}
```

---

## Excel-Aligned Composite Indices Explained

### 1. Building Visibility Index

**Range:** `[0, 1]`  
**Formula:** `building_percentage / 100`  
**Interpretation:** Proportion of built environment in static analysis  
**Example:** `0.25` = 25% building coverage (excluding moving objects)

### 2. Vegetation Coverage Index

**Range:** `[0, 1]`  
**Formula:** `vegetation_percentage / 100`  
**Interpretation:** Green space density indicator  
**Example:** `0.55` = high vegetation coverage (park-like area)

### 3. Sky Openness Index

**Range:** `[0, 1]`  
**Formula:** `sky_percentage / 100`  
**Interpretation:** Open sky visibility (urban canyon effect)  
**Example:** `0.15` = restricted sky view (dense buildings)

### 4. Ground Accessibility Index

**Range:** `[0, 1]`  
**Formula:** `(road_percentage + sidewalk_percentage) / 100`  
**Interpretation:** Combined walkable ground surface  
**Key Feature:** Separates road vs sidewalk internally  
**Example:** `0.35` = moderate ground accessibility

### 5. Human Activity Index

**Range:** `[0, 1]` (scaled down)  
**Formula:** `pedestrian_percentage * 0.01`  
**Interpretation:** Pedestrian presence (filtered from static analysis)  
**Note:** Low-weight indicator, not included in normalization  
**Example:** `0.0089` = minimal human activity

### 6. Vehicle Intensity Index

**Range:** `[0, 1]` (scaled down)  
**Formula:** `(vehicle_4w + vehicle_2w)_percentage * 0.01`  
**Interpretation:** Vehicle traffic density  
**Note:** Separate from static ground analysis  
**Limitation:** Cityscapes doesn't distinguish 4w vs 2w vehicles (estimated ratio)  
**Example:** `0.0958` = low-moderate vehicle intensity

### 7. Traffic Infrastructure Index

**Range:** `[0, 1]`  
**Formula:** `(signage_percentage + street_furniture_percentage) / 100`  
**Interpretation:** Road signage and infrastructure density  
**Example:** `0.0123` = sparse traffic infrastructure

### 8. Heritage Dominance Index ⭐ NEW

**Range:** `[0, 1]`  
**Formula:** `heritage_building_percentage / total_building_percentage`  
**Current Status:** Placeholder (requires custom CNN training)  
**Future:** Detect historical/architectural heritage buildings  
**Example:** `0.1567` = ~16% of buildings are heritage (when trained)

**Implementation Note:** 
- Currently returns 0 unless heritage class is manually annotated
- Requires fine-tuning on heritage building dataset
- See `/docs/HERITAGE_TRAINING_GUIDE.md` (coming soon)

---

## Moving Object Filtering

### What Gets Filtered?

**Excluded from Static Analysis:**
- ✓ Pedestrians (`person`, `rider` classes)
- ✓ Vehicles (`car`, `truck`, `bus`, `motorcycle`, etc.)

**Included in Static Analysis:**
- ✓ Buildings
- ✓ Roads & Sidewalks (separated)
- ✓ Vegetation
- ✓ Sky
- ✓ Signage
- ✓ Street Furniture

### Why Filter Moving Objects?

1. **Focus on Infrastructure:** Analyze permanent structures only
2. **Consistency:** Remove temporal variance (traffic, pedestrians vary by time)
3. **Accurate Ground Metrics:** Road/sidewalk measurement不受移动物体影响
4. **Better Planning Data:** Urban planning focuses on fixed elements

### Normalization Behavior

When moving objects detected:

```
Raw percentages:
- Building: 10%
- Vegetation: 40%
- Pedestrians: 5%  ← Moving
- Vehicles: 10%     ← Moving
- Others: 35%

After filtering:
Total static = 85%
Normalization factor = 100 / 85 = 1.176x

Normalized percentages:
- Building: 11.76%
- Vegetation: 47.06%
- Pedestrians: 5% (reported separately, not normalized)
- Vehicles: 10% (reported separately, not normalized)
```

---

## Method: _compute_metrics() (Original)

### Signature

```python
def _compute_metrics(self, pct_by_class: dict, seg_map: np.ndarray) -> dict
```

### Return Structure

```python
{
    "green_coverage_pct": 45.23,
    "building_coverage_pct": 28.91,
    "walkability_ratio": 0.67,
    "visual_clutter_index": 0.12,
    "sky_visibility_pct": 15.44
}
```

### Metric Formulas

| Metric | Formula | Range | Purpose |
|--------|---------|-------|---------|
| `green_coverage_pct` | vegetation + sky | [0, 100]% | Environmental quality |
| `building_coverage_pct` | building + wall + fence | [0, 100]% | Built density |
| `walkability_ratio` | sidewalk / (sidewalk + road + vehicle) | [0, 1] | Pedestrian friendliness |
| `visual_clutter_index` | signage / (green + 1) | [0, ∞] | Visual pollution |
| `sky_visibility_pct` | sky | [0, 100]% | Urban canyon effect |

**Use Case:** Urban walkability assessment, environmental planning

---

## Method: _compute_excel_metrics()

### Signature

```python
def _compute_excel_metrics(self, pct_by_class: dict, seg_map: np.ndarray) -> dict
```

### Return Structure (Full)

```python
{
    # 8 Composite Indices
    "building_visibility_index": 0.2891,
    "vegetation_coverage_index": 0.4523,
    "sky_openness_index": 0.1544,
    "ground_accessibility_index": 0.2345,
    "human_activity_index": 0.0089,
    "vehicle_intensity_index": 0.0958,
    "traffic_infrastructure_index": 0.0123,
    "heritage_dominance_index": 0.1567,
    
    # Metadata
    "_static_normalization_factor": 1.1234,
    "_moving_object_percentage": 12.47,
    "_raw_percentages": {
        "building": 28.91,
        "vegetation": 45.23,
        "road": 12.34,
        "sidewalk": 11.11,
        "sky": 15.44,
        "signage": 1.23,
        "street_furniture": 2.45,
        "heritage": 3.29
    }
}
```

**Use Case:** Excel-alignment, comprehensive visual analysis, infrastructure mapping

---

## Method: free_memory()

### Signature

```python
def free_memory(self) -> None
```

### Purpose

Unload model from GPU/CPU memory, essential for:
- Batch processing multiple images
- Low-VRAM environments
- Preventing memory leaks in long-running scripts

### Usage Pattern

```python
# Process batch of images
for img_path in image_list:
    result = model.infer(img_path, excel_mode=True)
    process(result)
    model.free_memory()  # Clear memory between batches
    
    # Or keep model loaded for throughput
    # (if VRAM allows)
```

---

## Class Constants

### CLASS_MAP

Mapping from cityscapes 19 classes to 10 categories:

| Cityscapes Class | Category | Notes |
|------------------|----------|-------|
| `road` | `road` | Vehicle traffic surface |
| `sidewalk` | `sidewalk` | Pedestrian walkway |
| `building` | `building` | General buildings |
| `wall` | `building` | Building structure |
| `fence` | `building` | Enclosure |
| `vegetation` | `vegetation` | Plants, grass |
| `tree` | `vegetation` | Trees |
| `sky` | `sky` | Sky visibility |
| `person` | `pedestrian` | Moving object |
| `rider` | `pedestrian` | Cyclist/motorcyclist |
| `car` | `vehicle` | 4-wheeled (40% estimated) |
| `truck` | `vehicle` | Large vehicles (60% estimated) |
| `bus` | `vehicle` | Public transport |
| `train` | `vehicle` | Rail vehicles |
| `motorcycle` | `vehicle` | 2-wheeled |
| `bicycle` | `vehicle` | Non-motorized |
| `signage` | `signage` | Traffic signs |
| `pole` | `street_furniture` | Infrastructure poles |
| `traffic_light` | `signage` | Signal lights |
| `terrain` | `other` | Other surfaces |

### METRIC_CLASSES

List of all 10 metric categories:
```python
[
    "vegetation", "building", "road", "sidewalk", "sky",
    "signage", "vehicle", "pedestrian", "street_furniture", "heritage"
]
```

### EXCEL_INDICATORS

Names of 8 Excel composite indices:
```python
[
    "building_visibility",
    "vegetation_coverage",
    "sky_openness",
    "ground_accessibility",
    "human_activity",
    "vehicle_intensity",
    "traffic_infrastructure",
    "heritage_dominance"
]
```

---

## Error Handling

### CUDA Out of Memory

**Symptoms:** `torch.cuda.OutOfMemoryError`

**Solutions:**
```python
# Option 1: Lower resolution
result = model.infer(image, max_resolution=256)

# Option 2: Enable low-VRAM mode (default)
model = SegformerB5(low_vram_mode=True)

# Option 3: Reduce batch size
batch_size = 1  # Process one image at a time
```

### Model Not Found

**Symptoms:** `OSError: Can't load tokenizer`

**Solution:** Manual download:
```bash
huggingface-cli download nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --local-dir ./models/segformer-b5
```

Then:
```python
model = SegformerB5(model_id="./models/segformer-b5")
```

### Invalid Image Format

**Symptoms:** `ValueError: cannot convert ... to PIL Image`

**Solution:** Ensure valid format:
```python
from PIL import Image
import cv2

# Convert to PIL first
img_cv = cv2.imread("image.jpg")
img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
result = model.infer(img_pil)
```

---

## Integration Examples

### CSV Export

```python
import csv

results = []
for img_path in ["img1.jpg", "img2.jpg", "img3.jpg"]:
    result = model.infer(img_path, excel_mode=True)
    metrics = result["metrics"]
    
    row = {
        "image": img_path,
        "building_visibility_index": metrics["building_visibility_index"],
        "vegetation_coverage_index": metrics["vegetation_coverage_index"],
        # ... other indices
    }
    results.append(row)

# Save to CSV
with open("output.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", *list(results[0].keys())[1:]])
    writer.writeheader()
    writer.writerows(results)
```

### Database Storage (SQLite)

```python
import sqlite3

conn = sqlite3.connect("analysis.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS segmentation_results (
        id INTEGER PRIMARY KEY,
        image_path TEXT,
        building_visibility REAL,
        vegetation_coverage REAL,
        sky_openness REAL,
        ground_accessibility REAL,
        human_activity REAL,
        vehicle_intensity REAL,
        traffic_infrastructure REAL,
        heritage_dominance REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")

for img_path in ["img1.jpg", "img2.jpg"]:
    result = model.infer(img_path, excel_mode=True)
    m = result["metrics"]
    
    cursor.execute("""
        INSERT INTO segmentation_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        img_path, m["building_visibility_index"], m["vegetation_coverage_index"],
        m["sky_openness_index"], m["ground_accessibility_index"],
        m["human_activity_index"], m["vehicle_intensity_index"],
        m["traffic_infrastructure_index"], m["heritage_dominance_index"]
    ))

conn.commit()
conn.close()
```

### Web API (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = SegformerB5(excel_mode=True)

class InferenceRequest(BaseModel):
    image_base64: str
    resolution: int = 512

@app.post("/analyze")
async def analyze_image(request: InferenceRequest):
    # Decode base64 image
    import base64
    img_data = base64.b64decode(request.image_base64)
    img_array = np.frombuffer(img_data, np.uint8)
    img_pil = Image.open(img_io.BytesIO(img_data)).convert("RGB")
    
    # Run inference
    result = model.infer(img_pil, max_resolution=request.resolution, excel_mode=True)
    
    return {
        "indices": result["metrics"],
        "percentages": result["pct_by_class"]
    }
```

---

## Performance Tips

### Optimal Resolution Settings

| Use Case | Resolution | Speed | Accuracy |
|----------|------------|-------|----------|
| Rapid prototyping | 256px | Fast | Low |
| Production batch | 512px | Balanced | Good |
| High-quality analysis | 768px | Slow | High |
| Research-grade | 1024px | Very slow | Maximum |

### Batch Processing Optimization

```python
# Efficient batch processing
image_paths = [...]  # List of paths

results = []
for img_path in image_paths:
    result = model.infer(img_path, max_resolution=512, excel_mode=True)
    results.append(result)
    
    # Optional: Free memory if VRAM limited
    if len(results) % 10 == 0:
        model.free_memory()

# Keep model loaded for best throughput (if VRAM allows)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-09-14 | Initial release with Excel alignment |
| 1.0.1 | 2026-09-14 | Added heritage placeholder, moving object filtering |

---

## Related Documentation

- `/docs/HARDWARE_REQUIREMENTS.md` - Hardware specs for deployment
- `src/uvip_ai/segmentation/examples/excel_mode_usage.py` - Full usage example
- `/README.md` - Project overview

---

**Author:** Qoder AI Assistant  
**Last Updated:** 2026-09-14  
**License:** As per project license
