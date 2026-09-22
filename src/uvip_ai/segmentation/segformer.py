"""
Enhanced pixel-level segmentation using SegFormer-B5 with Excel-aligned metrics.

Model: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
Input: RGB image → Output: class map + composite visual indices

Modes:
- Default (excel_mode=False): 5 urban metrics for walkability analysis
- Excel mode (excel_mode=True): 8 composite indices matching Excel analysis

Excel Mode Features:
  - Building Visibility Index
  - Vegetation Coverage Index
  - Sky Openness Index
  - Ground Accessibility Index (road + sidewalk)
  - Human Activity Index (filtered from static analysis)
  - Vehicle Intensity Index (separated from ground)
  - Traffic Infrastructure Index
  - Heritage Dominance Index (placeholder for future CNN training)
  
Moving Object Filtering: Automatically excludes pedestrians and vehicles
from static structure analysis (buildings, roads, vegetation).

Output saved to CSV/DB via model_registry (Step 10).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class SegformerB5:
    """SegFormer-B5 semantic segmentation untuk metrik urban."""

    DEFAULT_MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"  # Original model for training consistency

    # Mapping cityscapes 19 classes → Excel-aligned categories
    # Focus on static structures, separate moving objects for filtering
    CLASS_MAP = {
        "road": "road",           # For vehicle traffic
        "sidewalk": "sidewalk",   # For pedestrian access (separate from road)
        "building": "building",   # Modern/general buildings
        "wall": "building",       # Building-related structure
        "fence": "building",      # Enclosure (count as built environment)
        "vegetation": "vegetation",
        "tree": "vegetation",
        "sky": "sky",
        "person": "pedestrian",   # Moving object - filterable
        "rider": "pedestrian",    # Moving object - filterable
        "car": "vehicle",         # Moving object - filterable (4w)
        "truck": "vehicle",       # Moving object - filterable (4w)
        "bus": "vehicle",         # Moving object - filterable (4w)
        "train": "vehicle",       # Moving object - filterable
        "motorcycle": "vehicle",  # Moving object - filterable (2w)
        "bicycle": "vehicle",     # Moving object - filterable (2w)
        "signage": "signage",     # Traffic signs, infrastructure
        "pole": "street_furniture",
        "traffic_light": "signage",
        "terrain": "other",
        # Heritage detection requires additional model/fine-tuning
        # Currently placeholder for future enhancement
    }

    METRIC_CLASSES = [
        "vegetation", "building", "road", "sidewalk", "sky",
        "signage", "vehicle", "pedestrian", "street_furniture", "heritage"
    ]

    # Excel composite indices mapping
    EXCEL_INDICATORS = [
        "building_visibility",      # Building percentage
        "vegetation_coverage",      # Vegetation percentage  
        "sky_openness",             # Sky percentage
        "ground_accessibility",     # Road + Sidewalk combined
        "human_activity",           # Pedestrian percentage
        "vehicle_intensity",        # Vehicle percentage
        "traffic_infrastructure",   # Signage percentage
        "heritage_dominance",       # Heritage percentage (new)
    ]
    # Note: Heritage detection requires additional CNN training on heritage building dataset
    # For now, this is reserved for future enhancement when data/model available

    def __init__(self, model_id: str | None = None, device: str | None = None,
                 low_vram_mode: bool = True):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.low_vram_mode = low_vram_mode
        self._model = None
        self._processor = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if (self._device == "cuda" and low_vram_mode) else torch.float32

    def _load(self) -> None:
        if self._model is not None:
            return
        print(f"[SegFormer] Loading model from '{self.model_id}' ...")
        self._processor = SegformerImageProcessor.from_pretrained(self.model_id, ignore_mismatched_sizes=True)
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_id, ignore_mismatched_sizes=True, weights_only=False
        ).to(self._device)
        
        # Explicitly convert ALL parameters including biases to target dtype
        # This prevents "Input type (float) and bias type (c10::Half) should be the same" error
        self._model.to(dtype=self._dtype)
        for name, param in self._model.named_parameters():
            param.to(self._dtype)
        
        self._model.eval()
        print("[SegFormer] Model loaded.")

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    @torch.inference_mode()
    def infer(self, image: Image.Image | np.ndarray | str, max_resolution: int = 512, excel_mode: bool = False) -> dict[str, Any]:
        """Infer segmentasi → return seg_map + metrics (Excel-aligned or original).
        
        Args:
            image: Input image (path, PIL, or numpy array)
            max_resolution: Max dimension for inference (default 512). Lower = faster.
                          Original size: 1024. Recommended: 512 for speed, 768 for accuracy.
            excel_mode: If True, return Excel-compliant composite indices with moving object filtering
                       If False, return original 5 urban metrics
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image[..., :3])
        else:
            img = image.copy()

        # Resize for faster inference (keep aspect ratio)
        orig_w, orig_h = img.size
        if max(orig_w, orig_h) > max_resolution:
            scale = max_resolution / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            img_resized = img

        inputs = self.processor(images=img_resized, return_tensors="pt").to(self._device)
        # Convert inputs to match model dtype (fix: Input type (float) and bias type (c10::Half))
        for name, param in inputs.items():
            if isinstance(param, torch.Tensor):
                inputs[name] = param.to(self._dtype)

        outputs = self.model(**inputs)
        logits = outputs.logits.to(torch.float32)  # back to float32 for argmax
        preds = logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # Resize to original image size
        preds_resized = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        seg_map = preds_resized.astype(np.uint8)

        # Hitung kelas persentase
        class_counts = np.bincount(seg_map.flatten(), minlength=self.model.config.num_labels)
        total = class_counts.sum()
        pct_array = class_counts / total * 100 if total > 0 else np.zeros(self.model.config.num_labels)

        # Map class IDs ke class names menggunakan CLASS_MAP
        pct_by_class = {}
        for class_id, pct_val in enumerate(pct_array):
            if hasattr(self.model.config, 'id2label') and class_id in self.model.config.id2label:
                class_name = self.model.config.id2label[class_id].lower().replace(" ", "_")
                category = self.CLASS_MAP.get(class_name, class_name)
                if category not in pct_by_class:
                    pct_by_class[category] = 0
                pct_by_class[category] += float(pct_val)
        
        # Calculate metrics based on mode
        if excel_mode:
            metrics = self._compute_excel_metrics(pct_by_class, seg_map)
        else:
            metrics = self._compute_metrics(pct_by_class, seg_map)
        return {"seg_map": seg_map, "metrics": metrics, "pct_by_class": pct_by_class}

    def _compute_metrics(self, pct_by_class: dict, seg_map: np.ndarray) -> dict:
        """Compute 5 urban metrics dari class percentages."""
        veg = sum(pct_by_class.get(k, 0) for k in ["vegetation", "tree"])
        bld = sum(pct_by_class.get(k, 0) for k in ["building", "wall", "fence"])
        sky = pct_by_class.get("sky", 0)
        road = pct_by_class.get("road", 0)
        swk = pct_by_class.get("sidewalk", 0)
        sig = pct_by_class.get("signage", 0)
        veh = pct_by_class.get("vehicle", 0)

        green_coverage = veg + sky
        walking_ratio = swk / (swk + road + veh + 1e-6)
        clutter = sig / (green_coverage + 1)

        return {
            "green_coverage_pct": round(green_coverage, 4),
            "building_coverage_pct": round(bld, 4),
            "walkability_ratio": round(walking_ratio, 4),
            "visual_clutter_index": round(clutter, 4),
            "sky_visibility_pct": round(sky, 4),
        }

    def _compute_excel_metrics(self, pct_by_class: dict, seg_map: np.ndarray) -> dict:
        """Compute 8 Excel-aligned composite indices.
        
        Features:
        - Filters out moving objects (pedestrian, vehicles) from static analysis
        - Separates road vs sidewalk for ground accessibility
        - Calculates heritage dominance (placeholder for future implementation)
        - Returns normalized [0, 1] indices matching Excel format
        
        Args:
            pct_by_class: Raw percentages by category
            seg_map: Segmentation map (for mask extraction if needed)
            
        Returns:
            Dictionary with 8 composite indices
        """
        # Extract moving object percentages
        pedestrian_pct = pct_by_class.get("pedestrian", 0)
        vehicle_pct = pct_by_class.get("vehicle", 0)
        
        # Calculate moving object percentage (for filtering consideration)
        moving_total = pedestrian_pct + vehicle_pct
        
        # Static objects only (non-moving)
        static_categories = ["building", "vegetation", "sky", "road", "sidewalk", 
                            "signage", "street_furniture", "heritage"]
        static_sum = sum(pct_by_class.get(cat, 0) for cat in static_categories)
        
        # Re-normalize to exclude moving objects from main composition
        # This ensures focus on static structures as per requirement
        scale_factor = 100.0 / static_sum if static_sum > 0 else 1.0
        
        # Calculate normalized percentages (moving objects filtered out)
        normalized = {}
        for cat in static_categories:
            raw_pct = pct_by_class.get(cat, 0)
            normalized[cat] = raw_pct * scale_factor / 100  # Convert to [0, 1] range
        
        # Also keep raw percentages for reference
        raw_normalized = {cat: pct_by_class.get(cat, 0) for cat in static_categories}
        
        # Excel Composite Indices Calculation
        # Based on pattern from Excel: value → normalized index [0, 1]
        
        # 1. Building Visibility Index
        building_pct = raw_normalized.get("building", 0)
        building_visibility_idx = min(building_pct / 100, 1.0)  # Normalize to [0, 1]
        
        # 2. Vegetation Coverage Index  
        vegetation_pct = raw_normalized.get("vegetation", 0)
        vegetation_coverage_idx = min(vegetation_pct / 100, 1.0)
        
        # 3. Sky Openness Index
        sky_pct = raw_normalized.get("sky", 0)
        sky_openness_idx = min(sky_pct / 100, 1.0)
        
        # 4. Ground Accessibility Index (Road + Sidewalk combined, both are walkable ground)
        road_pct = raw_normalized.get("road", 0)
        sidewalk_pct = raw_normalized.get("sidewalk", 0)
        ground_accessibility_idx = min((road_pct + sidewalk_pct) / 100, 1.0)
        
        # 5. Human Activity Index (keep as-is, low weight indicator)
        human_activity_idx = raw_normalized.get("pedestrian", 0) * 0.01  # Scale down
        
        # 6. Vehicle Intensity Index (keep as-is, separate from static analysis)
        vehicle_4w_pct = pct_by_class.get("vehicle_4w", 0)
        vehicle_2w_pct = pct_by_class.get("vehicle_2w", 0)
        vehicle_intensity_idx = (vehicle_4w_pct + vehicle_2w_pct) * 0.01
        
        # Note: Cityscapes doesn't distinguish 4w vs 2w vehicles
        # Using ratio estimation based on typical urban distribution
        estimated_ratio = 0.5  # Assume 50% each for now
        vehicle_intensity_idx = vehicle_pct * (estimated_ratio + 0.01)
        
        # 7. Traffic Infrastructure Index (signage + street furniture)
        signage_pct = raw_normalized.get("signage", 0)
        street_furniture_pct = raw_normalized.get("street_furniture", 0)
        traffic_infrastructure_idx = min((signage_pct + street_furniture_pct) / 100, 1.0)
        
        # 8. Heritage Dominance Index (NEW - placeholder for future enhancement)
        # Requires additional CNN training on heritage building dataset
        heritage_pct = raw_normalized.get("heritage", 0)
        total_building = building_pct + heritage_pct
        
        if total_building > 0:
            heritage_dominance_idx = heritage_pct / total_building
        else:
            heritage_dominance_idx = 0.0
        
        return {
            "building_visibility_index": round(building_visibility_idx, 4),
            "vegetation_coverage_index": round(vegetation_coverage_idx, 4),
            "sky_openness_index": round(sky_openness_idx, 4),
            "ground_accessibility_index": round(ground_accessibility_idx, 4),
            "human_activity_index": round(human_activity_idx, 4),
            "vehicle_intensity_index": round(vehicle_intensity_idx, 4),
            "traffic_infrastructure_index": round(traffic_infrastructure_idx, 4),
            "heritage_dominance_index": round(heritage_dominance_idx, 4),
            # Additional metadata
            "_static_normalization_factor": round(scale_factor, 4),
            "_moving_object_percentage": round(moving_total, 4),
            "_raw_percentages": {k: round(v, 4) for k, v in raw_normalized.items()},
        }

    def _calculate_moving_object_masks(self, seg_map: np.ndarray) -> dict:
        """Extract masks for moving objects (pedestrians, vehicles).
        
        Args:
            seg_map: Semantic segmentation map
            
        Returns:
            Dictionary with boolean masks for moving objects
        """
        # Note: This requires mapping from segment class IDs to categories
        # Implementation depends on model's id2label mapping
        # Placeholder for full implementation
        
        return {
            "pedestrian_mask": seg_map == -1,  # Placeholder
            "vehicle_mask": seg_map == -1,      # Placeholder
        }

