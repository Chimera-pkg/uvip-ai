"""
Segmentasi pixel-level menggunakan SegFormer-B5 (Step 3).

Model: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
Input: gambar RGB → Output: class map + 5 metrik urban:
  - green_coverage_pct        (vegetasi + sky visibility)
  - building_coverage_pct     (building coverage)
  - walkability_ratio         = sidewalk_pct / (sidewalk_pct + road_pct + vehicle_pct)
  - visual_clutter_index      = signage_pct / (green_coverage_pct + 1)
  - sky_visibility_pct        (sky segment)

Output disimpan ke CSV/DB via model_registry (Step 10).
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

    DEFAULT_MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

    # Mapping cityscapes 19 classes → 5 metrik utama
    CLASS_MAP = {
        "road": "road",
        "sidewalk": "sidewalk",
        "building": "building",
        "wall": "building",
        "fence": "building",
        "vegetation": "vegetation",
        "tree": "vegetation",
        "sky": "sky",
        "person": "pedestrian",
        "rider": "pedestrian",
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "train": "vehicle",
        "motorcycle": "vehicle",
        "bicycle": "vehicle",
        "signage": "signage",  # traffic sign
        "pole": "street_furniture",
        "traffic_light": "signage",
        "terrain": "other",
    }

    METRIC_CLASSES = ["vegetation", "building", "road", "sidewalk", "sky",
                      "signage", "vehicle", "pedestrian", "street_furniture"]

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
        self._processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_id, ignore_mismatched_sizes=True
        ).to(self._device).to(dtype=self._dtype).eval()
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
    def infer(self, image: Image.Image | np.ndarray | str) -> dict[str, Any]:
        """Infer segmentasi → return seg_map + 5 metrik urban."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image[..., :3])
        else:
            img = image.copy()

        inputs = self.processor(images=img, return_tensors="pt").to(self._device)
        outputs = self.model(**inputs)
        logits = outputs.logits.to(torch.float32)  # back to float32 for argmax
        preds = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Resize to original image size
        orig_w, orig_h = img.size  # PIL size is (width, height)
        preds_resized = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        seg_map = preds_resized.astype(np.uint8)

        # Hitung kelas persentase
        class_counts = np.bincount(seg_map.flatten(), minlength=self.model.config.num_labels)
        total = class_counts.sum()
        pct = class_counts / total * 100 if total > 0 else {}

        metrics = self._compute_metrics(pct, seg_map)
        return {"seg_map": seg_map, "metrics": metrics, "pct_by_class": dict(pct)}

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

    def free_memory(self) -> None:
        """Unload model & clear GPU cache — penting untuk low_vram_mode."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            self._processor = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    import cv2
    model = SegformerB5(low_vram_mode=True)
    img = cv2.imread(str(Path("data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg")))
    if img is None:
        print("Image not found; create sample test image")
        img = np.zeros((512, 512, 3), dtype=np.uint8)
    result = model.infer(img)
    print("Metrics:", json.dumps(result["metrics"], indent=2))
    model.free_memory()
