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
        # id2label asli model menulis "traffic sign" -> dinormalisasi menjadi
        # "traffic_sign". Tanpa baris ini kelas itu jatuh ke nama mentah dan
        # signage selalu 0 (sign/traffic light tidak pernah masuk indeks).
        "traffic_sign": "signage",
        "terrain": "other",
        # Heritage detection requires additional model/fine-tuning
        # Currently placeholder for future enhancement
    }

    METRIC_CLASSES = [
        "vegetation", "building", "road", "sidewalk", "sky",
        "signage", "vehicle", "pedestrian", "street_furniture", "heritage"
    ]

    # Kuantisasi grayscale untuk GLCM. 32 level cukup agar homogeneity stabil
    # dan matriks 32x32 tetap cepat dihitung per request (~1 ms).
    TEXTURE_LEVELS = 32

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
        # Device TIDAK spesifik merek: cuma butuh torch.cuda.is_available().
        # GTX 10-series (Pascal, sm_61), RTX 20/30/40 (Turing/Ampere/Ada),
        # Tesla, dsb. semuanya masuk lewat jalur yang sama. Probe kernel di
        # bawah menangkap kasus "build torch tidak punya kernel untuk sm_XX
        # kartu ini" (mis. wheel CPU-only, atau GPU lebih baru dari build
        # torch) dan turun ke CPU dengan pesan jelas, bukan crash.
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cuda_probe_error = None
        if resolved == "cuda":
            # Hanya panggilan CUDA yang ada di dalam try ini. Kalau kode lain
            # (mis. typo atribut) ikut masuk, errornya akan tertangkap dan
            # menyamar sebagai "GPU rusak" -> turun ke CPU tanpa sebab jelas.
            try:
                # Probe CUDA with tiny tensor to catch "no kernel image" errors early
                _probe = torch.zeros((1,), device="cuda")
                _probe.cpu()  # force sync
                del _probe
                torch.cuda.synchronize()
            except Exception as cuda_err:
                cuda_probe_error = cuda_err
                resolved = "cpu"
        # dtype ditetapkan SETELAH device final, supaya log di bawah selalu
        # membaca nilai yang sudah ada (dulu dibaca sebelum di-assign ->
        # AttributeError -> tertangkap sebagai "GPU tidak bisa dipakai").
        self._device = resolved
        self._dtype = (torch.float16 if (self._device == "cuda" and low_vram_mode)
                       else torch.float32)
        if self._device == "cuda":
            _gpu = torch.cuda.get_device_name(0)
            _cap = torch.cuda.get_device_capability(0)
            print(f"[SegFormer] GPU terdeteksi: {_gpu} (sm_{_cap[0]}{_cap[1]}). "
                  f"Pakai CUDA, dtype={self._dtype}.")
        elif cuda_probe_error is not None:
            print(f"[SegFormer] GPU ada tapi tidak bisa dipakai "
                  f"({type(cuda_probe_error).__name__}: {cuda_probe_error}). "
                  f"Jalan di CPU (lebih lambat). Cek build torch (+cuXXX) cocok "
                  f"dengan driver/sm_XX kartu ini.")
        else:
            print("[SegFormer] Tidak ada GPU NVIDIA/CUDA -> jalan di CPU "
                  "(lebih lambat). Untuk GPU: install torch +cuXXX lalu restart.")
    def _load(self) -> None:
        if self._model is not None:
            return
        print(f"[SegFormer] Loading model from '{self.model_id}' ...")
        self._processor = SegformerImageProcessor.from_pretrained(self.model_id, ignore_mismatched_sizes=True)
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_id, ignore_mismatched_sizes=True, weights_only=False
        ).to(self._device)
        # Explicitly convert ALL parameters including biases to target dtype
        self._model.to(dtype=self._dtype)
        for name, param in self._model.named_parameters():
            param.to(self._dtype)
        self._model.eval()
        print(f"[SegFormer] Model loaded on {self._device}.")

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    def free_memory(self) -> None:
        """Lepas model & bersihkan cache VRAM (dipakai mode hemat VRAM).

        Dipanggil ``runpod_serverless``, ``DatasetBuilder``, dan contoh
        ``excel_mode_usage`` — tanpa metode ini ketiganya AttributeError.
        """
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        
        # Calculate metrics based on mode.
        # excel_mode butuh seg_map + piksel RGB: indeks penelitian memakai
        # komponen bentuk (solidity), warna (mean RGB bangunan) dan tekstur
        # (GLCM homogeneity) di samping rasio piksel.
        if excel_mode:
            metrics = self._compute_excel_metrics(
                pct_by_class,
                seg_map,
                rgb=np.array(img, dtype=np.uint8, copy=True),
                masks=self._category_masks(seg_map),
            )
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

    @staticmethod
    def _largest_contour(mask: np.ndarray):
        """Kontur terbesar dari mask boolean (None bila kosong)."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @classmethod
    def _shape_stats(cls, mask: np.ndarray, total_px: int) -> tuple[float, float]:
        """(luas kontur terbesar / total piksel, solidity) untuk satu mask."""
        if total_px <= 0 or mask is None or not mask.any():
            return 0.0, 0.0
        contour = cls._largest_contour(mask)
        if contour is None:
            return 0.0, 0.0
        area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        return area / total_px, (area / hull_area if hull_area > 0 else 0.0)

    @classmethod
    def _glcm_homogeneity(cls, gray: np.ndarray, mask: np.ndarray) -> float:
        """Homogeneity GLCM (jarak 1, arah horizontal) DI DALAM mask objek."""
        ys, xs = np.nonzero(mask)
        if ys.size < 64:
            return 0.0
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub = gray[y0:y1, x0:x1]
        inner = mask[y0:y1, x0:x1]
        levels = cls.TEXTURE_LEVELS
        quant = (sub.astype(np.uint16) * levels) // 256
        # Hanya pasangan yang KEDUA pikselnya di dalam objek.
        pair = inner[:, :-1] & inner[:, 1:]
        left = quant[:, :-1][pair]
        right = quant[:, 1:][pair]
        if left.size < 64:
            return 0.0
        counts = np.bincount(
            left.astype(np.int64) * levels + right, minlength=levels * levels
        ).astype(np.float64)
        total = counts.sum()
        if total <= 0:
            return 0.0
        prob = (counts / total).reshape(levels, levels)
        idx = np.arange(levels)
        weight = 1.0 / (1.0 + np.abs(idx[:, None] - idx[None, :]))
        return float((prob * weight).sum())

    @staticmethod
    def _mean_rgb(rgb: np.ndarray | None, mask: np.ndarray | None) -> float:
        """Rata-rata kanal piksel di dalam mask, dinormalisasi ke 0..1."""
        if rgb is None or mask is None or not mask.any():
            return 0.0
        pixels = rgb[mask]
        return float(pixels.mean()) / 255.0 if pixels.size else 0.0

    def _category_masks(self, seg_map: np.ndarray) -> dict[str, np.ndarray]:
        """Mask boolean per kategori analitis (road, building, vegetation, ...)."""
        groups: dict[str, list[int]] = {}
        id2label = getattr(self.model.config, "id2label", None) or {}
        for class_id, label in id2label.items():
            name = str(label).lower().replace(" ", "_")
            groups.setdefault(self.CLASS_MAP.get(name, name), []).append(int(class_id))
        return {cat: np.isin(seg_map, ids) for cat, ids in groups.items()}

    def _compute_excel_metrics(self, pct_by_class: dict, seg_map: np.ndarray,
                               rgb: np.ndarray | None = None,
                               masks: dict[str, np.ndarray] | None = None) -> dict:
        """8 indeks komposit 0..1 sesuai notebook penelitian PVI.

        Rumus (bagian "11. COMPOSITE VISUAL INDICES" di
        ``Photogenic_Value_Index_Multi_Foto.ipynb``)::

            BVI = 0.50*building_ratio + 0.30*building_area + 0.20*building_solidity
            VCI = 0.70*vegetation_ratio + 0.30*vegetation_homogeneity
            SOI = 0.80*sky_ratio + 0.20*sky_homogeneity
            GAI = 0.70*ground_ratio + 0.30*ground_solidity
            HAI = human_ratio
            VII = vehicle_ratio                       (4w + 2w)
            TII = traffic_sign_ratio
            HDI = 0.40*BVI + 0.25*building_solidity
                  + 0.20*building_mean_rgb + 0.15*building_homogeneity

        Dua penyimpangan yang disengaja dari notebook, karena versi notebook
        tidak bisa dipakai apa adanya di runtime:

        1. GLCM dihitung pada piksel DI DALAM mask objek saja. Notebook
           menyalin seluruh kanvas lalu mengisi background dengan 0, sehingga
           pasangan piksel (0,0) mendominasi matriks dan homogeneity hampir
           selalu ~0.95 untuk semua foto - komponen itu tidak punya daya beda.
        2. ``building_area`` = luas kontur terbesar / total piksel (definisi
           notebook), bukan jumlah seluruh piksel kelas bangunan.

        ``masks``/``rgb`` opsional. Tanpa keduanya komponen bentuk, warna dan
        tekstur = 0 dan indeks murni rasio piksel (dipakai jalur video).
        """
        ratio = {
            k: min(max(float(v), 0.0) / 100.0, 1.0)
            for k, v in pct_by_class.items()
        }
        masks = masks or {}
        total_px = int(seg_map.size)
        gray = (
            cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
            if rgb is not None else None
        )
        can_shape = gray is not None

        def shape(cat: str) -> tuple[float, float]:
            mask = masks.get(cat)
            if not can_shape or mask is None:
                return 0.0, 0.0
            return self._shape_stats(mask, total_px)

        def homogeneity(cat: str) -> float:
            mask = masks.get(cat)
            if not can_shape or mask is None:
                return 0.0
            return self._glcm_homogeneity(gray, mask)

        # 1. Building Visibility Index
        building_ratio = ratio.get("building", 0.0)
        building_area, building_solidity = shape("building")
        bvi = (0.50 * building_ratio + 0.30 * building_area
               + 0.20 * building_solidity)

        # 2. Vegetation Coverage Index
        vegetation_ratio = ratio.get("vegetation", 0.0)
        vci = 0.70 * vegetation_ratio + 0.30 * homogeneity("vegetation")

        # 3. Sky Openness Index
        sky_ratio = ratio.get("sky", 0.0)
        soi = 0.80 * sky_ratio + 0.20 * homogeneity("sky")

        # 4. Ground Accessibility Index (road + sidewalk = "ground")
        ground_ratio = min(ratio.get("road", 0.0) + ratio.get("sidewalk", 0.0), 1.0)
        ground_mask = None
        if masks:
            parts = [masks[c] for c in ("road", "sidewalk") if c in masks]
            if parts:
                ground_mask = parts[0] if len(parts) == 1 else np.logical_or(*parts)
        ground_solidity = 0.0
        if can_shape and ground_mask is not None:
            _, ground_solidity = self._shape_stats(ground_mask, total_px)
        gai = 0.70 * ground_ratio + 0.30 * ground_solidity

        # 5. Human Activity Index - DULU selalu 0 karena dibaca dari dict yang
        # hanya berisi kelas statis (pedestrian tidak pernah ada di sana).
        hai = ratio.get("pedestrian", 0.0)

        # 6. Vehicle Intensity Index - DULU = vehicle_pct * 0.51 sehingga bisa
        # bernilai 14x lipat batas atas indeks (0..1).
        vii = ratio.get("vehicle", 0.0)

        # 7. Traffic Infrastructure Index (traffic sign + traffic light + pole)
        tii = min(ratio.get("signage", 0.0) + ratio.get("street_furniture", 0.0), 1.0)

        # 8. Heritage Dominance Index
        building_mean_rgb = self._mean_rgb(rgb, masks.get("building"))
        hdi = (0.40 * bvi + 0.25 * building_solidity
               + 0.20 * building_mean_rgb + 0.15 * homogeneity("building"))

        static_categories = ["building", "vegetation", "sky", "road", "sidewalk",
                             "signage", "street_furniture", "heritage"]
        static_sum = sum(pct_by_class.get(cat, 0) for cat in static_categories)
        moving_total = pct_by_class.get("pedestrian", 0) + pct_by_class.get("vehicle", 0)

        return {
            "building_visibility_index": round(bvi, 4),
            "vegetation_coverage_index": round(vci, 4),
            "sky_openness_index": round(soi, 4),
            "ground_accessibility_index": round(gai, 4),
            "human_activity_index": round(hai, 4),
            "vehicle_intensity_index": round(vii, 4),
            "traffic_infrastructure_index": round(tii, 4),
            "heritage_dominance_index": round(hdi, 4),
            # Metadata audit (tidak dipakai scorer, hanya untuk penelusuran).
            "_static_normalization_factor": round(
                100.0 / static_sum if static_sum > 0 else 1.0, 4),
            "_moving_object_percentage": round(moving_total, 4),
            "_components": {
                "building_area": round(building_area, 4),
                "building_solidity": round(building_solidity, 4),
                "ground_solidity": round(ground_solidity, 4),
                "building_mean_rgb": round(building_mean_rgb, 4),
            },
            "_raw_percentages": {
                cat: round(pct_by_class.get(cat, 0), 4)
                for cat in static_categories
            },
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

