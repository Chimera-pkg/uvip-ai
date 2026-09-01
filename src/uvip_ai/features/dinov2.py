"""
Feature Extraction — DINOv2 (Step 4).

Model: facebook/dinov2-{small|base|large} → embedding per image.
Tidak fine-tuning, hanya inference pretrained. Output: vektor embedding
yang digabung dengan metrik segmentasi untuk input XGBoost.

Default: dinov2-small untuk CPU-only VPS (hemat RAM ~500MB).
Untuk GPU: bisa pakai dinov2-large (1024-d) via env DINOV2_MODEL.

Embedding dimensions per variant:
  - small:  384-d  (~500MB RAM)
  - base:   768-d  (~1.5GB RAM)
  - large:  1024-d (~4GB RAM)
  - giant:  1536-d (~8GB RAM)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

logger = logging.getLogger(__name__)

# Mapping model → embedding dimension
EMBED_DIMS = {
    "facebook/dinov2-small": 384,
    "facebook/dinov2-base": 768,
    "facebook/dinov2-large": 1024,
    "facebook/dinov2-giant": 1536,
}


class Dinov2Extractor:
    """DINOv2 feature extractor — model variant configurable via config/settings."""

    DEFAULT_MODEL_ID = "facebook/dinov2-small"
    PATCH_SIZE = 14

    def __init__(self, model_id: str | None = None, device: str | None = None,
                 low_vram_mode: bool = True):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.embed_dim = EMBED_DIMS.get(self.model_id, 384)
        self.low_vram_mode = low_vram_mode
        self._model = None
        self._processor = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if (self._device == "cuda" and low_vram_mode) else torch.float32
        logger.info("DINOv2 model: %s (embed_dim=%d, device=%s)",
                     self.model_id, self.embed_dim, self._device)

    def _load(self) -> None:
        if self._model is not None:
            return
        print(f"[DINOv2] Loading model from '{self.model_id}' ...")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = Dinov2Model.from_pretrained(
            self.model_id, ignore_mismatched_sizes=True
        ).to(self._device).to(dtype=self._dtype).eval()
        # Freeze all params
        for p in self._model.parameters():
            p.requires_grad_(False)
        print("[DINOv2] Model loaded.")

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    @torch.inference_mode()
    def extract(self, image: Image.Image | np.ndarray | str) -> np.ndarray:
        """Extract 1024-d embedding dari gambar."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image[..., :3])
        else:
            img = image.copy()

        inputs = self.processor(images=img, return_tensors="pt").to(self._device)
        with torch.autocast(device_type=self._device, dtype=self._dtype):
            outputs = self.model(**inputs)
        hidden_state = outputs.last_hidden_state.to(torch.float32)  # back to float32
        # Global average pooling over patch tokens + class token (dim 1)
        embeddings = hidden_state.mean(dim=1)  # [B, embed_dim]
        emb = F.normalize(embeddings[0], p=2, dim=-1).cpu().numpy()
        return emb.astype(np.float32)

    def extract_batch(self, images: list[str]) -> np.ndarray:
        """Batch extraction dengan chunking untuk hemat VRAM."""
        batch_size = 1 if self.low_vram_mode else 4
        all_embs = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i+batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in chunk]
            embedded = []
            for img in pil_imgs:
                emb = self.extract(img)
                embedded.append(emb)
            all_embs.extend(embedded)
            if self.low_vram_mode:
                del embedded; torch.cuda.empty_cache()
        return np.stack(all_embs, axis=0)

    def free_memory(self) -> None:
        """Unload model & clear GPU cache."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            self._processor = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    emb_model = Dinov2Extractor(low_vram_mode=True)
    img_path = Path("data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg")
    if img_path.exists():
        emb = emb_model.extract(str(img_path))
        print("Embedding shape:", emb.shape, "norm:", np.linalg.norm(emb))
    emb_model.free_memory()
