"""
Modul Privacy Guard untuk deteksi dan masking wajah & plat nomor menggunakan YOLOv8n.

Modul ini menyediakan fungsi untuk:
- Memuat model YOLOv8n dari ultralytics
- Mendeteksi objek kelas person (wajah) dan car (untuk plate detection)
- Menerapkan Gaussian blur pada region yang terdeteksi
- Support batch inference untuk multiple images

Contoh Penggunaan:
    from uvip_ai.privacy.guard import PrivacyGuard

    guard = PrivacyGuard()
    result = guard.process_image("path/to/image.jpg")

    for box in result["detections"]:
        print(f"{box['type']}: {box['bbox']} ({box['confidence']:.2f})")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from ultralytics import YOLO

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class PrivacyGuard:
    """
    Kelas untuk melakukan privasi pada gambar dengan mendeteksi dan memblur wajah & plat nomor.

    Menggunakan model YOLOv8n untuk mendeteksi:
    - Person (wajah/orang)
    - Car (untuk deteksi plat nomor)

    Atribut:
        model: Model YOLO yang telah dimuat
        person_classes: ID kelas untuk person (default: 0 dalam COCO dataset)
        car_classes: ID kelas untuk car (default: 2 dalam COCO dataset)

    Contoh:
        >>> guard = PrivacyGuard(model_path="yolov8n.pt")
        >>> result = guard.process_image("foto.jpg")
        >>> blurred_img = result["blurred_image"]
        >>> detections = result["detections"]
    """

    # Mapping tipe deteksi ke kelas COCO
    PERSON_CLASS_ID = 0  # 'person' dalam COCO
    CAR_CLASS_ID = 2     # 'car' dalam COCO (mencakup mobil, truck, bus, dll)

    # Label untuk hasil deteksi
    LABEL_FACE = "face"
    LABEL_PLATE = "plate"

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ) -> None:
        """
        Inisialisasi PrivacyGuard dengan model YOLO.

        Args:
            model_path: Path ke file weights model YOLOv8n.
                       Default: 'yolov8n.pt' (akan di-download jika tidak ada).
            confidence_threshold: Threshold confidence untuk filtering detections.
            iou_threshold: Threshold IoU untuk NMS.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        logger.info("Memuat model YOLO dari %s", model_path)
        self._model: YOLO = YOLO(model_path)
        logger.info("Model berhasil dimuat: %s", self._model)

    @property
    def model(self) -> YOLO:
        """Model YOLO yang telah dimuat."""
        return self._model

    @staticmethod
    def blur_regions(
        image: Union[np.ndarray, Path],
        bboxes: List[List[float]],
        sigma: float = 5.0
    ) -> np.ndarray:
        """
        Terapkan Gaussian blur pada region-region tertentu dalam gambar.

        Metode ini mengiterasi setiap bounding box dan menerapkan blur
        Gaussian pada ROI yang didefinisikan oleh bbox tersebut.

        Args:
            image: Input image dalam bentuk numpy array atau path file.
                   Format expected: BGR (OpenCV default).
            bboxes: List of bounding boxes dalam format [x1, y1, x2, y2].
                    Koordinat harus dalam pixel relatif terhadap ukuran gambar.
            sigma: Standar deviasi untuk filter Gaussian.
                   Nilai lebih besar = blur lebih kuat. Default: 5.0.

        Returns:
            Image dengan blur applied pada region yang specified.

        Raises:
            ValueError: Jika bboxes memiliki format yang tidak valid.
            FileNotFoundError: Jika image adalah path dan file tidak ditemukan.

        Contoh:
            >>> img = cv2.imread("input.jpg")
            >>> boxes = [[100, 100, 200, 300], [300, 200, 400, 350]]
            >>> blurred = PrivacyGuard.blur_regions(img, boxes, sigma=10)
        """
        # Load image jika diberikan sebagai path
        if isinstance(image, Path):
            if not image.exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            image_array = cv2.imread(str(image))
        elif isinstance(image, str):
            image_array = cv2.imread(image)
            if image_array is None:
                raise ValueError(f"Cannot load image: {image}")
        elif isinstance(image, np.ndarray):
            image_array = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if image_array is None:
            raise ValueError("Failed to load image")

        original_shape = image_array.shape

        for bbox in bboxes:
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox format: {bbox}. Expected [x1, y1, x2, y2]")

            x1, y1, x2, y2 = map(int, bbox)

            # Clamp coordinates to image bounds
            x1 = max(0, min(x1, original_shape[1] - 1))
            y1 = max(0, min(y1, original_shape[0] - 1))
            x2 = max(x1 + 1, min(x2, original_shape[1]))
            y2 = max(y1 + 1, min(y2, original_shape[0]))

            # Extract ROI
            roi = image_array[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Apply Gaussian blur
            # Kernel size dihitung berdasarkan sigma (harus odd)
            kernel_size = int(2 * round(0.5 * 4 * sigma / 5) + 1) or 3
            kernel_size = min(kernel_size, max(roi.shape[0], roi.shape[1]) // 2 * 2 + 1)

            blurred_roi = cv2.GaussianBlur(
                roi,
                ksize=(kernel_size, kernel_size),
                sigmaX=sigma,
                sigmaY=sigma
            )

            # Replace ROI in original image
            image_array[y1:y2, x1:x2] = blurred_roi

        return image_array

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Deteksi objek wajah dan mobil dalam gambar menggunakan YOLO.

        Args:
            image: Input image dalam format BGR (numpy array).

        Returns:
            List of dictionaries containing detection results:
                - type: 'face' untuk person, 'plate' untuk car
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: confidence score (0-1)

        Example:
            >>> img = cv2.imread("photo.jpg")
            >>> detections = guard.detect_objects(img)
            >>> for d in detections:
            ...     print(f"{d['type']}: {d['bbox']} @ {d['confidence']:.2f}")
        """
        if image is None:
            raise ValueError("Input image cannot be None")

        if len(image.shape) < 3:
            raise ValueError("Image must have at least 3 dimensions (H, W, C)")

        # Run inference
        results = self._model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            stream=True
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy().tolist()
            xyxy = boxes.xyxy.cpu().numpy()

            for cls, conf, bbox in zip(classes, confidences, xyxy):
                if cls == self.PERSON_CLASS_ID:
                    det_type = self.LABEL_FACE
                elif cls == self.CAR_CLASS_ID:
                    det_type = self.LABEL_PLATE
                else:
                    continue  # Skip other classes

                detections.append({
                    'type': det_type,
                    'bbox': bbox.tolist(),
                    'confidence': float(conf)
                })

        logger.debug("Detected %d objects: %d faces, %d plates",
                     len(detections),
                     sum(1 for d in detections if d['type'] == self.LABEL_FACE),
                     sum(1 for d in detections if d['type'] == self.LABEL_PLATE))

        return detections

    def process_image(self, image_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Proses single image untuk privasi (blur wajah & plat nomor).

        Pipeline:
        1. Detect objects (person/car)
        2. Ekstrak bounding boxes
        3. Apply Gaussian blur pada regions
        4. Return hasil processing

        Args:
            image_input: Path gambar atau numpy array image.

        Returns:
            Dictionary berisi:
                - blurred_image: Gambar yang telah di-blur (numpy array)
                - detections: List of detected objects dengan info lengkap
                - metadata: Info tambahan seperti path asli, ukuran gambar

        Raises:
            FileNotFoundError: Jika image path tidak valid
            ValueError: Jika image invalid

        Example:
            >>> result = guard.process_image("secret_photo.jpg")
            >>> cv2.imwrite("masked.jpg", result["blurred_image"])
        """
        logger.info("Processing image: %s",
                   str(image_input) if isinstance(image_input, (str, Path)) else "array input")

        # Load image jika path
        if isinstance(image_input, (Path, str)):
            if isinstance(image_input, Path):
                image_input_str = str(image_input)
                if not image_input.exists():
                    raise FileNotFoundError(f"File not found: {image_input}")
            else:
                image_input_str = image_input

            image = cv2.imread(image_input_str)
            if image is None:
                raise ValueError(f"Cannot read image: {image_input_str}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
            image_input_str = "array"
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        original_h, original_w = image.shape[:2]

        # Detect objects
        detections = self.detect_objects(image)

        # Extract bboxes untuk blur
        bboxes = [d['bbox'] for d in detections]

        # Apply blur pada regions
        blurred = self.blur_regions(image, bboxes, sigma=5.0)

        result = {
            'blurred_image': blurred,
            'detections': detections,
            'metadata': {
                'original_path': str(image_input) if isinstance(image_input, (str, Path)) else None,
                'original_size': {'height': original_h, 'width': original_w},
                'total_detections': len(detections),
                'faces_count': sum(1 for d in detections if d['type'] == self.LABEL_FACE),
                'plates_count': sum(1 for d in detections if d['type'] == self.LABEL_PLATE)
            }
        }

        logger.info("Processed %s: %d faces, %d plates",
                   image_input_str, result['metadata']['faces_count'],
                   result['metadata']['plates_count'])

        return result

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        skip_failed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch processing untuk multiple images.

        Args:
            image_paths: List of paths ke gambar yang akan diproses.
            skip_failed: Jika True, lewati gambar yang gagal diproses.
                        Jika False, include error info di result.

        Returns:
            List of result dictionaries dari process_image().
            Untuk gambar yang gagal, jika skip_failed=False, akan ada entry
            dengan key 'error' berisi pesan error.

        Raises:
            No exceptions jika skip_failed=True.

        Example:
            >>> results = guard.process_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
            >>> for r in results:
            ...     print(r['metadata']['original_path'], r['metadata']['total_detections'])
        """
        logger.info("Batch processing %d images", len(image_paths))

        results = []

        for idx, path in enumerate(image_paths):
            try:
                logger.debug("Processing batch item %d/%d: %s",
                            idx + 1, len(image_paths), path)

                result = self.process_image(path)
                results.append(result)

            except Exception as e:
                error_info = {
                    'error': str(e),
                    'path': str(path),
                    'index': idx
                }

                logger.warning("Failed to process %s: %s", path, str(e))

                if not skip_failed:
                    results.append(error_info)

        logger.info("Batch completed: %d succeeded, %d failed",
                   len(results), len(image_paths) - len(results))

        return results

    def save_result(
        self,
        result: Dict[str, Any],
        output_path: Union[str, Path],
        quality: int = 95
    ) -> bool:
        """
        Simpan hasil processing ke file.

        Args:
            result: Result dictionary dari process_image().
            output_path: Path untuk menyimpan hasil.
            quality: Kualitas JPEG (1-100). Default: 95.

        Returns:
            True jika berhasil, False jika gagal.

        Example:
            >>> success = guard.save_result(result, "output/blurred.jpg")
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            img = result['blurred_image']

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success = cv2.imwrite(str(output_path), img, encode_params)

            if success:
                logger.info("Saved result to %s", output_path)
            else:
                logger.error("Failed to save image to %s", output_path)

            return success

        except Exception as e:
            logger.error("Error saving result: %s", str(e))
            return False


def blur_regions(
    image: Union[np.ndarray, Path, str],
    bboxes: List[List[float]],
    sigma: float = 5.0
) -> np.ndarray:
    """
    Function standalone untuk blur regions pada image.

    Convenience function yang memanggil method static PrivacyGuard.blur_regions.
    Berguna tanpa perlu instantiate class.

    Args:
        image: Input image (numpy array, Path, atau string path).
        bboxes: List of bounding boxes [x1, y1, x2, y2].
        sigma: Sigma untuk Gaussian blur.

    Returns:
        Blurred image as numpy array.

    Example:
        >>> img = cv2.imread("photo.jpg")
        >>> boxes = [[100, 100, 200, 300]]
        >>> blurred = blur_regions(img, boxes)
        >>> cv2.imwrite("out.jpg", blurred)
    """
    return PrivacyGuard.blur_regions(image, bboxes, sigma)


def process_single(
    image_input: Union[str, Path, np.ndarray],
    model_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Function convenience untuk processing single image.

    Instantiates PrivacyGuard dan process single image.
    Berguna untuk quick usage tanpa setup manual.

    Args:
        image_input: Image path atau array.
        model_path: Path ke YOLO model.
        confidence_threshold: Confidence threshold.

    Returns:
        Result dictionary dari PrivacyGuard.process_image().

    Example:
        >>> result = process_single("secret.jpg")
        >>> cv2.imwrite("masked.jpg", result['blurred_image'])
    """
    guard = PrivacyGuard(model_path=model_path, confidence_threshold=confidence_threshold)
    return guard.process_image(image_input)


if __name__ == "__main__":
    """
    Contoh penggunaan modul PrivacyGuard.

    Script ini dapat dijalankan langsung untuk demonstrasi:
    python -m src.uvip_ai.privacy.guard
    """
    import argparse
    import sys
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Privacy Guard - Deteksi dan blur wajah & plat nomor menggunakan YOLOv8n"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Path ke satu atau lebih gambar untuk diproses"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path ke model YOLO (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_privacy",
        help="Direktori untuk menyimpan hasil (default: output_privacy)"
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Tidak download model otomatis jika tidak ada"
    )
    parser.add_argument(
        "--show-detections",
        action="store_true",
        help="Tampilkan bounding boxes pada output image"
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate PrivacyGuard
    logger.info("=" * 60)
    logger.info("PRIVACY GUARD - YOLOv8n Face & License Plate Masking")
    logger.info("=" * 60)

    try:
        guard = PrivacyGuard(model_path=args.model)

    except Exception as e:
        logger.error("Gagal memuat model: %s", str(e))
        logger.info("Pastikan model sudah di-download atau path benar")
        sys.exit(1)

    # Process images
    results = guard.process_batch(args.images, skip_failed=False)

    successful = 0
    failed = 0

    for idx, result in enumerate(results):
        if 'error' in result:
            failed += 1
            logger.error("Gagal: %s - %s", result.get('path'), result['error'])
            continue

        successful += 1

        # Generate output filename
        input_path = Path(result['metadata'].get('original_path', f'image_{idx}'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = input_path.stem
        suffix = input_path.suffix or ".jpg"
        output_filename = f"{stem}_privacy_{timestamp}{suffix}"
        output_path = output_dir / output_filename

        # Draw bounding boxes jika diminta
        if args.show_detections:
            img_display = result['blurred_image'].copy()
            for det in result['detections']:
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = f"{det['type']}:{det['confidence']:.2f}"
                color = (0, 255, 0) if det['type'] == 'face' else (0, 0, 255)
                cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_display, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imwrite(str(output_path), img_display)
            logger.info("Output (with boxes): %s", output_path)

        # Save result
        guard.save_result(result, output_path)

        # Print summary
        meta = result['metadata']
        logger.info("-" * 40)
        logger.info("Input:  %s (%dx%d)",
                   input_path.name, meta['original_size']['width'],
                   meta['original_size']['height'])
        logger.info("Faces:  %d | Plates: %d | Total: %d",
                   meta['faces_count'], meta['plates_count'],
                   meta['total_detections'])
        logger.info("Output: %s", output_path)

    logger.info("=" * 60)
    logger.info("Summary: %d succeeded, %d failed", successful, failed)
    logger.info("=" * 60)
