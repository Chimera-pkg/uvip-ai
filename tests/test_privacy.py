"""
Unit Test untuk Modul Privacy Guard.

Test coverage:
- Init dan konfigurasi model
- Blur regions dengan berbagai input
- Deteksi objects (mocked untuk efisiensi)
- Processing single image
- Batch processing
- Error handling

Jalankan dengan:
    pytest tests/test_privacy.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestPrivacyGuardInit:
    """Test untuk inisialisasi PrivacyGuard."""

    def test_init_defaults(self):
        """Test init dengan parameter default."""
        from uvip_ai.privacy.guard import PrivacyGuard

        with patch('uvip_ai.privacy.guard.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            guard = PrivacyGuard()

            assert guard.model_path == "yolov8n.pt"
            assert guard.confidence_threshold == 0.5
            assert guard.iou_threshold == 0.45
            assert guard.PERSON_CLASS_ID == 0
            assert guard.CAR_CLASS_ID == 2
            assert guard.LABEL_FACE == "face"
            assert guard.LABEL_PLATE == "plate"

    def test_init_custom_params(self):
        """Test init dengan parameter custom."""
        from uvip_ai.privacy.guard import PrivacyGuard

        with patch('uvip_ai.privacy.guard.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            guard = PrivacyGuard(
                model_path="custom_model.pt",
                confidence_threshold=0.75,
                iou_threshold=0.3
            )

            assert guard.model_path == "custom_model.pt"
            assert guard.confidence_threshold == 0.75
            assert guard.iou_threshold == 0.3


class TestBlurRegions:
    """Test untuk fungsi blur_regions."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Generate sample image untuk testing."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_blur_with_list_bbox(self, sample_image):
        """Test blur dengan list bounding box."""
        from uvip_ai.privacy.guard import PrivacyGuard

        bboxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
        result = PrivacyGuard.blur_regions(sample_image, bboxes, sigma=3.0)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_blur_with_different_sigma(self, sample_image):
        """Test blur dengan nilai sigma berbeda."""
        from uvip_ai.privacy.guard import PrivacyGuard

        bboxes = [[10, 10, 50, 50]]

        for sigma in [1.0, 5.0, 10.0]:
            result = PrivacyGuard.blur_regions(sample_image, bboxes, sigma=sigma)
            assert result.shape == sample_image.shape

    def test_blur_clamp_coordinates(self, sample_image):
        """Test clamp coordinates di batas gambar."""
        from uvip_ai.privacy.guard import PrivacyGuard

        # Bbox yang keluar dari batas
        bboxes = [[-10, -10, 50, 50], [80, 80, 150, 150]]
        result = PrivacyGuard.blur_regions(sample_image, bboxes)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    def test_blur_empty_bboxes(self, sample_image):
        """Test blur dengan list bbox kosong."""
        from uvip_ai.privacy.guard import PrivacyGuard

        result = PrivacyGuard.blur_regions(sample_image, [])

        # Hasil harus sama dengan input (karena tidak ada blur)
        assert np.array_equal(result, sample_image)

    def test_blur_invalid_bbox_format(self, sample_image):
        """Test blur dengan format bbox invalid."""
        from uvip_ai.privacy.guard import PrivacyGuard

        # Bbox dengan panjang salah
        bboxes = [[10, 10, 50]]  # Hanya 3 elemen

        with pytest.raises(ValueError, match="Invalid bbox"):
            PrivacyGuard.blur_regions(sample_image, bboxes)

    def test_blur_unsupported_input_type(self, sample_image):
        """Test blur dengan tipe input tidak didukung."""
        from uvip_ai.privacy.guard import PrivacyGuard

        with pytest.raises(TypeError, match="Unsupported image type"):
            PrivacyGuard.blur_regions("string_path", [], sigma=5.0)


class TestDetectObjects:
    """Test untuk fungsi deteksi objects."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Generate sample image untuk testing."""
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def test_detect_objects_none_image(self):
        """Test detect dengan image None."""
        from uvip_ai.privacy.guard import PrivacyGuard

        with patch('uvip_ai.privacy.guard.YOLO'):
            guard = PrivacyGuard.__new__(PrivacyGuard)

            with pytest.raises(ValueError, match="Input image cannot be None"):
                guard.detect_objects(None)

    def test_detect_objects_invalid_dimensions(self):
        """Test detect dengan dimensi image salah."""
        from uvip_ai.privacy.guard import PrivacyGuard

        with patch('uvip_ai.privacy.guard.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            guard = PrivacyGuard.__new__(PrivacyGuard)

            # 2D array (grayscale atau wrong shape)
            invalid_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

            with pytest.raises(ValueError, match="at least 3 dimensions"):
                guard.detect_objects(invalid_img)

    @patch('uvip_ai.privacy.guard.YOLO')
    def test_detect_objects_returns_correct_structure(self, mock_yolo_cls, sample_image):
        """Test struktur hasil deteksi."""
        from uvip_ai.privacy.guard import PrivacyGuard

        # Setup mock untuk hasil inference
        mock_result = MagicMock()
        mock_boxes = MagicMock()

        # Mock detections untuk person dan car
        mock_boxes.cls = MagicMock()
        mock_boxes.cls.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([0, 2])))  # person & car

        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([0.8, 0.9])))

        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([
            [100, 100, 200, 200],  # person
            [300, 300, 400, 400]   # car
        ])))

        mock_result.boxes = mock_boxes

        mock_yolo_cls.return_value.predict.return_value = [mock_result]

        guard = PrivacyGuard(confidence_threshold=0.5)
        detections = guard.detect_objects(sample_image)

        assert isinstance(detections, list)
        assert len(detections) >= 1  # Minimal satu detection

        for det in detections:
            assert 'type' in det
            assert 'bbox' in det
            assert 'confidence' in det
            assert det['type'] in ['face', 'plate']
            assert len(det['bbox']) == 4
            assert 0 <= det['confidence'] <= 1

    @patch('uvip_ai.privacy.guard.YOLO')
    def test_detect_objects_low_confidence_filtered(self, mock_yolo_cls, sample_image):
        """Test deteksi dengan threshold tinggi (low confidence ter-filter)."""
        from uvip_ai.privacy.guard import PrivacyGuard

        # Setup mock untuk low confidence detection
        mock_result = MagicMock()
        mock_boxes = MagicMock()

        mock_boxes.cls = MagicMock()
        mock_boxes.cls.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([0])))

        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([0.3])))

        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([
            [100, 100, 200, 200]
        ])))

        mock_result.boxes = mock_boxes

        mock_yolo_cls.return_value.predict.return_value = [mock_result]

        # Gunakan threshold 0.5, detection 0.3 seharusnya ter-filter
        guard = PrivacyGuard(confidence_threshold=0.5)
        detections = guard.detect_objects(sample_image)

        # Seharusnya empty karena confidence < threshold
        assert len(detections) == 0


class TestProcessImage:
    """Test untuk proses single image."""

    @pytest.fixture
    def sample_image_bytes(self, tmp_path) -> Path:
        """Create temporary image file untuk testing."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_file = tmp_path / "test_image.jpg"
        cv2.imwrite(str(test_file), img)
        return test_file

    def test_process_image_with_string_path(self, sample_image_bytes):
        """Test process image dengan string path."""
        from uvip_ai.privacy.guard import PrivacyGuard

        with patch('uvip_ai.privacy.guard.YOLO') as mock_yolo_cls:
            mock_yolo_cls.return_value = MagicMock()

            guard = PrivacyGuard()

            # Temporarily set _model untuk avoid actual loading
            guard._model = MagicMock()
            guard._model.predict.return_value = []

            result = guard.process_image(str(sample_image_bytes))

            assert 'blurred_image' in result
            assert 'detections' in result
            assert 'metadata' in result
            assert isinstance(result['blurred_image'], np.ndarray)
            assert result['detections'] == []

    def test_process_image_with_path_object(self, sample_image_bytes):
        """Test process image dengan Path object."""
        from uvip_ai.privacy.guard import PrivacyGuard

        with patch('uvip_ai.privacy.guard.YOLO'):
            guard = PrivacyGuard()
            guard._model = MagicMock()
            guard._model.predict.return_value = []

            result = guard.process_image(sample_image_bytes)

            assert result['metadata']['original_path'] == str(sample_image_bytes)

    def test_process_image_with_numpy_array(self):
        """Test process image dengan numpy array langsung."""
        from uvip_ai.privacy.guard import PrivacyGuard

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch('uvip_ai.privacy.guard.YOLO'):
            guard = PrivacyGuard()
            guard._model = MagicMock()
            guard._model.predict.return_value = []

            result = guard.process_image(img)

            assert isinstance(result['blurred_image'], np.ndarray)
            assert result['metadata']['original_path'] is None

    def test_process_image_nonexistent_file(self):
        """Test process image dengan file yang tidak ada."""
        from uvip_ai.privacy.guard import PrivacyGuard

        with patch('uvip_ai.privacy.guard.YOLO'):
            guard = PrivacyGuard()

            nonexistent = "/path/to/nonexistent/image.jpg"

            with pytest.raises(FileNotFoundError, match="File not found"):
                guard.process_image(nonexistent)

    def test_process_image_output_structure(self):
        """Test struktur lengkap output process_image."""
        from uvip_ai.privacy.guard import PrivacyGuard

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch('uvip_ai.privacy.guard.YOLO') as mock_yolo_cls:
            mock_result = MagicMock()
            mock_boxes = MagicMock()

            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([0])))
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([0.85])))
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([[100, 100, 200, 200]])))

            mock_result.boxes = mock_boxes
            mock_yolo_cls.return_value.predict.return_value = [mock_result]

            guard = PrivacyGuard()

            result = guard.process_image(img)

            # Check metadata
            meta = result['metadata']
            assert 'original_size' in meta
            assert 'height' in meta['original_size']
            assert 'width' in meta['original_size']
            assert 'total_detections' in meta
            assert 'faces_count' in meta
            assert 'plates_count' in meta
            assert meta['total_detections'] == 1
            assert meta['faces_count'] == 1
            assert meta['plates_count'] == 0

            # Check detections structure
            assert len(result['detections']) == 1
            det = result['detections'][0]
            assert det['type'] == 'face'
            assert len(det['bbox']) == 4
            assert 0.8 < det['confidence'] < 0.9

    def test_process_image_invalid_input_type(self):
        """Test process image dengan tipe input invalid."""
        from uvip_ai.privacy.guard import PrivacyGuard

        with patch('uvip_ai.privacy.guard.YOLO'):
            guard = PrivacyGuard()

            with pytest.raises(TypeError, match="Unsupported input type"):
                guard.process_image(12345)


class TestProcessBatch:
    """Test untuk batch processing."""

    def test_batch_processing_success(self, tmp_path):
        """Test batch processing dengan semua sukses."""
        from uvip_ai.privacy.guard import PrivacyGuard

        # Create multiple temp images
        image_paths = []
        for i in range(3):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_file = tmp_path / f"test_{i}.jpg"
            cv2.imwrite(str(test_file), img)
            image_paths.append(test_file)

        with patch('uvip_ai.privacy.guard.YOLO') as mock_yolo_cls:
            mock_yolo_cls.return_value = MagicMock()
            mock_yolo_cls.return_value.predict.return_value = []

            guard = PrivacyGuard()

            results = guard.process_batch(image_paths)

            assert len(results) == 3
            for r in results:
                assert 'error' not in r
                assert 'blurred_image' in r

    def test_batch_processing_with_failures_skip(self, tmp_path):
        """Test batch processing dengan beberapa gagal, skip_failed=True."""
        from uvip_ai.privacy.guard import PrivacyGuard

        valid_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        valid_file = tmp_path / "valid.jpg"
        cv2.imwrite(str(valid_file), valid_img)

        paths = [valid_file, "/nonexistent/image.jpg"]

        with patch('uvip_ai.privacy.guard.YOLO'):
            guard = PrivacyGuard()

            results = guard.process_batch(paths, skip_failed=True)

            # Hanya yang sukses yang masuk
            assert len(results) == 1
            assert 'error' not in results[0]

    def test_batch_processing_with_failures_keep(self, tmp_path):
        """Test batch processing dengan beberapa gagal, skip_failed=False."""
        from uvip_ai.privacy.guard import PrivacyGuard

        valid_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        valid_file = tmp_path / "valid.jpg"
        cv2.imwrite(str(valid_file), valid_img)

        paths = [valid_file, "/nonexistent/image.jpg"]

        with patch('uvip_ai.privacy.guard.YOLO'):
            guard = PrivacyGuard()

            results = guard.process_batch(paths, skip_failed=False)

            assert len(results) == 2
            assert results[0]['metadata']['original_path'] == str(valid_file)
            assert 'error' in results[1]
            assert '/nonexistent/image.jpg' in results[1].get('path', '')


class TestSaveResult:
    """Test untuk penyimpanan hasil."""

    def test_save_result_success(self, tmp_path):
        """Test save result ke file."""
        from uvip_ai.privacy.guard import PrivacyGuard

        guard = PrivacyGuard()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = {
            'blurred_image': img,
            'detections': [],
            'metadata': {}
        }

        output_path = tmp_path / "output.jpg"
        success = guard.save_result(result, output_path)

        assert success is True
        assert output_path.exists()

    def test_save_result_creates_directory(self, tmp_path):
        """Test save result membuat direktori jika belum ada."""
        from uvip_ai.privacy.guard import PrivacyGuard

        guard = PrivacyGuard()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = {'blurred_image': img, 'detections': [], 'metadata': {}}

        nested_dir = tmp_path / "nested" / "dir" / "output.jpg"
        success = guard.save_result(result, nested_dir)

        assert success is True
        assert nested_dir.exists()

    def test_save_result_quality_param(self, tmp_path):
        """Test save result dengan parameter quality."""
        from uvip_ai.privacy.guard import PrivacyGuard

        guard = PrivacyGuard()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = {'blurred_image': img, 'detections': [], 'metadata': {}}

        # Test dengan quality rendah vs tinggi
        path_low = tmp_path / "low_quality.jpg"
        path_high = tmp_path / "high_quality.jpg"

        guard.save_result(result, path_low, quality=50)
        guard.save_result(result, path_high, quality=95)

        assert path_low.exists()
        assert path_high.exists()
        # File dengan quality lebih tinggi seharusnya lebih besar
        assert path_high.stat().st_size >= path_low.stat().st_size


class TestConvenienceFunctions:
    """Test convenience functions di module level."""

    def test_blur_regions_function(self, sample_image):
        """Test function blur_regions standalone."""
        from uvip_ai.privacy.guard import blur_regions

        bboxes = [[10, 10, 50, 50]]
        result = blur_regions(sample_image, bboxes, sigma=3.0)

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    @patch('uvip_ai.privacy.guard.PrivacyGuard')
    def test_process_single_function(self, mock_guard_cls, sample_image):
        """Test function process_single standalone."""
        from uvip_ai.privacy.guard import process_single

        mock_instance = MagicMock()
        mock_instance.process_image.return_value = {
            'blurred_image': sample_image,
            'detections': [],
            'metadata': {'original_path': None}
        }
        mock_guard_cls.return_value = mock_instance

        result = process_single(sample_image, model_path="test.pt")

        mock_guard_cls.assert_called_once()
        mock_instance.process_image.assert_called_once()
        assert result == mock_instance.process_image.return_value


class TestIntegration:
    """Test integrasi end-to-end."""

    def test_end_to_end_processing(self, tmp_path):
        """Test pipeline lengkap dari load sampai save."""
        from uvip_ai.privacy.guard import PrivacyGuard

        # Buat test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        input_path = tmp_path / "input_test.jpg"
        cv2.imwrite(str(input_path), img)

        output_path = tmp_path / "output_test.jpg"

        with patch('uvip_ai.privacy.guard.YOLO') as mock_yolo_cls:
            # Setup mock detection
            mock_result = MagicMock()
            mock_boxes = MagicMock()

            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([0, 2])))

            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([0.8, 0.9])))

            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu = MagicMock(return_value=MagicMock(numpy=lambda: np.array([
                [100, 100, 200, 200],
                [300, 300, 400, 400]
            ])))

            mock_result.boxes = mock_boxes
            mock_yolo_cls.return_value.predict.return_value = [mock_result]

            # Execute full pipeline
            guard = PrivacyGuard()
            result = guard.process_image(input_path)

            # Save
            guard.save_result(result, output_path)

            # Verify
            assert output_path.exists()
            saved_img = cv2.imread(str(output_path))
            assert saved_img is not None
            assert saved_img.shape[:2] == img.shape[:2]

            # Verify detections
            assert len(result['detections']) == 2
            types = [d['type'] for d in result['detections']]
            assert 'face' in types
            assert 'plate' in types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
