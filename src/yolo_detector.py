"""
YOLO Detector Implementation (CNN-based, fast)
Supports YOLOv8, YOLOv11, and YOLO with ONNX runtime
"""
import numpy as np
import cv2
from typing import List, Dict, Any
import torch
from ultralytics import YOLO

from .base_detector import BaseDetector, DetectionResult


class YOLODetector(BaseDetector):
    """YOLO-based detector (fast CNN-based approach)"""

    def __init__(self, model_path: str = "yolo11l.pt", conf_threshold: float = 0.3,
                 iou_threshold: float = 0.45, max_det: int = 300,
                 device: str = "cuda", use_onnx: bool = False):
        """
        Args:
            model_path: Path to YOLO model (.pt or .onnx)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_det: Maximum number of detections per image
            device: Device to run on (cuda/cpu)
            use_onnx: Whether to use ONNX runtime (faster inference)
        """
        super().__init__(conf_threshold, iou_threshold, max_det, device)
        self.model_path = model_path
        self.use_onnx = use_onnx or model_path.endswith('.onnx')
        self.model = None
        self.class_names = []
        self.model_name = f"YOLO-{model_path.split('/')[-1]}"

    def load_model(self, model_path: str = None):
        """Load YOLO model"""
        if model_path:
            self.model_path = model_path

        print(f"Loading YOLO model from: {self.model_path}")

        if self.use_onnx:
            # Load ONNX model for faster inference
            self.model = YOLO(self.model_path, task="detect")
            self.model_name = f"YOLO-ONNX-{self.model_path.split('/')[-1]}"
        else:
            # Load PyTorch model
            self.model = YOLO(self.model_path, task="detect")

        # Get class names from model
        self.class_names = self.model.names
        print(f"Model loaded. Classes: {self.class_names}")
        print(f"Model info: {self.get_model_info()}")

        return self

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Perform YOLO detection on a single image

        Args:
            image: Input image in BGR format (from cv2)

        Returns:
            DetectionResult with boxes, scores, and class IDs
        """
        if self.model is None:
            self.load_model()

        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False
        )

        # Extract results
        return self._parse_results(results[0])

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Perform detection on a batch of images"""
        results_list = []
        for img in images:
            results_list.append(self.detect(img))
        return results_list

    def detect_with_tracking(self, image: np.ndarray, persist: bool = True,
                             tracker: str = "bytetrack.yaml") -> DetectionResult:
        """
        Perform YOLO detection with tracking

        Args:
            image: Input image
            persist: Whether to persist tracks across frames
            tracker: Tracker config file or name

        Returns:
            DetectionResult with track IDs
        """
        if self.model is None:
            self.load_model()

        results = self.model.track(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            persist=persist,
            tracker=tracker,
            verbose=False
        )

        return self._parse_results(results[0])

    def _parse_results(self, result) -> DetectionResult:
        """Parse ultralytics result into DetectionResult"""
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionResult(
                boxes=np.zeros((0, 4)),
                scores=np.zeros(0),
                class_ids=np.zeros(0, dtype=int),
                track_ids=None
            )

        # Extract boxes (xyxy format)
        boxes = result.boxes.xyxy.cpu().numpy()

        # Extract scores
        scores = result.boxes.conf.cpu().numpy()

        # Extract class IDs
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Extract track IDs if available
        track_ids = None
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)

        return DetectionResult(boxes, scores, class_ids, track_ids)

    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        if self.model is None:
            self.load_model()
        return list(self.class_names.values())

    def train(self, data_yaml: str, epochs: int = 100, batch: int = 16,
              imgsz: int = 640, **kwargs) -> Dict[str, Any]:
        """
        Fine-tune the YOLO model

        Args:
            data_yaml: Path to data.yaml configuration
            epochs: Number of training epochs
            batch: Batch size
            imgsz: Image size
            **kwargs: Additional training arguments

        Returns:
            Training metrics
        """
        if self.model is None:
            self.load_model()

        print(f"Starting training for {epochs} epochs...")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            **kwargs
        )

        # Return training metrics
        return {
            "model": self.model,
            "results": results
        }

    def export_to_onnx(self, output_path: str, dynamic: bool = False,
                       simplify: bool = True):
        """
        Export model to ONNX format for faster inference

        Args:
            output_path: Output ONNX file path
            dynamic: Enable dynamic axes
            simplify: Simplify model
        """
        if self.model is None:
            self.load_model()

        print(f"Exporting model to ONNX: {output_path}")
        self.model.export(format="onnx", dynamic=dynamic, simplify=simplify)
        print(f"Model exported successfully!")


class YOLOv8Detector(YOLODetector):
    """YOLOv8 specific detector"""

    def __init__(self, model_size: str = "l", **kwargs):
        """
        Args:
            model_size: Model size (n, s, m, l, x)
        """
        model_path = f"yolov8{model_size}.pt"
        super().__init__(model_path=model_path, **kwargs)
        self.model_name = f"YOLOv8-{model_size.upper()}"


class YOLOv11Detector(YOLODetector):
    """YOLOv11 specific detector (latest)"""

    def __init__(self, model_size: str = "l", **kwargs):
        """
        Args:
            model_size: Model size (n, s, m, l, x)
        """
        model_path = f"yolo11{model_size}.pt"
        super().__init__(model_path=model_path, **kwargs)
        self.model_name = f"YOLOv11-{model_size.upper()}"
