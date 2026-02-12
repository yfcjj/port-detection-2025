"""
DETR (DEtection TRansformer) Detector Implementation
Transformer-based detector, slower than YOLO but potentially more accurate
"""
import numpy as np
import cv2
import torch
from typing import List, Dict, Any, Optional

from .base_detector import BaseDetector, DetectionResult


class DETRDetector(BaseDetector):
    """
    DETR-based detector (Transformer-based approach)

    Supports:
    - Facebook DETR (ResNet backbone)
    - RT-DETR (Real-Time DETR)
    - Conditional DETR
    """

    def __init__(self, model_name: str = "facebook/detr-resnet-101",
                 conf_threshold: float = 0.5, iou_threshold: float = 0.5,
                 max_det: int = 300, device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace model name or path
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_det: Maximum detections per image
            device: Device to run on
        """
        super().__init__(conf_threshold, iou_threshold, max_det, device)
        self.model_name = f"DETR-{model_name.split('/')[-1]}"
        self.model_path = model_name
        self.model = None
        self.processor = None
        self.class_descriptions = []

    def load_model(self, model_path: str = None):
        """Load DETR model from HuggingFace or local path"""
        try:
            from transformers import AutoModelForObjectDetection, AutoImageProcessor
        except ImportError:
            raise ImportError(
                "transformers library required for DETR. "
                "Install with: pip install transformers accelerate"
            )

        if model_path:
            self.model_path = model_path

        print(f"Loading DETR model from: {self.model_path}")

        # Load model and processor
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_path,
            revision="no_timm"
        ).to(self.device)

        self.processor = AutoImageProcessor.from_pretrained(self.model_path)

        # Set model to evaluation mode
        self.model.eval()

        print(f"DETR model loaded. Device: {self.device}")
        return self

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Run DETR detection on image

        Args:
            image: Input image (BGR format from cv2)

        Returns:
            DetectionResult
        """
        if self.model is None:
            self.load_model()

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        return self._parse_outputs(outputs, image.shape[:2])

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Run detection on batch of images"""
        results_list = []
        for img in images:
            results_list.append(self.detect(img))
        return results_list

    def _parse_outputs(self, outputs, img_shape) -> DetectionResult:
        """Parse DETR model outputs"""
        # Get original image size
        target_sizes = torch.tensor([img_shape[::-1]]).to(self.device)

        # Post-process to get detections
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.conf_threshold
        )[0]

        # Extract boxes, scores, and labels
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        class_ids = results['labels'].cpu().numpy().astype(int)

        # Convert to xyxy format if needed
        if boxes.shape[1] == 4:
            # Already in xyxy format
            pass
        else:
            # Convert from xywh to xyxy
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0]  # x1
            boxes_xyxy[:, 1] = boxes[:, 1]  # y1
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2
            boxes = boxes_xyxy

        # Filter by max_det
        if len(scores) > self.max_det:
            top_indices = np.argsort(scores)[-self.max_det:][::-1]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            class_ids = class_ids[top_indices]

        return DetectionResult(boxes, scores, class_ids)

    def train(self, train_dataset, **kwargs):
        """
        Fine-tune DETR model (requires custom training loop)

        Args:
            train_dataset: Training dataset
            **kwargs: Training arguments
        """
        raise NotImplementedError(
            "DETR fine-tuning requires custom implementation. "
            "Consider using RT-DETR for easier training."
        )


class RTDETRDetector(BaseDetector):
    """
    Real-Time DETR Detector (faster than original DETR)
    Uses ultralytics implementation for consistency with YOLO
    """

    def __init__(self, model_path: str = "rtdetr-l.pt",
                 conf_threshold: float = 0.5, iou_threshold: float = 0.45,
                 max_det: int = 300, device: str = "cuda"):
        """
        Args:
            model_path: Path to RT-DETR model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            max_det: Maximum detections
            device: Device to run on
        """
        super().__init__(conf_threshold, iou_threshold, max_det, device)
        self.model_path = model_path
        self.model = None
        self.model_name = f"RT-DETR-{model_path.split('/')[-1]}"

    def load_model(self, model_path: str = None):
        """Load RT-DETR model"""
        from ultralytics import RTDETR

        if model_path:
            self.model_path = model_path

        print(f"Loading RT-DETR model from: {self.model_path}")
        self.model = RTDETR(self.model_path)
        print(f"RT-DETR model loaded. Device: {self.device}")
        return self

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run RT-DETR detection"""
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

        return self._parse_results(results[0])

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Run detection on batch"""
        results_list = []
        for img in images:
            results_list.append(self.detect(img))
        return results_list

    def _parse_results(self, result) -> DetectionResult:
        """Parse ultralytics RT-DETR result"""
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionResult(
                boxes=np.zeros((0, 4)),
                scores=np.zeros(0),
                class_ids=np.zeros(0, dtype=int)
            )

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        return DetectionResult(boxes, scores, class_ids)

    def train(self, data_yaml: str, epochs: int = 100, batch: int = 16,
              imgsz: int = 640, **kwargs) -> Dict[str, Any]:
        """Fine-tune RT-DETR model"""
        if self.model is None:
            self.load_model()

        print(f"Starting RT-DETR training for {epochs} epochs...")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            **kwargs
        )

        return {"model": self.model, "results": results}


class HuggingFaceDetector(BaseDetector):
    """
    Generic HuggingFace detector for experimenting with latest models
    Supports DETR variants, Grounding DINO, etc.
    """

    AVAILABLE_MODELS = {
        "detr-resnet-50": "facebook/detr-resnet-50",
        "detr-resnet-101": "facebook/detr-resnet-101",
        "detr-resnet-101-dc5": "facebook/detr-resnet-101-dc5",
        "conditional-detr": "microsoft/conditional-detr-resnet-50",
        "table-transformer": "microsoft/table-transformer-detection",
    }

    def __init__(self, model_key: str = "detr-resnet-101", **kwargs):
        """
        Args:
            model_key: Key from AVAILABLE_MODELS or custom HuggingFace path
        """
        model_name = self.AVAILABLE_MODELS.get(model_key, model_key)
        self.detr = DETRDetector(model_name=model_name, **kwargs)
        self.model_name = f"HF-{model_key}"

    def load_model(self, model_path: str = None):
        return self.detr.load_model(model_path)

    def detect(self, image: np.ndarray) -> DetectionResult:
        return self.detr.detect(image)

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        return self.detr.detect_batch(images)
