"""
Base Detector Interface for easy model swapping
Supports both YOLO (CNN) and DETR (Transformer) based detectors
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2


class DetectionResult:
    """Standard detection result format"""
    def __init__(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                 track_ids: Optional[np.ndarray] = None):
        """
        Args:
            boxes: N x 4 array of [x1, y1, x2, y2] in pixel coordinates
            scores: N array of confidence scores
            class_ids: N array of class IDs
            track_ids: Optional N array of track IDs
        """
        self.boxes = boxes  # [x1, y1, x2, y2]
        self.scores = scores
        self.class_ids = class_ids
        self.track_ids = track_ids

    def to_xywh(self) -> np.ndarray:
        """Convert boxes to [x_center, y_center, width, height] format"""
        boxes = self.boxes.copy()
        boxes_xywh = np.zeros_like(boxes)
        boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
        boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        return boxes_xywh

    def to_normalized(self, img_width: int, img_height: int) -> np.ndarray:
        """Convert boxes to normalized coordinates [0-1]"""
        boxes_xywh = self.to_xywh()
        boxes_norm = boxes_xywh.copy()
        boxes_norm[:, 0] /= img_width  # x_center
        boxes_norm[:, 1] /= img_height  # y_center
        boxes_norm[:, 2] /= img_width  # width
        boxes_norm[:, 3] /= img_height  # height
        return boxes_norm

    def filter_by_conf(self, threshold: float) -> 'DetectionResult':
        """Filter detections by confidence threshold"""
        mask = self.scores >= threshold
        new_track_ids = self.track_ids[mask] if self.track_ids is not None else None
        return DetectionResult(
            self.boxes[mask], self.scores[mask], self.class_ids[mask], new_track_ids
        )


class BaseDetector(ABC):
    """Abstract base class for object detectors"""

    def __init__(self, conf_threshold: float = 0.5, iou_threshold: float = 0.45,
                 max_det: int = 300, device: str = "cuda"):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.device = device
        self.model_name = "BaseDetector"

    @abstractmethod
    def load_model(self, model_path: str):
        """Load model from disk"""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Perform detection on a single image

        Args:
            image: Input image (BGR format from cv2)

        Returns:
            DetectionResult object with boxes, scores, and class IDs
        """
        pass

    @abstractmethod
    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Perform detection on a batch of images"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_det": self.max_det,
            "device": self.device
        }


class BaseTracker(ABC):
    """Abstract base class for object trackers"""

    def __init__(self, tracker_config: Dict[str, Any]):
        self.config = tracker_config

    @abstractmethod
    def update(self, detections: DetectionResult, frame_id: int) -> DetectionResult:
        """
        Update tracker with new detections

        Args:
            detections: DetectionResult for current frame
            frame_id: Current frame number

        Returns:
            DetectionResult with track_ids assigned
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset tracker state"""
        pass
