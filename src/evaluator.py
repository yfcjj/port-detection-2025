"""
Evaluation metrics for object detection and tracking
Computes mAP, precision, recall, F1, and tracking metrics (IDF1, IDP, IDR)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import cv2

from .base_detector import DetectionResult


class DetectionEvaluator:
    """
    Evaluates detection performance using standard metrics:
    - Precision, Recall, F1
    - mAP at different IoU thresholds
    - Per-class metrics
    """

    def __init__(self, iou_thresholds: List[float] = [0.5, 0.75],
                 num_classes: int = 1):
        """
        Args:
            iou_thresholds: IoU thresholds for mAP calculation
            num_classes: Number of object classes
        """
        self.iou_thresholds = iou_thresholds
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all accumulated metrics"""
        self.all_detections = defaultdict(list)  # frame_id -> list of detections
        self.all_ground_truths = defaultdict(list)  # frame_id -> list of ground truths
        self.ap_per_iou = {}
        self.precision_per_iou = {}
        self.recall_per_iou = {}

    def update(self, frame_id: int, detections: DetectionResult,
                ground_truths: DetectionResult):
        """
        Add detections and ground truths for a frame

        Args:
            frame_id: Frame identifier
            detections: Model predictions
            ground_truths: Ground truth annotations
        """
        self.all_detections[frame_id] = detections
        self.all_ground_truths[frame_id] = ground_truths

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics

        Returns:
            Dictionary with metrics:
            - mAP@0.5, mAP@0.75, mAP@0.5:0.95
            - precision, recall, f1
            - tp, fp, fn counts
        """
        metrics = {}

        # Compute metrics for each IoU threshold
        for iou_thresh in self.iou_thresholds:
            ap, precision, recall = self._compute_ap(iou_thresh)

            self.ap_per_iou[iou_thresh] = ap
            self.precision_per_iou[iou_thresh] = precision
            self.recall_per_iou[iou_thresh] = recall

            metrics[f'mAP@{iou_thresh}'] = ap
            metrics[f'Precision@{iou_thresh}'] = precision
            metrics[f'Recall@{iou_thresh}'] = recall

        # Compute mAP@0.5:0.95 (average over multiple IoU thresholds)
        iou_range = np.linspace(0.5, 0.95, 10)
        aps = []
        for iou_thresh in iou_range:
            ap, _, _ = self._compute_ap(iou_thresh)
            aps.append(ap)
        metrics['mAP@0.5:0.95'] = np.mean(aps)

        # Overall F1 score
        precision = metrics.get('Precision@0.5', 0)
        recall = metrics.get('Recall@0.5', 0)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        metrics['F1'] = f1

        # Count true positives, false positives, false negatives
        tp, fp, fn = self._count_detections(0.5)
        metrics['TP'] = tp
        metrics['FP'] = fp
        metrics['FN'] = fn

        return metrics

    def _compute_ap(self, iou_threshold: float) -> Tuple[float, float, float]:
        """Compute Average Precision at specific IoU threshold"""
        all_precisions = []
        all_recalls = []
        aps = []

        # Process each class
        for class_id in range(self.num_classes):
            # Collect all detections and ground truths for this class
            class_detections = []
            class_ground_truths = []

            for frame_id in sorted(self.all_detections.keys()):
                det = self.all_detections[frame_id]
                gt = self.all_ground_truths[frame_id]

                # Filter by class
                det_mask = det.class_ids == class_id
                gt_mask = gt.class_ids == class_id

                for i, score in enumerate(det.scores[det_mask]):
                    box = det.boxes[det_mask][i]
                    class_detections.append({
                        'frame': frame_id,
                        'box': box,
                        'score': score
                    })

                for i, box in enumerate(gt.boxes[gt_mask]):
                    class_ground_truths.append({
                        'frame': frame_id,
                        'box': box,
                        'matched': False
                    })

            if len(class_ground_truths) == 0:
                continue

            # Sort detections by confidence
            class_detections.sort(key=lambda x: x['score'], reverse=True)

            # Compute precision-recall curve
            tp = np.zeros(len(class_detections))
            fp = np.zeros(len(class_detections))

            for i, det in enumerate(class_detections):
                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1

                for j, gt in enumerate(class_ground_truths):
                    if gt['frame'] != det['frame']:
                        continue

                    iou = self._compute_iou(det['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_threshold:
                    if not class_ground_truths[best_gt_idx]['matched']:
                        tp[i] = 1
                        class_ground_truths[best_gt_idx]['matched'] = True
                    else:
                        fp[i] = 1  # Duplicate detection
                else:
                    fp[i] = 1

            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            recalls = tp_cumsum / (len(class_ground_truths) + 1e-10)

            # Compute AP using 11-point interpolation
            ap = 0
            for t in np.linspace(0, 1, 11):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11

            aps.append(ap)

            # Store overall precision/recall
            if len(precisions) > 0:
                all_precisions.append(precisions[-1])
                all_recalls.append(recalls[-1])

        # Mean AP across all classes
        mean_ap = np.mean(aps) if len(aps) > 0 else 0

        # Overall precision and recall
        mean_precision = np.mean(all_precisions) if len(all_precisions) > 0 else 0
        mean_recall = np.mean(all_recalls) if len(all_recalls) > 0 else 0

        return mean_ap, mean_precision, mean_recall

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in xyxy format"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Compute union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-10)

    def _count_detections(self, iou_threshold: float) -> Tuple[int, int, int]:
        """Count TP, FP, FN at specific IoU threshold"""
        tp = 0
        fp = 0
        fn = 0

        for frame_id in self.all_detections.keys():
            det = self.all_detections[frame_id]
            gt = self.all_ground_truths[frame_id]

            matched_gt = set()

            for i, det_box in enumerate(det.boxes):
                best_iou = 0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt.boxes):
                    if j in matched_gt:
                        continue
                    iou = self._compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn += len(gt.boxes) - len(matched_gt)

        return tp, fp, fn

    def print_report(self):
        """Print evaluation report"""
        metrics = self.compute_metrics()

        print("=" * 60)
        print("DETECTION EVALUATION REPORT")
        print("=" * 60)
        print(f"{'Metric':<25} {'Value':>15}")
        print("-" * 40)

        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"{key:<25} {value:>15.4f}")
            else:
                print(f"{key:<25} {value:>15}")

        print("=" * 60)

    def save_to_file(self, filepath: str):
        """Save metrics to file"""
        metrics = self.compute_metrics()

        with open(filepath, 'w') as f:
            f.write("Detection Evaluation Metrics\n")
            f.write("=" * 40 + "\n")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")


class TrackingEvaluator:
    """
    Evaluates tracking performance using metrics:
    - IDF1 (Identity F1 score)
    - IDP (Identity Precision)
    - IDR (Identity Recall)
    - MOTA (Multiple Object Tracking Accuracy)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics"""
        self.frame_data = []

    def update(self, detections: DetectionResult,
               ground_truths: DetectionResult, frame_id: int):
        """
        Add tracking data for a frame

        Args:
            detections: Predictions with track IDs
            ground_truths: Ground truth with track IDs
            frame_id: Frame identifier
        """
        self.frame_data.append({
            'frame_id': frame_id,
            'detections': detections,
            'ground_truths': ground_truths
        })

    def compute_tracking_metrics(self) -> Dict[str, float]:
        """
        Compute tracking metrics (IDF1, IDP, IDR)

        Returns:
            Dictionary with tracking metrics
        """
        # Collect all track data
        gt_tracks = defaultdict(list)  # track_id -> list of boxes
        pred_tracks = defaultdict(list)

        for frame_data in self.frame_data:
            frame_id = frame_data['frame_id']
            det = frame_data['detections']
            gt = frame_data['ground_truths']

            # Extract ground truth tracks
            if gt.track_ids is not None:
                for i, track_id in enumerate(gt.track_ids):
                    gt_tracks[track_id].append({
                        'frame': frame_id,
                        'box': gt.boxes[i]
                    })

            # Extract predicted tracks
            if det.track_ids is not None:
                for i, track_id in enumerate(det.track_ids):
                    pred_tracks[track_id].append({
                        'frame': frame_id,
                        'box': det.boxes[i]
                    })

        # Compute IDF1 metrics
        id_tp = 0  # Identity true positives
        id_fp = 0  # Identity false positives
        id_fn = 0  # Identity false negatives

        # Compute track overlap
        for gt_id in gt_tracks:
            gt_boxes = gt_tracks[gt_id]

            # Find best matching predicted track
            best_overlap = 0
            best_pred_id = None

            for pred_id in pred_tracks:
                pred_boxes = pred_tracks[pred_id]

                # Compute overlap between tracks
                overlap = self._compute_track_overlap(gt_boxes, pred_boxes)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pred_id = pred_id

            if best_overlap > 0.5:  # Threshold for track match
                id_tp += best_overlap
            else:
                id_fn += 1

        # Count false positive tracks (predictions with no match)
        for pred_id in pred_tracks:
            pred_boxes = pred_tracks[pred_id]

            best_overlap = 0
            for gt_id in gt_tracks:
                gt_boxes = gt_tracks[gt_id]
                overlap = self._compute_track_overlap(gt_boxes, pred_boxes)
                if overlap > best_overlap:
                    best_overlap = overlap

            if best_overlap <= 0.5:
                id_fp += 1

        # Compute IDF1, IDP, IDR
        idp = id_tp / (id_tp + id_fp + 1e-10)
        idr = id_tp / (id_tp + id_fn + 1e-10)
        idf1 = 2 * idp * idr / (idp + idr + 1e-10)

        return {
            'IDF1': idf1,
            'IDP': idp,
            'IDR': idr,
            'ID_TP': id_tp,
            'ID_FP': id_fp,
            'ID_FN': id_fn
        }

    def _compute_track_overlap(self, track1: List[Dict], track2: List[Dict]) -> float:
        """Compute overlap between two tracks"""
        # Find frames where both tracks exist
        frames1 = {item['frame']: item['box'] for item in track1}
        frames2 = {item['frame']: item['box'] for item in track2}

        common_frames = set(frames1.keys()) & set(frames2.keys())

        if len(common_frames) == 0:
            return 0.0

        # Compute average IoU over common frames
        total_iou = 0
        for frame in common_frames:
            iou = self._compute_iou(frames1[frame], frames2[frame])
            total_iou += iou

        return total_iou / len(common_frames)

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-10)


class PositionEvaluator:
    """
    Evaluates positioning accuracy for geo-localization
    Compares detected positions to ground truth positions
    """

    def __init__(self, lateral_threshold: float = 0.5,
                 longitudinal_threshold: float = 6.0):
        """
        Args:
            lateral_threshold: Maximum lateral error in meters
            longitudinal_threshold: Maximum longitudinal error in meters
        """
        self.lateral_threshold = lateral_threshold
        self.longitudinal_threshold = longitudinal_threshold
        self.reset()

    def reset(self):
        """Reset accumulated errors"""
        self.lateral_errors = []
        self.longitudinal_errors = []
        self.total_errors = []

    def update(self, pred_positions: np.ndarray,
               gt_positions: np.ndarray):
        """
        Add position errors

        Args:
            pred_positions: N x 2 array of predicted (lon, lat) positions
            gt_positions: N x 2 array of ground truth (lon, lat) positions
        """
        # Compute errors in meters (approximate)
        # 1 degree lat ≈ 111 km
        # 1 degree lon ≈ 111 km * cos(lat)

        for pred, gt in zip(pred_positions, gt_positions):
            pred_lon, pred_lat = pred
            gt_lon, gt_lat = gt

            # Approximate conversion to meters
            lat_m_per_deg = 111000
            lon_m_per_deg = 111000 * np.cos(np.radians(gt_lat))

            # Lateral error (longitude direction)
            lateral_error = abs(pred_lon - gt_lon) * lon_m_per_deg
            self.lateral_errors.append(lateral_error)

            # Longitudinal error (latitude direction)
            longitudinal_error = abs(pred_lat - gt_lat) * lat_m_per_deg
            self.longitudinal_errors.append(longitudinal_error)

            # Total error
            total_error = np.sqrt(lateral_error**2 + longitudinal_error**2)
            self.total_errors.append(total_error)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute positioning metrics"""
        if len(self.total_errors) == 0:
            return {
                'Lateral_Error_Mean': 0.0,
                'Lateral_Error_Max': 0.0,
                'Longitudinal_Error_Mean': 0.0,
                'Longitudinal_Error_Max': 0.0,
                'Lateral_Accuracy': 0.0,
                'Longitudinal_Accuracy': 0.0
            }

        lat_errors = np.array(self.lateral_errors)
        lon_errors = np.array(self.longitudinal_errors)

        # Percentage within thresholds
        lateral_acc = np.mean(lat_errors <= self.lateral_threshold)
        longitudinal_acc = np.mean(lon_errors <= self.longitudinal_threshold)

        return {
            'Lateral_Error_Mean': np.mean(lat_errors),
            'Lateral_Error_Max': np.max(lat_errors),
            'Longitudinal_Error_Mean': np.mean(lon_errors),
            'Longitudinal_Error_Max': np.max(lon_errors),
            'Lateral_Accuracy': lateral_acc,
            'Longitudinal_Accuracy': longitudinal_acc
        }

    def print_report(self):
        """Print positioning report"""
        metrics = self.compute_metrics()

        print("=" * 60)
        print("POSITIONING EVALUATION REPORT")
        print("=" * 60)
        print(f"{'Metric':<30} {'Value':>15}")
        print("-" * 45)

        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                if 'Accuracy' in key or 'Mean' in key:
                    print(f"{key:<30} {value:>15.4f}")
                else:
                    print(f"{key:<30} {value:>15.2f} m")
            else:
                print(f"{key:<30} {value:>15}")

        print("=" * 60)
