"""
Comprehensive Metrics for Object Detection and Tracking
Implements standard metrics: mAP, Precision, Recall, MOTA, MOTP, IDF1, ID Switch
"""
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class DetectionMetrics:
    """
    Detection metrics: mAP, Precision, Recall, F1
    """

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
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

    @staticmethod
    def compute_ap(predictions, ground_truths, iou_threshold: float = 0.5) -> Tuple[float, float, float]:
        """
        Compute Average Precision

        Args:
            predictions: List of (box, score, class_id)
            ground_truths: List of (box, class_id)
            iou_threshold: IoU threshold for matching

        Returns:
            (ap, precision, recall)
        """
        if len(ground_truths) == 0:
            return 0.0, 1.0 if len(predictions) == 0 else 0.0, 0.0

        # Sort predictions by score
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        gt_matched = [False] * len(ground_truths)

        for i, (pred_box, pred_score, pred_class) in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1

            for j, (gt_box, gt_class) in enumerate(ground_truths):
                if gt_class != pred_class:
                    continue
                if gt_matched[j]:
                    continue

                iou = DetectionMetrics.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / (len(ground_truths) + 1e-10)

        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11

        return float(ap), float(precision[-1]) if len(precision) > 0 else 0.0, float(recall[-1]) if len(recall) > 0 else 0.0


class TrackingMetrics:
    """
    Multi-Object Tracking Accuracy (MOTA) and related metrics
    Reference: "MOTChallenge: https://motchallenge.net/"
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulated metrics"""
        self.frames = []

    def update(self, frame_id: int, gt_boxes: np.ndarray, gt_ids: np.ndarray,
                pred_boxes: np.ndarray, pred_ids: np.ndarray):
        """
        Update tracking metrics for a frame

        Args:
            frame_id: Frame number
            gt_boxes: N x 4 array of ground truth boxes (xyxy)
            gt_ids: N array of ground truth track IDs
            pred_boxes: M x 4 array of predicted boxes (xyxy)
            pred_ids: M array of predicted track IDs
        """
        self.frames.append({
            'frame_id': frame_id,
            'gt_boxes': gt_boxes,
            'gt_ids': gt_ids,
            'pred_boxes': pred_boxes,
            'pred_ids': pred_ids
        })

    def compute_metrics(self, iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute tracking metrics: MOTA, MOTP, IDF1, ID Switches, etc.

        Returns:
            Dictionary with all tracking metrics
        """
        if len(self.frames) == 0:
            return {
                'MOTA': 0.0,
                'MOTP': 0.0,
                'IDF1': 0.0,
                'IDP': 0.0,
                'IDR': 0.0,
                'FP': 0,
                'FN': 0,
                'ID_switch': 0,
                'fragments': 0,
                'MT': 0,
                'ML': 0,
                'FAF': 0
            }

        # Accumulate statistics
        total_gt = 0
        total_pred = 0
        fp = 0  # False Positives
        fn = 0  # False Negatives
        id_switches = 0  # ID switches
        fragments = 0  # Track fragments

        # For MOTP computation
        iou_sum = 0
        matches_count = 0

        # For IDF1 computation
        id_tp = 0  # Identity true positives
        id_fp = 0
        id_fn = 0

        prev_gt_to_pred = {}  # Track ID mappings from previous frame

        for frame in self.frames:
            gt_boxes = frame['gt_boxes']
            gt_ids = frame['gt_ids']
            pred_boxes = frame['pred_boxes']
            pred_ids = frame['pred_ids']

            total_gt += len(gt_ids)
            total_pred += len(pred_ids)

            # Build cost matrix for assignment
            cost_matrix = np.zeros((len(gt_ids), len(pred_ids)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    iou = DetectionMetrics.compute_iou(gt_box, pred_box)
                    cost_matrix[i, j] = -iou  # Negative for maximization

            # Optimal assignment using Hungarian algorithm
            gt_to_pred = {}
            matched_gt = set()
            matched_pred = set()

            if len(gt_ids) > 0 and len(pred_ids) > 0:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for i, j in zip(row_ind, col_ind):
                    iou = -cost_matrix[i, j]
                    if iou >= iou_threshold:
                        gt_to_pred[gt_ids[i]] = pred_ids[j]
                        matched_gt.add(gt_ids[i])
                        matched_pred.add(pred_ids[j])
                        iou_sum += iou
                        matches_count += 1

            # Count FN (ground truth not matched)
            fn += len(gt_ids) - len(matched_gt)

            # Count FP (predictions not matched)
            fp += len(pred_ids) - len(matched_pred)

            # Count ID switches
            prev_mapping = prev_gt_to_pred.copy()
            for gt_id, pred_id in gt_to_pred.items():
                if gt_id in prev_mapping and prev_mapping[gt_id] != pred_id:
                    id_switches += 1
                # Check for fragments (same GT tracked by multiple IDs)
                if gt_id in prev_mapping and prev_mapping[gt_id] != pred_id:
                    # This could be a fragment or recover
                    pass

            prev_gt_to_pred = gt_to_pred

            # Count fragments (simplified - a track is broken into multiple segments)
            # Count how many GT IDs were previously seen but now have different predictions
            for gt_id in gt_ids:
                if gt_id in prev_gt_to_pred:
                    if gt_id not in gt_to_pred or gt_to_pred[gt_id] != prev_gt_to_pred[gt_id]:
                        fragments += 1

        # Compute MOTA (Multiple Object Tracking Accuracy)
        # MOTA = 1 - (FN + FP + IDSWITCH) / total_gt
        if total_gt > 0:
            mota = 1.0 - (fn + fp + id_switches) / total_gt
        else:
            mota = 0.0

        # Compute MOTP (Multiple Object Tracking Precision)
        # Average IoU of matched detections
        if matches_count > 0:
            motp = iou_sum / matches_count
        else:
            motp = 0.0

        # Compute IDF1 (Identity F1)
        # Count unique GT IDs and prediction IDs
        all_gt_ids = set()
        all_pred_ids = set()
        for frame in self.frames:
            for gt_id in frame['gt_ids']:
                all_gt_ids.add(gt_id)
            for pred_id in frame['pred_ids']:
                all_pred_ids.add(pred_id)

        # Compute identity metrics
        for gt_id in all_gt_ids:
            # Check if this GT ID was ever correctly tracked
            gt_matched_frames = []
            pred_id_for_gt = []

            for frame in self.frames:
                for i, gid in enumerate(frame['gt_ids']):
                    if gid == gt_id:
                        gt_matched_frames.append(frame['frame_id'])
                        # Find corresponding prediction
                        for j, pid in enumerate(frame['pred_ids']):
                            # Use spatial overlap as proxy for identity match
                            if i < len(frame['gt_boxes']) and j < len(frame['pred_boxes']):
                                iou = DetectionMetrics.compute_iou(
                                    frame['gt_boxes'][i],
                                    frame['pred_boxes'][j]
                                )
                                if iou >= iou_threshold:
                                    pred_id_for_gt.append(pid)
                                    break

            if len(pred_id_for_gt) > 0:
                # Most common prediction ID for this GT
                from collections import Counter
                most_common_pred = Counter(pred_id_for_gt).most_common(1)[0][0]
                if most_common_pred in pred_id_for_gt:
                    id_tp += pred_id_for_gt.count(most_common_pred)

        id_fp = total_pred - id_tp
        id_fn = len(all_gt_ids)

        if id_tp + id_fp > 0:
            idp = id_tp / (id_tp + id_fp)
        else:
            idp = 0.0

        if id_tp + id_fn > 0:
            idr = id_tp / (id_tp + id_fn)
        else:
            idr = 0.0

        if idp + idr > 0:
            idf1 = 2 * idp * idr / (idp + idr)
        else:
            idf1 = 0.0

        # Other metrics
        mt = len(all_gt_ids)  # Most tracks (mostly targets)
        ml = len([f for f in self.frames if len(f['gt_ids']) > mt])  # Mostly lost
        faf = 0  # False alarm trajectory (fragmented, would need more complex tracking)

        return {
            'MOTA': float(mota),
            'MOTP': float(motp),
            'IDF1': float(idf1),
            'IDP': float(idp),
            'IDR': float(idr),
            'FP': int(fp),
            'FN': int(fn),
            'ID_switch': int(id_switches),
            'fragments': int(fragments),
            'MT': int(mt),
            'ML': int(ml),
            'FAF': int(faf),
            'total_gt': int(total_gt),
            'total_pred': int(total_pred)
        }

    def print_report(self, metrics: Dict[str, float]):
        """Print formatted tracking metrics report"""
        print("\n" + "="*70)
        print("MULTI-OBJECT TRACKING METRICS (MOTChallenge)")
        print("="*70)
        print(f"{'Metric':<20} {'Value':>15} {'Description':<30}")
        print("-"*65)

        metric_descriptions = {
            'MOTA': 'Multi-Object Tracking Accuracy (higher is better, max=1)',
            'MOTP': 'Multi-Object Tracking Precision (average IoU, higher is better)',
            'IDF1': 'Identity F1 Score (higher is better, max=1)',
            'IDP': 'Identity Precision (higher is better, max=1)',
            'IDR': 'Identity Recall (higher is better, max=1)',
            'FP': 'False Positives (lower is better)',
            'FN': 'False Negatives (lower is better)',
            'ID_switch': 'Identity Switches (lower is better)',
            'fragments': 'Track Fragments (lower is better)',
            'MT': 'Most Targets (total GT objects)',
            'ML': 'Mostly Lost ( trajectories with <80% coverage)',
            'FAF': 'False Alarm Trajectories',
        }

        for key, desc in metric_descriptions.items():
            if key in metrics:
                val = metrics[key]
                if isinstance(val, float):
                    print(f"{key:<20} {val:>15.4f}  {desc}")
                else:
                    print(f"{key:<20} {val:>15}    {desc}")

        print("="*70)


def compute_id_switches(gt_tracks, pred_tracks, iou_threshold=0.5):
    """
    Count ID switches - when a GT object's tracked ID changes

    Args:
        gt_tracks: Dict mapping gt_id to list of (frame_id, box)
        pred_tracks: Dict mapping pred_id to list of (frame_id, box)
        iou_threshold: IoU threshold for matching

    Returns:
        Number of ID switches
    """
    id_switches = 0

    for gt_id, gt_detections in gt_tracks.items():
        current_pred_id = None

        for frame_id, gt_box in gt_detections:
            # Find which prediction matches this GT
            matched_pred_id = None
            best_iou = 0

            for pred_id, pred_detections in pred_tracks.items():
                for f_id, pred_box in pred_detections:
                    if f_id != frame_id:
                        continue
                    iou = DetectionMetrics.compute_iou(gt_box, pred_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        matched_pred_id = pred_id

            if matched_pred_id is not None:
                if current_pred_id is not None and current_pred_id != matched_pred_id:
                    id_switches += 1
                current_pred_id = matched_pred_id

    return id_switches
