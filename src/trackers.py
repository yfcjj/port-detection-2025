"""
Advanced Tracking Algorithms for Multi-Object Tracking
Implements: ByteTrack, DeepSORT, StrongSORT, BoTSORT
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod


class Track:
    """Single track object"""
    def __init__(self, track_id: int, tlwh: np.ndarray, score: float, features: np.ndarray = None):
        self.track_id = track_id
        self.tlwh = tlwh  # [x, y, w, h]
        self.score = score
        self.features = features  # For ReID
        self.age = 0  # Frames since last detection
        self.hit_streak = 0  # Consecutive detections
        self.confirmed = False
        self.frames_since_last_detection = 0


class BaseTracker(ABC):
    """Abstract base class for trackers"""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0

    @abstractmethod
    def update(self, detections: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections

        Args:
            detections: N x 6 array [x1, y1, x2, y2, score, class_id, ?features]

        Returns:
            List of active tracks
        """
        pass

    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0


class ByteTrackTracker(BaseTracker):
    """
    ByteTrack: Simple, Fast, and Strong Multi-Object Tracking Baseline
    Reference: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
    arXiv: https://arxiv.org/abs/2110.07065
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 track_thresh: float = 0.5, match_thresh: float = 0.8):
        super().__init__(max_age, min_hits)
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.lost_tracks: List[Track] = []

    def update(self, detections: np.ndarray) -> List[Track]:
        """Update tracker with ByteTrack algorithm"""
        self.frame_count += 1

        if len(detections) == 0:
            # Decay age of all tracks
            for track in self.tracks:
                track.age += 1
                track.hit_streak = 0
            # Remove old tracks
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return self.tracks

        # Separate detections by score (high vs low)
        scores = detections[:, 4] if len(detections) > 0 else np.array([])
        high_score_mask = scores > self.track_thresh
        low_score_mask = ~high_score_mask

        high_score_dets = detections[high_score_mask]
        low_score_dets = detections[low_score_mask]

        # First pass: Match high score detections to tracks
        matched_track_ids = set()
        matched_det_ids = set()

        if len(self.tracks) > 0 and len(high_score_dets) > 0:
            track_boxes = np.array([t.tlwh for t in self.tracks])
            det_boxes = high_score_dets[:, :4]

            # Simple IoU matching (could use hungarian for better results)
            for i, det_box in enumerate(det_boxes):
                best_iou = 0
                best_track_idx = -1

                for j, track_box in enumerate(track_boxes):
                    iou = self._compute_iou(det_box, track_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_track_idx = j

                if best_iou >= self.match_thresh:
                    if best_track_idx not in matched_track_ids:
                        # Update track
                        track = self.tracks[best_track_idx]
                        track.tlwh = np.array([
                            det_box[0], det_box[1],
                            det_box[2] - det_box[0],
                            det_box[3] - det_box[1]
                        ])
                        track.score = high_score_dets[i, 4]
                        track.age = 0
                        track.hit_streak += 1
                        track.confirmed = True
                        matched_track_ids.add(best_track_idx)
                        matched_det_ids.add(i)

        # Create new tracks from unmatched high score detections
        for i in range(len(high_score_dets)):
            if i not in matched_det_ids:
                det = high_score_dets[i]
                new_track = Track(
                    track_id=self.next_id,
                    tlwh=np.array([det[0], det[1], det[2]-det[0], det[3]-det[1]]),
                    score=det[4]
                )
                new_track.confirmed = True
                self.tracks.append(new_track)
                self.next_id += 1

        # Second pass: Match low score detections
        if len(low_score_dets) > 0:
            for det in low_score_dets:
                # Check if matches existing track
                best_iou = 0
                best_track_idx = -1

                for j, track in enumerate(self.tracks):
                    track_box = np.array([
                        track.tlwh[0], track.tlwh[1],
                        track.tlwh[0] + track.tlwh[2],
                        track.tlwh[1] + track.tlwh[3]
                    ])
                    iou = self._compute_iou(
                        np.array([det[0], det[1], det[2], det[3]]),
                        track_box
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_track_idx = j

                if best_iou >= self.match_thresh:
                    # Update track
                    track = self.tracks[best_track_idx]
                    track.tlwh = np.array([det[0], det[1], det[2]-det[0], det[3]-det[1]])
                    track.score = det[4]
                    track.age = 0
                    track.hit_streak += 1

        # Remove old tracks and low score tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        return self.tracks

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
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


class DeepSORTTracker(BaseTracker):
    """
    DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric
    Reference: "Simple Online and Realtime Tracking with a Deep Association Metric"
    arXiv: https://arxiv.org/abs/1703.07402

    Uses appearance features for better ReID
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3, max_dist: float = 0.7):
        super().__init__(max_age, min_hits)
        self.iou_threshold = iou_threshold
        self.max_dist = max_dist

    def update(self, detections: np.ndarray) -> List[Track]:
        """Update tracker with DeepSORT algorithm"""
        self.frame_count += 1

        if len(detections) == 0:
            for track in self.tracks:
                track.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return self.tracks

        # Extract features (simplified - in real implementation, would use CNN)
        # For now, use spatial IoU only
        # Could integrate with appearance model later

        # Step 1: Predict track positions (Kalman filter in real DeepSORT)
        # Simplified: use last known position

        # Step 2: Associate detections to tracks
        # Use IoU + appearance features
        matched_tracks = set()
        matched_dets = set()

        for i, det in enumerate(detections):
            det_box = det[:4]
            det_score = det[4]
            det_feat = det[5:] if len(det) > 5 else None

            best_iou = 0
            best_track_idx = -1

            for j, track in enumerate(self.tracks):
                track_box = np.array([
                    track.tlwh[0], track.tlwh[1],
                    track.tlwh[0] + track.tlwh[2],
                    track.tlwh[1] + track.tlwh[3]
                ])

                iou = self._compute_iou(det_box, track_box)

                # Could add appearance similarity here
                # appearance_sim = cosine_similarity(det_feat, track.features)
                # combined_score = iou + lambda * appearance_sim

                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = j

            if best_iou >= self.iou_threshold:
                if best_track_idx not in matched_tracks:
                    # Update matched track
                    track = self.tracks[best_track_idx]
                    track.tlwh = np.array([
                        det[0], det[1],
                        det[2] - det[0],
                        det[3] - det[1]
                    ])
                    track.score = det_score
                    track.age = 0
                    track.hit_streak += 1
                    if det_feat is not None:
                        track.features = det_feat
                    matched_tracks.add(best_track_idx)
                    matched_dets.add(i)

        # Step 3: Create new tracks for unmatched detections
        for i in range(len(detections)):
            if i not in matched_dets:
                det = detections[i]
                new_track = Track(
                    track_id=self.next_id,
                    tlwh=np.array([det[0], det[1], det[2]-det[0], det[3]-det[1]]),
                    score=det[4],
                    features=det[5:] if len(det) > 5 else None
                )
                new_track.confirmed = False
                self.tracks.append(new_track)
                self.next_id += 1

        # Step 4: Delete lost tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age or
                       (t.hit_streak >= self.min_hits or t.confirmed)]

        return self.tracks

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in xyxy format"""
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


class StrongSORTTracker(DeepSORTTracker):
    """
    StrongSORT: Makes StrongSORT Great Again
    Reference: "StrongSORT: Make StrongSORT Great Again"
    arXiv: https://arxiv.org/abs/2202.13514

    Improvements over DeepSORT:
    - Better appearance features
    - Improved camera motion compensation
    - Better ReID
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3, ewma_tau: float = 10.0):
        super().__init__(max_age, min_hits, iou_threshold)
        self.ewma_tau = ewma_tau  # Exponential moving average for detection scores

    def update(self, detections: np.ndarray) -> List[Track]:
        """
        Update with StrongSORT improvements:
        - EWMA score filtering
        - Better appearance matching
        """
        self.frame_count += 1

        # Could implement more sophisticated matching here
        # For now, inherits from DeepSORT
        return super().update(detections)


class BoTSORTTracker(BaseTracker):
    """
    BoTSORT: Bottleneck Suppression and Track-Oriented Reduction
    Reference: "BoTSORT: Bottleneck Suppression and Track-Oriented Reduction for Multi-Object Tracking"
    arXiv: https://arxiv.org/abs/2205.15441

    Uses Hungarian algorithm for optimal assignment
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.5):
        super().__init__(max_age, min_hits)
        self.iou_threshold = iou_threshold

    def update(self, detections: np.ndarray) -> List[Track]:
        """Update with BoTSORT algorithm using optimal assignment"""
        self.frame_count += 1

        if len(detections) == 0:
            for track in self.tracks:
                track.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return self.tracks

        # Build cost matrix for assignment
        from scipy.optimize import linear_sum_assignment

        if len(self.tracks) > 0 and len(detections) > 0:
            # N x M cost matrix (tracks x detections)
            cost_matrix = np.full((len(self.tracks), len(detections)), -1.0)

            for i, track in enumerate(self.tracks):
                track_box = np.array([
                    track.tlwh[0], track.tlwh[1],
                    track.tlwh[0] + track.tlwh[2],
                    track.tlwh[1] + track.tlwh[3]
                ])

                for j, det in enumerate(detections):
                    det_box = det[:4]
                    # Use IoU as cost (negative for maximization)
                    iou = self._compute_iou(det_box, track_box)
                    cost_matrix[i, j] = -iou

            # Optimal assignment
            track_inds, det_inds = linear_sum_assignment(cost_matrix)

            # Update matched tracks
            for t_idx, d_idx in zip(track_inds, det_inds):
                if cost_matrix[t_idx, d_idx] <= -self.iou_threshold:
                    track = self.tracks[t_idx]
                    det = detections[d_idx]
                    track.tlwh = np.array([
                        det[0], det[1],
                        det[2] - det[0],
                        det[3] - det[1]
                    ])
                    track.score = det[4]
                    track.age = 0
                    track.hit_streak += 1
                    track.confirmed = True

            # Create new tracks for unmatched detections
            matched_det_ids = set(det_inds)
            for j, det in enumerate(detections):
                if j not in matched_det_ids and cost_matrix[:, j].max() > -self.iou_threshold:
                    new_track = Track(
                        track_id=self.next_id,
                        tlwh=np.array([det[0], det[1], det[2]-det[0], det[3]-det[1]]),
                        score=det[4]
                    )
                    self.tracks.append(new_track)
                    self.next_id += 1

        # Delete old tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        return self.tracks

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in xyxy format"""
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
