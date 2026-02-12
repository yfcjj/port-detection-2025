"""
Main evaluation script for port vehicle detection
Evaluates model on test set and outputs quantitative metrics
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any
import time

from src.base_detector import DetectionResult
from src.yolo_detector import YOLODetector, YOLOv11Detector
from src.detr_detector import DETRDetector, RTDETRDetector
from src.evaluator import DetectionEvaluator, TrackingEvaluator, PositionEvaluator


# Vehicle class mapping (COCO to single vehicle class)
VEHICLE_CLASSES = {2: 0, 5: 0, 7: 0}  # car, bus, truck -> vehicle (0)
OTHER_VEHICLE_CLASSES = {0: 0, 1: 0, 3: 0, 6: 0, 8: 0}  # person, bicycle, motorcycle, train, boat (optional)


def load_annotations(annotations_dir: Path) -> Dict[int, DetectionResult]:
    """
    Load ground truth annotations from JSON format

    Args:
        annotations_dir: Directory containing annotations.json

    Returns:
        Dictionary mapping frame_id to DetectionResult
    """
    json_path = annotations_dir / "annotations.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Annotations not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = {}

    for frame_str, frame_data in data['frames'].items():
        frame_id = int(frame_str)

        boxes = []
        class_ids = []
        track_ids = []

        for ann in frame_data['annotations']:
            # Convert from normalized YOLO format to pixel xyxy
            # We'll get image dimensions during evaluation
            boxes.append([
                ann['x_center'],
                ann['y_center'],
                ann['width'],
                ann['height']
            ])
            class_ids.append(ann['class_id'])
            track_ids.append(ann['track_id'])

        # Store in YOLO format (will convert during evaluation)
        annotations[frame_id] = {
            'boxes_yolo': np.array(boxes),  # x_center, y_center, width, height (normalized)
            'class_ids': np.array(class_ids),
            'track_ids': np.array(track_ids)
        }

    return annotations


def yolo_to_xyxy(boxes_yolo: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """Convert YOLO format to xyxy pixel coordinates"""
    # Handle empty arrays
    if len(boxes_yolo) == 0:
        return np.zeros((0, 4))

    # Ensure 2D array
    if boxes_yolo.ndim == 1:
        boxes_yolo = boxes_yolo.reshape(1, -1)

    # Ensure 4 columns
    if boxes_yolo.shape[1] != 4:
        raise ValueError(f"Expected 4 columns (x_center, y_center, width, height), got {boxes_yolo.shape[1]}")

    boxes_xyxy = np.zeros((len(boxes_yolo), 4))

    # x_center, y_center, width, height -> x1, y1, x2, y2
    boxes_xyxy[:, 0] = (boxes_yolo[:, 0] - boxes_yolo[:, 2] / 2) * img_width  # x1
    boxes_xyxy[:, 1] = (boxes_yolo[:, 1] - boxes_yolo[:, 3] / 2) * img_height  # y1
    boxes_xyxy[:, 2] = (boxes_yolo[:, 0] + boxes_yolo[:, 2] / 2) * img_width  # x2
    boxes_xyxy[:, 3] = (boxes_yolo[:, 1] + boxes_yolo[:, 3] / 2) * img_height  # y2

    return boxes_xyxy


def evaluate_model(detector, video_path: str, annotations_dir: Path,
                   output_dir: Path, tracking: bool = True,
                   skip_frames: int = 1) -> Dict[str, Any]:
    """
    Evaluate detector on test video

    Args:
        detector: Detector instance
        video_path: Path to test video
        annotations_dir: Directory containing ground truth annotations
        output_dir: Directory to save results
        tracking: Whether to evaluate tracking
        skip_frames: Process every Nth frame

    Returns:
        Dictionary containing all metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    print(f"Loading annotations from {annotations_dir}...")
    annotations = load_annotations(annotations_dir)
    print(f"Loaded {len(annotations)} annotated frames")

    # Initialize evaluators
    det_evaluator = DetectionEvaluator(iou_thresholds=[0.5, 0.75], num_classes=1)
    track_evaluator = TrackingEvaluator() if tracking else None

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Evaluating on video: {video_path}")
    print(f"Total frames: {total_frames}")

    # Statistics
    inference_times = []
    processed_frames = 0

    # Process frames
    for frame_idx in range(0, total_frames, skip_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        h, w = frame.shape[:2]

        # Run inference with timing
        start_time = time.time()

        if tracking:
            result = detector.detect_with_tracking(frame, persist=True)
        else:
            result = detector.detect(frame)

        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Filter and map to vehicle class only
        # Keep only car(2), bus(5), truck(7) and map to class 0
        vehicle_mask = np.isin(result.class_ids, list(VEHICLE_CLASSES.keys()))
        result = DetectionResult(
            boxes=result.boxes[vehicle_mask],
            scores=result.scores[vehicle_mask],
            class_ids=np.array([VEHICLE_CLASSES.get(cid, 0) for cid in result.class_ids[vehicle_mask]]),
            track_ids=result.track_ids[vehicle_mask] if result.track_ids is not None else None
        )

        # Get ground truth for this frame
        if frame_idx in annotations:
            gt_data = annotations[frame_idx]

            # Convert GT to pixel coordinates
            gt_boxes = yolo_to_xyxy(gt_data['boxes_yolo'], w, h)

            ground_truth = DetectionResult(
                boxes=gt_boxes,
                scores=np.ones(len(gt_boxes)),  # GT has perfect confidence
                class_ids=gt_data['class_ids'],
                track_ids=gt_data['track_ids']
            )

            # Update detection evaluator
            det_evaluator.update(frame_idx, result, ground_truth)

            # Update tracking evaluator if tracking enabled
            if track_evaluator and result.track_ids is not None:
                track_evaluator.update(result, ground_truth, frame_idx)

        processed_frames += 1

        # Progress update
        if processed_frames % 10 == 0:
            avg_time = np.mean(inference_times[-10:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Processed {processed_frames} frames | Avg inference: {avg_time*1000:.1f}ms | FPS: {fps:.1f}")

    cap.release()

    # Compute final metrics
    print("\n" + "="*60)
    print("COMPUTING METRICS")
    print("="*60)

    detection_metrics = det_evaluator.compute_metrics()
    det_evaluator.print_report()

    # Add timing metrics
    detection_metrics['avg_inference_time_ms'] = np.mean(inference_times) * 1000
    detection_metrics['min_inference_time_ms'] = np.min(inference_times) * 1000
    detection_metrics['max_inference_time_ms'] = np.max(inference_times) * 1000
    detection_metrics['fps'] = 1.0 / np.mean(inference_times)
    detection_metrics['processed_frames'] = processed_frames

    # Tracking metrics
    tracking_metrics = {}
    if track_evaluator:
        tracking_metrics = track_evaluator.compute_tracking_metrics()
        print("\nTRACKING METRICS")
        print("="*40)
        for key, value in tracking_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    # Combine all metrics
    all_metrics = {
        'detection': detection_metrics,
        'tracking': tracking_metrics,
        'model_info': detector.get_model_info()
    }

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate detection model')
    parser.add_argument('--model', type=str, default='yolo11l.onnx',
                       help='Path to model file')
    parser.add_argument('--model-type', type=str, default='yolo',
                       choices=['yolo', 'detr', 'rtdetr'],
                       help='Type of model')
    parser.add_argument('--video', type=str,
                       default='/data/ljw/ljw/宁波大榭港/监控视频/2N测试右侧低_20251224111945/2N测试右侧低_20251224010000-20251224015959_1.mp4',
                       help='Path to test video')
    parser.add_argument('--annotations', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/data/annotations',
                       help='Path to annotations directory')
    parser.add_argument('--output', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/results',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--no-tracking', action='store_true',
                       help='Disable tracking evaluation')
    parser.add_argument('--skip-frames', type=int, default=5,
                       help='Process every Nth frame')

    args = parser.parse_args()

    # Create detector based on type
    print(f"Loading {args.model_type} detector from: {args.model}")

    if args.model_type == 'yolo':
        detector = YOLODetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            use_onnx=args.model.endswith('.onnx')
        )
    elif args.model_type == 'detr':
        detector = DETRDetector(
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    elif args.model_type == 'rtdetr':
        detector = RTDETRDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Evaluate
    metrics = evaluate_model(
        detector=detector,
        video_path=args.video,
        annotations_dir=Path(args.annotations),
        output_dir=Path(args.output),
        tracking=not args.no_tracking,
        skip_frames=args.skip_frames
    )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"mAP@0.5: {metrics['detection'].get('mAP@0.5', 0):.4f}")
    print(f"Precision@0.5: {metrics['detection'].get('Precision@0.5', 0):.4f}")
    print(f"Recall@0.5: {metrics['detection'].get('Recall@0.5', 0):.4f}")
    print(f"F1: {metrics['detection'].get('F1', 0):.4f}")
    print(f"FPS: {metrics['detection'].get('fps', 0):.1f}")

    if metrics['tracking']:
        print(f"IDF1: {metrics['tracking'].get('IDF1', 0):.4f}")


if __name__ == '__main__':
    main()
