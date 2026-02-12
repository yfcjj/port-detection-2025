"""
Quick start script for port vehicle detection evaluation
Prepares data, runs evaluation, and generates report
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.yolo_detector import YOLODetector
from src.evaluator import DetectionEvaluator, TrackingEvaluator, PositionEvaluator


# Homography matrix from original code (for positioning)
H = np.array([[-0.03313215142319288, -4.203239026231838, 121.91918568302543],
              [-0.008126421477698355, -1.0317445514083663, 29.916497471186226],
              [-0.0002716899287631901, -0.03447319877318248, 1.0]])

# Coordinate system parameters
CP = {'lon_min': 121.927456, 'lon_max': 121.929914, 'lat_min': 29.928481,
      'lat_max': 29.930089, 'scale': 400000}


def pixel_to_world(x: float, y: float, img_width: int, img_height: int) -> tuple:
    """
    Convert pixel coordinates to world coordinates (lon, lat)

    Args:
        x, y: Pixel coordinates
        img_width, img_height: Image dimensions
    """
    # Scale to PAINT coordinates
    PAINT_W, PAINT_H = 1807, 1020
    cx_paint = x * (PAINT_W / img_width)
    cy_paint = y * (PAINT_H / img_height)

    # Apply homography
    pixel_pt = np.array([[[cx_paint, cy_paint]]], dtype='float64')
    world_pt = cv2.perspectiveTransform(pixel_pt, H)[0][0]

    return world_pt[0], world_pt[1]  # lon, lat


def box_to_world_position(box: np.ndarray, img_width: int, img_height: int) -> tuple:
    """
    Convert detection box to world position (using bottom center)

    Args:
        box: [x1, y1, x2, y2] in pixel coordinates
        img_width, img_height: Image dimensions

    Returns:
        (lon, lat) tuple
    """
    # Use bottom center of box
    cx = (box[0] + box[2]) / 2
    cy = box[3]  # Bottom edge

    return pixel_to_world(cx, cy, img_width, img_height)


def quick_evaluation(model_path: str, video_path: str, output_dir: str,
                     num_frames: int = 100):
    """
    Run quick evaluation without ground truth (for testing inference speed)

    Args:
        model_path: Path to model
        video_path: Path to video
        output_dir: Output directory
        num_frames: Number of frames to process
    """
    import time

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("QUICK MODEL EVALUATION (No Ground Truth)")
    print("="*60)

    # Load model
    print(f"\nLoading model: {model_path}")
    detector = YOLODetector(model_path=model_path, use_onnx=model_path.endswith('.onnx'))
    detector.load_model()

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Processing {num_frames} frames...")

    # Statistics
    inference_times = []
    detection_counts = []

    frame_step = max(1, total_frames // num_frames)

    for i in range(0, min(total_frames, num_frames * frame_step), frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            continue

        h, w = frame.shape[:2]

        # Run inference
        start = time.time()
        result = detector.detect(frame)
        inference_time = time.time() - start

        inference_times.append(inference_time)
        detection_counts.append(len(result.boxes))

        if i % 20 == 0:
            avg_time = np.mean(inference_times[-10:]) if len(inference_times) >= 10 else inference_time
            fps = 1.0 / avg_time
            print(f"Frame {i}: {len(result.boxes)} detections | "
                  f"Time: {inference_time*1000:.1f}ms | FPS: {fps:.1f}")

    cap.release()

    # Compute statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps = 1.0 / avg_time

    avg_detections = np.mean(detection_counts)

    print(f"Processed frames: {len(inference_times)}")
    print(f"\nTiming:")
    print(f"  Avg: {avg_time*1000:.1f}ms")
    print(f"  Min: {min_time*1000:.1f}ms")
    print(f"  Max: {max_time*1000:.1f}ms")
    print(f"  FPS: {fps:.1f}")
    print(f"\nDetections per frame:")
    print(f"  Avg: {avg_detections:.1f}")

    # Check against target metrics
    print("\n" + "="*60)
    print("TARGET METRICS CHECK")
    print("="*60)

    refresh_rate_hz = fps
    tracking_ok = "✓ PASS" if refresh_rate_hz >= 1.0 else "✗ FAIL"
    print(f"Refresh rate: {refresh_rate_hz:.1f} Hz (target: ≥1 Hz) {tracking_ok}")

    # Save results
    results = {
        'model_path': model_path,
        'video_path': video_path,
        'frames_processed': len(inference_times),
        'avg_inference_ms': float(avg_time * 1000),
        'min_inference_ms': float(min_time * 1000),
        'max_inference_ms': float(max_time * 1000),
        'fps': float(fps),
        'avg_detections_per_frame': float(avg_detections)
    }

    import json
    results_path = output_dir / "quick_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Quick model evaluation')
    parser.add_argument('--model', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/data/models/yolo11l.onnx',
                       help='Path to model')
    parser.add_argument('--video', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/data/test_set/cctv.mp4',
                       help='Path to test video')
    parser.add_argument('--output', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/results',
                       help='Output directory')
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames to process')

    args = parser.parse_args()

    quick_evaluation(
        model_path=args.model,
        video_path=args.video,
        output_dir=args.output,
        num_frames=args.frames
    )


if __name__ == '__main__':
    main()
