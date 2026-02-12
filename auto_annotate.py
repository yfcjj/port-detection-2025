"""
Automatic annotation tool using existing detection model
Pre-annotates video frames to speed up manual annotation process
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
import json
import time

from src.yolo_detector import YOLODetector
from src.annotation_tool import Annotation, FrameAnnotations


def auto_annotate_video(model_path: str, video_path: str, output_dir: str,
                        conf_threshold: float = 0.3, sampling_rate: int = 5,
                        max_frames: int = None, class_filter: list = None):
    """
    Automatically annotate video using detection model

    Args:
        model_path: Path to YOLO model
        video_path: Path to input video
        output_dir: Directory to save annotations
        conf_threshold: Confidence threshold for detections
        sampling_rate: Process every Nth frame
        max_frames: Maximum number of frames to process
        class_filter: List of class IDs to keep (None = all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("AUTOMATIC ANNOTATION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Sampling rate: {sampling_rate}")

    # Load model
    print(f"\nLoading model...")
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        use_onnx=model_path.endswith('.onnx')
    )
    detector.load_model()

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video info: {total_frames} frames @ {fps:.2f} FPS")

    # Process frames
    annotations = {}
    track_id = 1

    frame_indices = list(range(0, total_frames, sampling_rate))
    if max_frames:
        frame_indices = frame_indices[:max_frames]

    print(f"\nProcessing {len(frame_indices)} frames...")

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        h, w = frame.shape[:2]

        # Run detection
        result = detector.detect(frame)

        if len(result.boxes) == 0:
            continue

        # Create frame annotations
        frame_ann = FrameAnnotations(frame_idx)

        # Filter classes (keep only vehicles: car=2, bus=5, truck=7)
        vehicle_classes = [2, 5, 7] if class_filter is None else class_filter

        for j, box in enumerate(result.boxes):
            cls_id = int(result.class_ids[j])

            # Filter by class
            if cls_id not in vehicle_classes:
                continue

            # Convert to normalized YOLO format
            x_center = ((box[0] + box[2]) / 2) / w
            y_center = ((box[1] + box[3]) / 2) / h
            width = (box[2] - box[0]) / w
            height = (box[3] - box[1]) / h

            ann = Annotation(
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                class_id=0,  # Map to single 'vehicle' class
                track_id=track_id
            )

            frame_ann.add_annotation(ann)
            track_id += 1

        annotations[frame_idx] = frame_ann

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(frame_indices)} frames ({len(annotations)} frames with detections)")

    cap.release()

    # Save annotations
    print(f"\nSaving {len(annotations)} annotated frames...")

    # Save as JSON (includes track IDs)
    json_path = output_dir / "annotations.json"
    data = {
        'video_path': video_path,
        'class_names': ['vehicle'],
        'total_frames': total_frames,
        'annotated_frames': len(annotations),
        'frames': {}
    }

    for frame_id in sorted(annotations.keys()):
        frame_ann = annotations[frame_id]
        data['frames'][str(frame_id)] = {
            'frame_id': frame_id,
            'annotations': [a.to_dict() for a in frame_ann.annotations]
        }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Save as YOLO format
    for frame_id in sorted(annotations.keys()):
        frame_ann = annotations[frame_id]
        yolo_path = output_dir / f"frame_{frame_id:06d}.txt"

        with open(yolo_path, 'w') as f:
            for ann in frame_ann.annotations:
                f.write(f"{ann.class_id} {ann.x_center:.6f} {ann.y_center:.6f} "
                       f"{ann.width:.6f} {ann.height:.6f}\n")

    # Also save a sample frame for verification
    print("\nSaving sample visualization...")
    cap = cv2.VideoCapture(video_path)

    sample_frames = sorted(annotations.keys())[:3]
    for frame_id in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if ret:
            vis_frame = draw_annotations_on_frame(frame, annotations[frame_id])
            sample_path = output_dir / f"sample_frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(sample_path), vis_frame)

    cap.release()

    print(f"\n✓ Auto-annotation complete!")
    print(f"  - Annotations saved to: {output_dir}")
    print(f"  - Total annotated frames: {len(annotations)}")
    print(f"  - Total annotations: {sum(len(a.annotations) for a in annotations.values())}")

    print("\nNext steps:")
    print("  1. Review sample images to verify accuracy")
    print("  2. Run manual annotation tool to correct errors:")
    print(f"     python annotate.py --video {video_path} --output {output_dir}")

    return annotations


def draw_annotations_on_frame(image: np.ndarray, frame_ann: FrameAnnotations) -> np.ndarray:
    """Draw annotations on frame"""
    img = image.copy()
    h, w = img.shape[:2]

    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]

    for i, ann in enumerate(frame_ann.annotations):
        # Convert normalized to pixel
        x_center = int(ann.x_center * w)
        y_center = int(ann.y_center * h)
        bw = int(ann.width * w)
        bh = int(ann.height * h)

        x1 = x_center - bw // 2
        y1 = y_center - bh // 2
        x2 = x_center + bw // 2
        y2 = y_center + bh // 2

        color = colors[ann.track_id % len(colors)]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"ID:{ann.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw + 5, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def main():
    parser = argparse.ArgumentParser(description='Auto-annotate video using detection model')
    parser.add_argument('--model', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/data/models/yolo11l.onnx',
                       help='Path to YOLO model')
    parser.add_argument('--video', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/data/test_set/cctv.mp4',
                       help='Path to input video')
    parser.add_argument('--output', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/data/annotations',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--sampling-rate', type=int, default=5,
                       help='Process every Nth frame')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')

    args = parser.parse_args()

    auto_annotate_video(
        model_path=args.model,
        video_path=args.video,
        output_dir=args.output,
        conf_threshold=args.conf,
        sampling_rate=args.sampling_rate,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()
