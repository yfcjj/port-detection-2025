"""
Training/Fine-tuning script for YOLO models
Supports training from scratch or fine-tuning existing models
"""
import argparse
import yaml
from pathlib import Path
import shutil

from src.yolo_detector import YOLOv11Detector, YOLODetector
from src.annotation_tool import create_yolo_data_yaml


def prepare_yolo_dataset(annotations_dir: Path, output_dir: Path,
                         video_path: str, split_ratio: float = 0.8):
    """
    Prepare YOLO dataset from annotations

    Args:
        annotations_dir: Directory with annotations.json
        output_dir: Output directory for YOLO dataset
        video_path: Path to source video
        split_ratio: Train/val split ratio
    """
    import json
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create image and label directories
    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Load annotations
    json_path = annotations_dir / "annotations.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Annotations not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get class names
    class_names = data.get('class_names', ['vehicle'])

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Process each annotated frame
    frames = sorted([int(k) for k in data['frames'].keys()])

    # Split into train/val
    split_idx = int(len(frames) * split_ratio)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]

    print(f"Preparing dataset...")
    print(f"Total annotated frames: {len(frames)}")
    print(f"Train frames: {len(train_frames)}")
    print(f"Val frames: {len(val_frames)}")

    # Process train frames
    for frame_id in train_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            continue

        # Save image
        img_path = images_train / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(img_path), frame)

        # Save labels in YOLO format
        frame_data = data['frames'][str(frame_id)]
        label_path = labels_train / f"frame_{frame_id:06d}.txt"

        with open(label_path, 'w') as f:
            for ann in frame_data['annotations']:
                # YOLO format: class_id x_center y_center width height
                f.write(f"{ann['class_id']} {ann['x_center']:.6f} "
                       f"{ann['y_center']:.6f} {ann['width']:.6f} "
                       f"{ann['height']:.6f}\n")

    # Process val frames
    for frame_id in val_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            continue

        # Save image
        img_path = images_val / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(img_path), frame)

        # Save labels
        frame_data = data['frames'][str(frame_id)]
        label_path = labels_val / f"frame_{frame_id:06d}.txt"

        with open(label_path, 'w') as f:
            for ann in frame_data['annotations']:
                f.write(f"{ann['class_id']} {ann['x_center']:.6f} "
                       f"{ann['y_center']:.6f} {ann['width']:.6f} "
                       f"{ann['height']:.6f}\n")

    cap.release()

    # Create data.yaml
    data_yaml_path = create_yolo_data_yaml(
        output_dir=output_dir,
        class_names=class_names,
        train_path="images/train",
        val_path="images/val"
    )

    print(f"\nDataset prepared at: {output_dir}")
    print(f"Train images: {len(list(images_train.glob('*.jpg')))}")
    print(f"Val images: {len(list(images_val.glob('*.jpg')))}")

    return data_yaml_path


def train_yolo(model_name: str = "yolo11l.pt", data_yaml: str = None,
               annotations_dir: str = None, video_path: str = None,
               epochs: int = 100, batch: int = 16, imgsz: int = 640,
               device: str = "cuda", output_dir: str = "runs/train",
               fine_tune: bool = True, **kwargs):
    """
    Train or fine-tune YOLO model

    Args:
        model_name: Base model to use
        data_yaml: Path to data.yaml (skip dataset preparation if provided)
        annotations_dir: Path to annotations (for dataset preparation)
        video_path: Path to source video (for dataset preparation)
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        device: Device to use
        output_dir: Output directory
        fine_tune: Whether to fine-tune (True) or train from scratch
        **kwargs: Additional training arguments
    """
    print("="*60)
    print("YOLO TRAINING")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device}")

    # Prepare dataset if data_yaml not provided
    if data_yaml is None:
        if annotations_dir is None or video_path is None:
            raise ValueError("Must provide either data_yaml or (annotations_dir + video_path)")

        print("\nPreparing YOLO dataset...")
        dataset_output = Path(output_dir) / "dataset"
        data_yaml_path = prepare_yolo_dataset(
            annotations_dir=Path(annotations_dir),
            output_dir=dataset_output,
            video_path=video_path
        )
        data_yaml = str(data_yaml_path)
    else:
        print(f"\nUsing existing data.yaml: {data_yaml}")

    # Create detector
    detector = YOLODetector(
        model_path=model_name,
        device=device
    )
    detector.load_model()

    # Train
    print(f"\nStarting training...")
    results = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=output_dir,
        name="train",
        **kwargs
    )

    print(f"\nTraining complete!")
    print(f"Best model saved to: {results['model'].save_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--model', type=str, default='yolo11l.pt',
                       help='Base model to use (or fine-tune)')
    parser.add_argument('--data-yaml', type=str, default=None,
                       help='Path to data.yaml (skip dataset preparation)')
    parser.add_argument('--annotations', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/data/annotations',
                       help='Path to annotations directory')
    parser.add_argument('--video', type=str,
                       default='/data/ljw/ljw/宁波大榭港/监控视频/2N测试右侧低_20251224111945/2N测试右侧低_20251224010000-20251224015959_1.mp4',
                       help='Path to source video')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default='runs/train',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')

    args = parser.parse_args()

    train_yolo(
        model_name=args.model,
        data_yaml=args.data_yaml,
        annotations_dir=args.annotations,
        video_path=args.video,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output,
        workers=args.workers,
        patience=args.patience
    )


if __name__ == '__main__':
    main()
