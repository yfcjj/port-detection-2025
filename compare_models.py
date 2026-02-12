"""
Model comparison script
Compares multiple detection models on the same test set
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

from src.yolo_detector import YOLODetector, YOLOv11Detector
from src.detr_detector import DETRDetector, RTDETRDetector
from evaluate_model import evaluate_model


# Model configurations to compare
MODEL_CONFIGS = {
    # YOLO models (CNN-based, fast)
    'yolo11n': {'type': 'yolo', 'path': 'yolo11n.pt', 'desc': 'YOLOv11 Nano (fastest)'},
    'yolo11s': {'type': 'yolo', 'path': 'yolo11s.pt', 'desc': 'YOLOv11 Small'},
    'yolo11m': {'type': 'yolo', 'path': 'yolo11m.pt', 'desc': 'YOLOv11 Medium'},
    'yolo11l': {'type': 'yolo', 'path': 'yolo11l.pt', 'desc': 'YOLOv11 Large'},
    'yolo11x': {'type': 'yolo', 'path': 'yolo11x.pt', 'desc': 'YOLOv11 XLarge (most accurate)'},

    # YOLOv8
    'yolov8l': {'type': 'yolo', 'path': 'yolov8l.pt', 'desc': 'YOLOv8 Large'},

    # ONNX optimized
    'yolo11l-onnx': {'type': 'yolo', 'path': 'yolo11l.onnx', 'desc': 'YOLOv11 Large ONNX'},

    # DETR models (Transformer-based, slower)
    'detr-resnet50': {'type': 'detr', 'path': 'facebook/detr-resnet-50', 'desc': 'DETR ResNet-50'},
    'detr-resnet101': {'type': 'detr', 'path': 'facebook/detr-resnet-101', 'desc': 'DETR ResNet-101'},

    # RT-DETR (Real-time DETR)
    'rtdetr-l': {'type': 'rtdetr', 'path': 'rtdetr-l.pt', 'desc': 'RT-DETR Large'},
}


def compare_models(models: List[str], video_path: str, annotations_dir: Path,
                   output_dir: Path, **eval_kwargs) -> pd.DataFrame:
    """
    Compare multiple models on the same test set

    Args:
        models: List of model names from MODEL_CONFIGS
        video_path: Path to test video
        annotations_dir: Path to annotations
        output_dir: Output directory
        **eval_kwargs: Additional arguments for evaluation

    Returns:
        DataFrame with comparison results
    """
    results = []

    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            print(f"Warning: Unknown model '{model_name}', skipping")
            continue

        config = MODEL_CONFIGS[model_name]
        print(f"\n{'='*60}")
        print(f"Evaluating: {config['desc']}")
        print(f"{'='*60}")

        # Create detector
        if config['type'] == 'yolo':
            detector = YOLODetector(
                model_path=config['path'],
                conf_threshold=eval_kwargs.get('conf', 0.3),
                iou_threshold=eval_kwargs.get('iou', 0.45),
                use_onnx=config['path'].endswith('.onnx')
            )
        elif config['type'] == 'detr':
            detector = DETRDetector(
                model_name=config['path'],
                conf_threshold=eval_kwargs.get('conf', 0.3),
                iou_threshold=eval_kwargs.get('iou', 0.5)
            )
        elif config['type'] == 'rtdetr':
            detector = RTDETRDetector(
                model_path=config['path'],
                conf_threshold=eval_kwargs.get('conf', 0.3),
                iou_threshold=eval_kwargs.get('iou', 0.45)
            )
        else:
            print(f"Unknown model type: {config['type']}")
            continue

        # Evaluate
        try:
            model_output_dir = output_dir / model_name
            metrics = evaluate_model(
                detector=detector,
                video_path=video_path,
                annotations_dir=annotations_dir,
                output_dir=model_output_dir,
                **eval_kwargs
            )

            # Extract key metrics
            result = {
                'Model': model_name,
                'Description': config['desc'],
                'Type': config['type'],
                'mAP@0.5': metrics['detection'].get('mAP@0.5', 0),
                'mAP@0.75': metrics['detection'].get('mAP@0.75', 0),
                'mAP@0.5:0.95': metrics['detection'].get('mAP@0.5:0.95', 0),
                'Precision': metrics['detection'].get('Precision@0.5', 0),
                'Recall': metrics['detection'].get('Recall@0.5', 0),
                'F1': metrics['detection'].get('F1', 0),
                'FPS': metrics['detection'].get('fps', 0),
                'Avg_Inference_ms': metrics['detection'].get('avg_inference_time_ms', 0),
                'IDF1': metrics['tracking'].get('IDF1', 0) if metrics['tracking'] else 0,
                'IDP': metrics['tracking'].get('IDP', 0) if metrics['tracking'] else 0,
                'IDR': metrics['tracking'].get('IDR', 0) if metrics['tracking'] else 0,
            }

            results.append(result)
            print(f"✓ {model_name} completed: mAP@0.5={result['mAP@0.5']:.4f}, FPS={result['FPS']:.1f}")

        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by F1 score
    if not df.empty:
        df = df.sort_values('F1', ascending=False)

    return df


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table"""
    print("\n" + "="*100)
    print("MODEL COMPARISON RESULTS")
    print("="*100)

    # Display columns
    display_cols = ['Model', 'mAP@0.5', 'Precision', 'Recall', 'F1', 'FPS', 'Avg_Inference_ms', 'IDF1']

    print(df[display_cols].to_string(index=False))

    # Find best model for each metric
    print("\n" + "-"*60)
    print("BEST MODELS BY METRIC:")
    print("-"*60)

    metrics_to_check = ['mAP@0.5', 'Precision', 'Recall', 'F1', 'FPS', 'IDF1']

    for metric in metrics_to_check:
        if metric in df.columns and len(df) > 0:
            best_idx = df[metric].idxmax()
            best_model = df.loc[best_idx, 'Model']
            best_value = df.loc[best_idx, metric]
            print(f"{metric:20s}: {best_model:15s} ({best_value:.4f})")


def save_comparison_results(df: pd.DataFrame, output_path: str):
    """Save comparison results"""
    output_path = Path(output_path)

    # Save as CSV
    df.to_csv(output_path.with_suffix('.csv'), index=False)

    # Save as JSON
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(df.to_dict('records'), f, indent=2)

    print(f"\nComparison results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple detection models')
    parser.add_argument('--models', nargs='+', default=['yolo11l', 'yolo11x'],
                       help='Models to compare')
    parser.add_argument('--video', type=str,
                       default='/data/ljw/ljw/宁波大榭港/监控视频/2N测试右侧低_20251224111945/2N测试右侧低_20251224010000-20251224015959_1.mp4',
                       help='Path to test video')
    parser.add_argument('--annotations', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/data/annotations',
                       help='Path to annotations directory')
    parser.add_argument('--output', type=str,
                       default='/data/ljw/ljw/port_detection_optimization/results/comparison',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--skip-frames', type=int, default=5,
                       help='Process every Nth frame')
    parser.add_argument('--no-tracking', action='store_true',
                       help='Disable tracking evaluation')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare models
    df = compare_models(
        models=args.models,
        video_path=args.video,
        annotations_dir=Path(args.annotations),
        output_dir=output_dir,
        conf=args.conf,
        iou=args.iou,
        skip_frames=args.skip_frames,
        tracking=not args.no_tracking
    )

    # Print results
    if not df.empty:
        print_comparison_table(df)
        save_comparison_results(df, output_dir / "comparison_results")
    else:
        print("No models were successfully evaluated!")


if __name__ == '__main__':
    main()
