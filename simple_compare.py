#!/usr/bin/env python3
"""
Simple YOLO model comparison - YOLOv11-L ONNX vs YOLOv11-X (if available)
"""
import sys
import os

# Add project directory to Python path
project_dir = '/data/ljw/ljw/port_detection_optimization'
sys.path.insert(0, project_dir)
os.chdir(project_dir)

from src.yolo_detector import YOLODetector
from evaluate_model import evaluate_model, VEHICLE_CLASSES
from pathlib import Path

def main():
    # Model configurations
    model_l = '/data/ljw/ljw/port_detection_optimization/data/models/yolo11l.onnx'
    model_x = '/data/ljw/ljw/port_detection_optimization/data/models/yolo11x.pt'

    # Check if models exist
    models_to_test = []
    if os.path.exists(model_l):
        models_to_test.append(('YOLOv11-L', model_l))
    if os.path.exists(model_x):
        models_to_test.append(('YOLOv11-X', model_x))
    else:
        print(f"Warning: {model_x} not found, will skip")

    if not models_to_test:
        print("No models found to compare!")
        print("Available models in data/models/:")
        for f in os.listdir('data/models'):
            print(f"  - {f}")
        return

    print(f"\nWill compare {len(models_to_test)} model(s):")
    for name, path in models_to_test:
        print(f"  - {name}: {path}")

    video_path = '/data/ljw/ljw/port_detection_optimization/data/test_set/slam_test_video.mp4'
    annotations_dir = 'data/slam_annotations'
    output_dir = 'results/compare'
    skip_frames = 5

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Annotations: {annotations_dir}")
    print(f"Skip frames: {skip_frames}")
    print(f"\nEvaluating models...")
    print("-"*60)

    results = {}

    for model_name, model_path in models_to_test:
        print(f"\n{model_name}...")

        # Create detector
        detector = YOLODetector(
            model_path=model_path,
            conf_threshold=0.3,
            iou_threshold=0.45,
            use_onnx=model_path.endswith('.onnx')
        )
        detector.load_model()

        # Evaluate
        try:
            metrics = evaluate_model(
                detector=detector,
                video_path=video_path,
                annotations_dir=Path(annotations_dir),
                output_dir=Path(f"{output_dir}/{model_name}"),
                skip_frames=skip_frames
            )

            # Extract key metrics
            det = metrics['detection']
            trk = metrics['tracking']

            results[model_name] = {
                'mAP@0.5': det.get('mAP@0.5', 0),
                'Precision': det.get('Precision@0.5', 0),
                'Recall': det.get('Recall@0.5', 0),
                'F1': det.get('F1', 0),
                'FPS': det.get('fps', 0),
                'IDF1': trk.get('IDF1', 0) if trk else 0
            }

            print(f"  mAP@0.5: {results[model_name]['mAP@0.5']:.4f}")
            print(f"  F1: {results[model_name]['F1']:.4f}")
            print(f"  FPS: {results[model_name]['FPS']:.1f}")
            print(f"  IDF1: {results[model_name]['IDF1']:.4f}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<15} {'mAP@0.5':<10} {'F1':<10} {'FPS':<10}")
    print("-"*60)

    for model_name in results:
        r = results[model_name]
        print(f"{model_name:<15} {r['mAP@0.5']:>10.4f} {r['F1']:>10.4f} {r['FPS']:>10.1f}")

    # Find best by each metric
    print("\n" + "-"*60)
    print("BEST MODELS BY METRIC:")
    print("-"*60)

    metrics_to_check = ['mAP@0.5', 'F1', 'FPS']
    for metric in metrics_to_check:
        best_model = max(results.keys(), key=lambda k: results[k][metric])
        best_value = results[best_model][metric]
        print(f"{metric:<15}: {best_model:<15} ({best_value:.4f})")

    # Save results
    import json
    results_path = Path(output_dir) / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
