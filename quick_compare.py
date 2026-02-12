"""
Quick model comparison script
Tests multiple models quickly with minimal processing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.yolo_detector import YOLODetector
from evaluate_model import evaluate_model, VEHICLE_CLASSES


def quick_test_model(model_path: str, model_name: str, video_path: str,
                   annotations_dir: str, output_base: str):
    """Quick test a single model"""
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    output_dir = Path(output_base) / model_name.replace('/', '_')

    try:
        detector = YOLODetector(
            model_path=model_path,
            conf_threshold=0.3,
            iou_threshold=0.45,
            use_onnx=model_path.endswith('.onnx')
        )

        # Run evaluation with reduced frames for speed
        metrics = evaluate_model(
            detector=detector,
            video_path=video_path,
            annotations_dir=Path(annotations_dir),
            output_dir=output_dir,
            tracking=True,
            skip_frames=10  # Process every 10th frame
        )

        det_metrics = metrics['detection']
        track_metrics = metrics['tracking']

        result = {
            'model': model_name,
            'mAP@0.5': det_metrics.get('mAP@0.5', 0),
            'Precision@0.5': det_metrics.get('Precision@0.5', 0),
            'Recall@0.5': det_metrics.get('Recall@0.5', 0),
            'F1': det_metrics.get('F1', 0),
            'FPS': det_metrics.get('fps', 0),
            'IDF1': track_metrics.get('IDF1', 0) if track_metrics else 0,
        }

        print(f"\n✓ {model_name} Complete!")
        print(f"  mAP@0.5: {result['mAP@0.5']:.4f}")
        print(f"  F1: {result['F1']:.4f}")
        print(f"  FPS: {result['FPS']:.1f}")
        print(f"  IDF1: {result['IDF1']:.4f}")

        return result

    except Exception as e:
        print(f"\n✗ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Quick model comparison')
    parser.add_argument('--models', nargs='+',
                       default=['yolo11l.onnx', 'yolo11x.pt'],
                       help='Models to compare')
    parser.add_argument('--video', type=str,
                       default='data/test_set/cctv.mp4',
                       help='Test video')
    parser.add_argument('--annotations', type=str,
                       default='data/annotations',
                       help='Annotations directory')
    parser.add_argument('--output', type=str,
                       default='results/quick_compare',
                       help='Output directory')

    args = parser.parse_args()

    results = []

    for model in args.models:
        result = quick_test_model(
            model_path=model,
            model_name=model,
            video_path=args.video,
            annotations_dir=args.annotations,
            output_base=args.output
        )

        if result:
            results.append(result)

    # Print comparison
    if results:
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(f"{'Model':<20} {'mAP@0.5':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FPS':<10} {'IDF1':<10}")
        print("-"*80)

        for r in results:
            print(f"{r['model']:<20} {r['mAP@0.5']:<10.4f} {r['Precision@0.5']:<10.4f} "
                  f"{r['Recall@0.5']:<10.4f} {r['F1']:<10.4f} {r['FPS']:<10.1f} {r['IDF1']:<10.4f}")

        # Find best by each metric
        print("\nBEST BY METRIC:")
        for metric in ['mAP@0.5', 'F1', 'FPS', 'IDF1']:
            best = max(results, key=lambda x: x[metric])
            print(f"  {metric:<10}: {best['model']:<20} ({best[metric]:.4f})")


if __name__ == '__main__':
    main()
