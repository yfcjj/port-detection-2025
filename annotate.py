"""
Interactive annotation tool for port vehicle detection
Creates YOLO format annotations with track IDs
"""
import argparse
from pathlib import Path

from src.annotation_tool import AnnotationTool, create_yolo_data_yaml


def main():
    parser = argparse.ArgumentParser(description='Interactive annotation tool')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for annotations')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame number')
    parser.add_argument('--sampling-rate', type=int, default=5,
                       help='Annotate every Nth frame')
    parser.add_argument('--class-names', nargs='+', default=['vehicle'],
                       help='List of class names')

    args = parser.parse_args()

    # Create annotation tool
    tool = AnnotationTool(
        video_path=args.video,
        output_dir=args.output,
        class_names=args.class_names
    )

    # Start annotation
    print(f"\nStarting annotation for: {args.video}")
    print(f"Output directory: {args.output}")
    print(f"Sampling rate: {args.sampling_rate}")
    print("\nPress any key to start...\n")

    tool.annotate(
        start_frame=args.start_frame,
        sampling_rate=args.sampling_rate
    )

    print(f"\nAnnotation complete!")
    print(f"Total annotated frames: {len(tool.annotations)}")


if __name__ == '__main__':
    main()
