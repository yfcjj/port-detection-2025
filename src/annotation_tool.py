"""
Annotation tool for creating ground truth labels
Supports YOLO format with track IDs
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import yaml


class Annotation:
    """Single annotation with box, class, and track ID"""

    def __init__(self, x_center: float, y_center: float, width: float,
                 height: float, class_id: int, track_id: int):
        """
        Args:
            x_center, y_center, width, height: Normalized coordinates [0-1]
            class_id: Class ID (0 for vehicle)
            track_id: Unique track identifier
        """
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.class_id = class_id
        self.track_id = track_id

    def to_yolo_format(self) -> str:
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'x_center': float(self.x_center),
            'y_center': float(self.y_center),
            'width': float(self.width),
            'height': float(self.height),
            'class_id': int(self.class_id),
            'track_id': int(self.track_id)
        }

    @staticmethod
    def from_dict(data: Dict) -> 'Annotation':
        """Create from dictionary"""
        return Annotation(
            x_center=data['x_center'],
            y_center=data['y_center'],
            width=data['width'],
            height=data['height'],
            class_id=data['class_id'],
            track_id=data['track_id']
        )

    @staticmethod
    def from_yolo_line(line: str, track_id: int) -> 'Annotation':
        """Parse YOLO format line with track ID"""
        parts = line.strip().split()
        return Annotation(
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
            class_id=int(parts[0]),
            track_id=track_id
        )


class FrameAnnotations:
    """Annotations for a single frame"""

    def __init__(self, frame_id: int, annotations: List[Annotation] = None):
        self.frame_id = frame_id
        self.annotations = annotations or []

    def add_annotation(self, annotation: Annotation):
        self.annotations.append(annotation)

    def get_next_track_id(self) -> int:
        """Get next available track ID"""
        if not self.annotations:
            return 1
        return max(a.track_id for a in self.annotations) + 1


class AnnotationTool:
    """
    Interactive annotation tool for video frames
    Creates YOLO format annotations with track IDs
    """

    def __init__(self, video_path: str, output_dir: str,
                 class_names: List[str] = None):
        """
        Args:
            video_path: Path to input video
            output_dir: Directory to save annotations
            class_names: List of class names
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names or ['vehicle']
        self.current_class_id = 0

        # Colors for different tracks
        self.colors = self._generate_colors(100)

        # Annotation state
        self.annotations: Dict[int, FrameAnnotations] = {}
        self.current_frame = 0
        self.total_frames = 0
        self.video_capture = None
        self.current_image = None

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_box = None

        # Track ID state
        self.next_track_id = 1
        self.current_track_id = None

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n distinct colors"""
        colors = []
        for i in range(n):
            hue = (i * 360 // n) % 360
            color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8),
                                  cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def open_video(self):
        """Open video file"""
        self.video_capture = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.total_frames == 0:
            raise ValueError(f"Could not open video: {self.video_path}")

        print(f"Video opened: {self.total_frames} frames")

    def load_existing_annotations(self):
        """Load existing annotations if available"""
        # Try loading JSON format with track IDs
        json_path = self.output_dir / "annotations.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                for frame_id, ann_data in data['frames'].items():
                    frame_ann = FrameAnnotations(int(frame_id))
                    for a in ann_data['annotations']:
                        frame_ann.add_annotation(Annotation.from_dict(a))
                    self.annotations[int(frame_id)] = frame_ann
                    self.next_track_id = max(self.next_track_id,
                                            max(a.track_id for a in frame_ann.annotations) + 1)
            print(f"Loaded {len(self.annotations)} frames from existing annotations")

    def save_annotations(self):
        """Save all annotations to JSON and YOLO format"""
        # Save as JSON (includes track IDs)
        json_path = self.output_dir / "annotations.json"
        data = {
            'video_path': self.video_path,
            'class_names': self.class_names,
            'frames': {}
        }

        for frame_id in sorted(self.annotations.keys()):
            frame_ann = self.annotations[frame_id]
            data['frames'][str(frame_id)] = {
                'frame_id': frame_id,
                'annotations': [a.to_dict() for a in frame_ann.annotations]
            }

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save as YOLO format (per frame)
        for frame_id in sorted(self.annotations.keys()):
            frame_ann = self.annotations[frame_id]
            yolo_path = self.output_dir / f"frame_{frame_id:06d}.txt"

            with open(yolo_path, 'w') as f:
                for ann in frame_ann.annotations:
                    f.write(ann.to_yolo_format() + f" # track_id={ann.track_id}\n")

        # Save metadata
        meta_path = self.output_dir / "metadata.yaml"
        metadata = {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'annotated_frames': len(self.annotations),
            'class_names': self.class_names,
            'num_classes': len(self.class_names)
        }

        with open(meta_path, 'w') as f:
            yaml.dump(metadata, f)

        print(f"Saved {len(self.annotations)} annotated frames to {self.output_dir}")

    def get_frame(self, frame_id: int) -> np.ndarray:
        """Get specific frame from video"""
        if self.video_capture is None:
            self.open_video()

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.video_capture.read()

        if not ret:
            raise ValueError(f"Could not read frame {frame_id}")

        return frame

    def draw_annotations(self, image: np.ndarray, frame_id: int) -> np.ndarray:
        """Draw all annotations for a frame"""
        img = image.copy()

        if frame_id not in self.annotations:
            return img

        frame_ann = self.annotations[frame_id]

        for ann in frame_ann.annotations:
            # Convert normalized coordinates to pixel
            h, w = img.shape[:2]

            x_center = int(ann.x_center * w)
            y_center = int(ann.y_center * h)
            bw = int(ann.width * w)
            bh = int(ann.height * h)

            x1 = x_center - bw // 2
            y1 = y_center - bh // 2
            x2 = x_center + bw // 2
            y2 = y_center + bh // 2

            # Get color for this track
            color = self.colors[ann.track_id % len(self.colors)]

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"ID:{ann.track_id} {self.class_names[ann.class_id]}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw + 5, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                self.current_box = (self.start_point, (x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y

                # Ensure proper ordering
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Only add if box has size
                if x2 > x1 and y2 > y1:
                    self._add_annotation_from_box(x1, y1, x2, y2)

                self.start_point = None
                self.current_box = None

    def _add_annotation_from_box(self, x1: int, y1: int, x2: int, y2: int):
        """Add annotation from pixel coordinates"""
        h, w = self.current_image.shape[:2]

        # Convert to YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        # Create annotation
        ann = Annotation(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            class_id=self.current_class_id,
            track_id=self.next_track_id
        )

        # Add to frame
        if self.current_frame not in self.annotations:
            self.annotations[self.current_frame] = FrameAnnotations(self.current_frame)

        self.annotations[self.current_frame].add_annotation(ann)
        self.next_track_id += 1

    def annotate(self, start_frame: int = 0, sampling_rate: int = 5):
        """
        Start interactive annotation

        Args:
            start_frame: Starting frame number
            sampling_rate: Annotate every Nth frame
        """
        self.open_video()
        self.load_existing_annotations()

        cv2.namedWindow("Annotation Tool")
        cv2.setMouseCallback("Annotation Tool", self.mouse_callback)

        self.current_frame = start_frame

        print("=" * 60)
        print("ANNOTATION TOOL")
        print("=" * 60)
        print("Controls:")
        print("  Mouse: Draw bounding boxes")
        print("  SPACE: Next frame")
        print("  A: Previous frame")
        print("  D: Delete last annotation")
        print("  S: Save annotations")
        print("  N: Skip to next sampling point")
        print("  G: Go to frame")
        print("  Q: Quit")
        print("=" * 60)

        while True:
            # Get current frame
            self.current_image = self.get_frame(self.current_frame)

            # Draw existing annotations
            display_img = self.draw_annotations(self.current_image, self.current_frame)

            # Draw current box being drawn
            if self.current_box:
                pt1, pt2 = self.current_box
                cv2.rectangle(display_img, pt1, pt2, (0, 255, 0), 2)

            # Add info text
            info_text = f"Frame: {self.current_frame}/{self.total_frames}"
            if self.current_frame in self.annotations:
                info_text += f" | Annotations: {len(self.annotations[self.current_frame].annotations)}"

            cv2.putText(display_img, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Sampling indicator
            if self.current_frame % sampling_rate != 0:
                cv2.putText(display_img, "(Sampling frame - skip with N)",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Annotation Tool", display_img)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                # Save and quit
                self.save_annotations()
                break

            elif key == ord(' '):
                # Next frame
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)

            elif key == ord('a'):
                # Previous frame
                self.current_frame = max(self.current_frame - 1, 0)

            elif key == ord('d'):
                # Delete last annotation
                if self.current_frame in self.annotations:
                    if self.annotations[self.current_frame].annotations:
                        self.annotations[self.current_frame].annotations.pop()
                        self.next_track_id -= 1

            elif key == ord('s'):
                # Save
                self.save_annotations()

            elif key == ord('n'):
                # Skip to next sampling point
                self.current_frame = min(self.current_frame + sampling_rate,
                                       self.total_frames - 1)

            elif key == ord('g'):
                # Go to frame
                frame_str = input("Enter frame number: ")
                try:
                    self.current_frame = max(0, min(int(frame_str), self.total_frames - 1))
                except ValueError:
                    pass

        cv2.destroyAllWindows()
        if self.video_capture:
            self.video_capture.release()


def create_yolo_data_yaml(output_dir: str, class_names: List[str],
                           train_path: str, val_path: str):
    """
    Create YOLO data.yaml configuration file

    Args:
        output_dir: Output directory
        class_names: List of class names
        train_path: Path to training images
        val_path: Path to validation images
    """
    data = {
        'path': str(Path(output_dir).absolute()),
        'train': train_path,
        'val': val_path,
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }

    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Created data.yaml at: {yaml_path}")
    return yaml_path
