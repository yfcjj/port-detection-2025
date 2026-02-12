# Port Vehicle Detection Optimization

## Overview

This project focuses on optimizing vehicle detection models for port monitoring applications. It provides:
- Modular detection framework supporting YOLO (CNN) and DETR (Transformer) models
- Comprehensive evaluation metrics (mAP, precision, recall, IDF1, etc.)
- Interactive annotation tool for creating ground truth labels
- Training/fine-tuning pipeline for custom models
- Model comparison utilities

## Project Structure

```
port_detection_optimization/
├── configs/
│   └── config.yaml          # Main configuration file
├── src/
│   ├── base_detector.py    # Abstract base classes for detectors
│   ├── yolo_detector.py    # YOLO implementation (fast, CNN-based)
│   ├── detr_detector.py    # DETR implementation (Transformer-based)
│   ├── evaluator.py        # Evaluation metrics (mAP, IDF1, etc.)
│   └── annotation_tool.py # Interactive annotation tool
├── data/
│   ├── test_set/           # Test video files
│   ├── annotations/        # Ground truth annotations
│   └── models/             # Trained model files
├── results/                # Evaluation results
├── annotate.py             # Run annotation tool
├── evaluate_model.py       # Evaluate single model
├── compare_models.py       # Compare multiple models
├── train_model.py          # Train/fine-tune models
└── requirements.txt
```

## Target Metrics

- Vehicle tracking accuracy: 99%
- Lateral positioning error: < 0.5m
- Longitudinal positioning error: < 6m
- Output refresh rate: ≥ 1Hz

## Installation

```bash
cd /data/ljw/ljw/port_detection_optimization
pip install -r requirements.txt
```

## Usage

### 1. Annotate Test Video

Create ground truth annotations with track IDs:

```bash
python annotate.py \
    --video /path/to/test_video.mp4 \
    --output data/annotations \
    --sampling-rate 5 \
    --class-names vehicle
```

**Controls:**
- Mouse: Draw bounding boxes
- SPACE: Next frame
- A: Previous frame
- D: Delete last annotation
- S: Save annotations
- N: Skip to next sampling point
- Q: Quit and save

### 2. Evaluate a Model

Evaluate a single model on annotated test set:

```bash
python evaluate_model.py \
    --model yolo11l.onnx \
    --model-type yolo \
    --video /path/to/test_video.mp4 \
    --annotations data/annotations \
    --output results/yolo11l \
    --conf 0.3 \
    --iou 0.45
```

### 3. Compare Multiple Models

Compare different models on the same test set:

```bash
python compare_models.py \
    --models yolo11l yolo11x detr-resnet101 rtdetr-l \
    --video /path/to/test_video.mp4 \
    --annotations data/annotations \
    --output results/comparison
```

### 4. Train/Fine-tune Model

Fine-tune a pre-trained model on your data:

```bash
python train_model.py \
    --model yolo11l.pt \
    --annotations data/annotations \
    --video /path/to/source_video.mp4 \
    --epochs 100 \
    --batch 16 \
    --device cuda
```

## Supported Models

### YOLO Models (CNN-based, Fast)
- YOLOv11: n, s, m, l, x variants
- YOLOv8: n, s, m, l, x variants
- ONNX optimized versions

### DETR Models (Transformer-based)
- facebook/detr-resnet-50
- facebook/detr-resnet-101
- RT-DETR (Real-time DETR)

## Annotation Format

The project uses YOLO format with track IDs:

```
class_id x_center y_center width height # track_id=X
```

Example:
```
0 0.512345 0.345678 0.123456 0.098765 # track_id=1
0 0.678901 0.456789 0.145678 0.112345 # track_id=2
```

## Evaluation Metrics

### Detection Metrics
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.75**: Mean Average Precision at IoU=0.75
- **mAP@0.5:0.95**: Mean AP across IoU thresholds 0.5 to 0.95
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: Harmonic mean of precision and recall

### Tracking Metrics
- **IDF1**: Identity F1 score
- **IDP**: Identity Precision
- **IDR**: Identity Recall

### Timing Metrics
- **FPS**: Frames per second
- **Avg Inference Time**: Average inference time in milliseconds

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
models:
  yolo:
    conf_threshold: 0.3
    iou_threshold: 0.45

target_metrics:
  tracking_accuracy: 0.99
  lateral_error_m: 0.5
  longitudinal_error_m: 6.0
  refresh_rate_hz: 1.0
```

## Quick Start Example

```bash
# 1. Annotate a few frames from your test video
python annotate.py \
    --video /data/ljw/ljw/宁波大榭港/监控视频/test.mp4 \
    --output data/annotations

# 2. Evaluate the current YOLO model
python evaluate_model.py \
    --model yolo11l.onnx \
    --model-type yolo \
    --video /data/ljw/ljw/宁波大榭港/监控视频/test.mp4 \
    --annotations data/annotations

# 3. Try the latest YOLOv11 XLarge model
python compare_models.py \
    --models yolo11l yolo11x \
    --video /data/ljw/ljw/宁波大榭港/监控视频/test.mp4 \
    --annotations data/annotations

# 4. Fine-tune on your annotated data
python train_model.py \
    --model yolo11l.pt \
    --annotations data/annotations \
    --video /data/ljw/ljw/宁波大榭港/监控视频/train.mp4 \
    --epochs 50
```

## License

This project is for research and development purposes.
