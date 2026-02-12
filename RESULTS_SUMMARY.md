# Port Vehicle Detection - Baseline Results

## Model Configuration
- **Model**: YOLOv11 Large (ONNX)
- **Confidence Threshold**: 0.3
- **IoU Threshold**: 0.45
- **Tracking**: ByteTrack

## Test Set
- **Video**: cctv.mp4 (6050 frames @ 25 FPS)
- **Annotated Frames**: 73 (sampled every 10 frames)
- **Total Annotations**: 157 vehicles

## Baseline Results

### Detection Metrics
| Metric | Value | Target | Status |
|--------|--------|--------|--------|
| **mAP@0.5** | 0.7273 | - | ✓ |
| **mAP@0.75** | 0.6898 | - | ✓ |
| **mAP@0.5:0.95** | 0.5928 | - | ✓ |
| **Precision@0.5** | 1.0000 | - | ✓ Excellent |
| **Recall@0.5** | 0.7580 | - | ⚠ Moderate |
| **F1 Score** | 0.8623 | - | ✓ Good |
| **True Positives** | 119 | - | ✓ |
| **False Positives** | 0 | - | ✓ Excellent |
| **False Negatives** | 38 | - | - |

### Tracking Metrics
| Metric | Value | Target | Status |
|--------|--------|--------|--------|
| **IDF1** | 0.8459 | 0.99 | ⚠ Below target |
| **IDP** | 1.0000 | - | ✓ Excellent |
| **IDR** | 0.7329 | - | ⚠ Moderate |

### Performance Metrics
| Metric | Value | Target | Status |
|--------|--------|--------|--------|
| **FPS** | 33.7 | ≥1 Hz | ✓ Pass |
| **Avg Inference Time** | 29.7 ms | ≤600 ms | ✓ Pass |
| **Min Inference Time** | 22.1 ms | - | ✓ |
| **Max Inference Time** | 7279.4 ms | - | (first frame only) |

## Analysis

### Strengths
- **Perfect Precision** (1.0): No false positives detected
- **High F1 Score** (0.86): Good balance between precision and recall
- **Excellent Speed** (33.7 FPS): Far exceeds the 1 Hz minimum requirement
- **No False Positives**: Very reliable detection

### Areas for Improvement
1. **Recall (0.758)**: Missing about 24% of vehicles
   - Could be improved with fine-tuning on port-specific data
   - Consider lowering confidence threshold or using larger model

2. **IDF1 (0.846)**: Below 99% target
   - Tracking identity switches are occurring
   - Could be improved with better tracker configuration

3. **Missing Detections (38 FN)**:
   - Small or distant vehicles may be missed
   - Partial occlusions not handled well

## Recommendations

### 1. Try Larger YOLO Models
Since speed is not an issue (33.7 FPS vs 1 Hz required), try:
- **YOLOv11-XL**: Better accuracy, slower (should still meet requirements)
- Compare with different model sizes

### 2. Fine-tune on Port Data
- Use the 73 annotated frames for fine-tuning
- Add more diverse annotations (different lighting, angles)
- Consider data augmentation for small/distant vehicles

### 3. Experiment with Transformer Models
- **RT-DETR**: Real-time DETR for better accuracy
- **DETR-ResNet101**: Higher accuracy, slower

### 4. Optimize Tracker Settings
- Experiment with different tracker configs (ByteTrack, BoTSort, StrongSORT)
- Adjust track buffer and match thresholds

## Next Steps
1. ✓ Baseline evaluation complete
2. ⏳ Test YOLOv11-XL model
3. ⏳ Fine-tune on annotated data
4. ⏳ Compare with DETR models
5. ⏳ Upload code and models to GitHub

---
*Generated: 2025-02-12*
*Test Video: cctv.mp4*
*Model: YOLOv11-L ONNX*
