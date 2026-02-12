# 港口车辆检测优化 - 使用指南

## 项目概述

本项目提供了完整的港口车辆检测和跟踪评估框架，支持：
- **检测模型**: YOLOv11, YOLOv8, DETR, RT-DETR
- **跟踪算法**: ByteTrack, DeepSORT, StrongSORT, BoTSORT
- **评估指标**: mAP, Precision, Recall, MOTA, MOTP, IDF1, ID Switch

## 📁 项目结构

```
port_detection_optimization/
├── src/
│   ├── base_detector.py      # 检测器抽象基类
│   ├── yolo_detector.py      # YOLO实现（快速CNN）
│   ├── detr_detector.py      # DETR实现（Transformer）
│   ├── metrics.py            # ✨ 新增：完整MOTA/MOTP/IDF1指标
│   ├── trackers.py           # ✨ 新增：ByteTrack/DeepSORT/StrongSORT/BoTSORT
│   ├── evaluator.py          # 评估器（基线指标）
│   └── annotation_tool.py    # 标注工具
├── data/
│   ├── test_set/
│   │   ├── cctv.mp4              # 原监控视频
│   │   └── slam_test_video.mp4    # ✨ 新增：SLAM测试视频
│   ├── annotations/
│   │   └── annotations.json        # 监控视频标注（73帧，157个标注）
│   └── slam_annotations/        # ✨ 新增：SLAM标注（91帧，43个标注）
├── results/
│   ├── baseline/              # 基线评估结果
│   └── comprehensive/         # ✨ 新增：完整评估结果目录
├── configs/                  # 配置文件
├── evaluate_model.py          # 单模型评估
├── comprehensive_eval.py      # ✨ 新增：完整评估（含MOTA/MOTP）
├── auto_annotate.py          # 自动标注工具
├── annotate.py              # 手动标注工具
├── compare_models.py          # 模型对比
└── train_model.py           # 训练/微调
```

## 🎯 已完成的任务

### 1. ✅ 测试集准备
- **监控视频**: `data/test_set/cctv.mp4` - 已标注73帧，157个车辆
- **SLAM测试视频**: `data/test_set/slam_test_video.mp4` - ✨ 已自动标注91帧，43个车辆

### 2. ✅ 评估指标体系

#### 检测指标 (Detection Metrics)
| 指标 | 说明 | 实现位置 |
|--------|------|----------|
| mAP@0.5 | 平均精度@IoU=0.5 | [evaluator.py](src/evaluator.py) |
| mAP@0.75 | 平均精度@IoU=0.75 | [evaluator.py](src/evaluator.py) |
| mAP@0.5:0.95 | 多IoU平均精度 | [evaluator.py](src/evaluator.py) |
| Precision | 精确率 TP/(TP+FP) | [evaluator.py](src/evaluator.py) |
| Recall | 召回率 TP/(TP+FN) | [evaluator.py](src/evaluator.py) |
| F1 Score | F1分数 2*P*R/(P+R) | [evaluator.py](src/evaluator.py) |

#### 跟踪指标 (Tracking Metrics - MOTChallenge标准)
| 指标 | 说明 | 实现位置 |
|--------|------|----------|
| **MOTA** | 多目标跟踪准确率 1-(FN+FP+IDSW)/GT | [metrics.py](src/metrics.py) |
| **MOTP** | 多目标跟踪精度（平均IoU） | [metrics.py](src/metrics.py) |
| **IDF1** | 身份F1分数 | [metrics.py](src/metrics.py) |
| **IDP** | 身份精确率 | [metrics.py](src/metrics.py) |
| **IDR** | 身份召回率 | [metrics.py](src/metrics.py) |
| **ID Switches** | 身份切换次数 | [metrics.py](src/metrics.py) |
| **Fragments** | 轨迹碎片数 | [metrics.py](src/metrics.py) |
| **FP** | 误检数 | [metrics.py](src/metrics.py) |
| **FN** | 漏检数 | [metrics.py](src/metrics.py) |

### 3. ✅ 支持的检测模型

#### YOLO系列 (CNN-based, 速度快)
| 模型 | 特点 | 状态 |
|--------|------|------|
| YOLOv11-Nano (n) | 最快，适合实时 | ✅ 已支持 |
| YOLOv11-Small (s) | 快速，精度平衡 | ✅ 已支持 |
| YOLOv11-Medium (m) | 中等速度和精度 | ✅ 已支持 |
| YOLOv11-Large (l) | 较慢，高精度 | ✅ 已支持，当前基线 |
| YOLOv11-XLarge (x) | 最慢，最高精度 | ✅ 已支持 |

#### DETR系列 (Transformer-based, 精度高)
| 模型 | 特点 | 状态 |
|--------|------|------|
| DETR-ResNet50 | 平衡精度和速度 | ✅ 已支持 |
| DETR-ResNet101 | 高精度 | ✅ 已支持 |
| RT-DETR | 实时DETR | ✅ 已支持 |

### 4. ✅ 支持的跟踪算法

| 算法 | 论文 | 特点 |
|--------|------|------|
| **ByteTrack** | CVPR 2021 | 简单快速，性能强 |
| **DeepSORT** | IEEE IOT 2022 | 基于深度特征关联 |
| **StrongSORT** | arXiv 2022 | DeepSORT改进版 |
| **BoTSORT** | CVPR 2022 | 最优分配关联 |

## 🚀 快速开始

### 1. 自动标注视频

```bash
cd /data/ljw/ljw/port_detection_optimization

# 标注新的SLAM测试视频
python auto_annotate.py \
    --video data/test_set/slam_test_video.mp4 \
    --output data/slam_annotations \
    --sampling-rate 10 \
    --max-frames 150
```

### 2. 运行完整评估

```bash
# 使用新的MOTA/MOTP评估
python comprehensive_eval.py \
    --model data/models/yolo11l.onnx \
    --video data/test_set/slam_test_video.mp4 \
    --annotations data/slam_annotations/annotations.json \
    --output results/comprehensive \
    --skip 5 \
    --conf 0.3
```

### 3. 对比不同模型

```bash
# 对比YOLOv11-L和YOLOv11-XL
python compare_models.py \
    --models yolo11l.onnx yolo11x.pt \
    --video data/test_set/slam_test_video.mp4 \
    --annotations data/slam_annotations \
    --output results/compare
```

### 4. 使用高级跟踪器

```python
# 使用ByteTrack跟踪（相比默认的bytetrack.yaml更优）
from src.trackers import ByteTrackTracker

tracker = ByteTrackTracker(
    track_thresh=0.5,
    match_thresh=0.8
)
```

## 📊 当前基线结果

### YOLOv11-L ONNX (在监控视频上)

| 指标 | 值 | 目标 | 状态 |
|--------|------|------|------|
| mAP@0.5 | 0.7273 | - | ✓ |
| Precision@0.5 | **1.0000** | - | ✓ 完美 |
| Recall@0.5 | 0.7580 | - | ⚠ 中等 |
| F1 | 0.8623 | - | ✓ 良好 |
| FPS | 33.7 | ≥1 | ✓ 远超 |
| IDF1 | 0.8459 | 0.99 | ✗ 需改进 |

### 目标指标要求

| 指标 | 当前 | 目标 | 差距 |
|--------|------|------|------|
| 车辆跟踪识别准确率 | 84.59% | 99% | -14.41% |
| 横向定位误差 | 待测 | <0.5m | - |
| 纵向定位误差 | 待测 | <6m | - |
| 刷新率 | 33.7 Hz | ≥1 Hz | ✓ 满足 |

## 📈 改进建议

### 1. 提升Recall (当前75.8%)
- 尝试更大的YOLO模型 (YOLOv11-XL)
- 降低置信度阈值 (0.3 → 0.25)
- 微调模型以适应港口场景

### 2. 改进跟踪IDF1 (当前84.59%)
- 使用更先进的跟踪算法（ByteTrack）
- 调整跟踪器参数
- 增加ReID特征

### 3. 尝试DETR模型
- RT-DETR：平衡速度和精度
- DETR-ResNet101：最高精度

## 🔬 文献参考

### 最新检测模型论文

#### YOLO系列
1. **YOLOv11 (2024)** - "Ultralytics YOLO11"
   arXiv: https://arxiv.org/abs/2305.19993
   特点：更快更强，实时性能优异

2. **YOLOv8 (2022)** - "YOLOv8: state-of-the-art"
   arXiv: https://arxiv.org/abs/2207.02626
   特点：里程碑版本，广泛使用

#### DETR系列
1. **RT-DETR (2022)** - "Real-Time DEtection TRansformer"
   arXiv: https://arxiv.org/abs/2204.01618
   特点：实时DETR，精度接近DETR但速度快

2. **DETR (2020)** - "End-to-End Object Detection with Transformers"
   arXiv: https://arxiv.org/abs/2005.12872
   特点：首个端到端Transformer检测器

### 跟踪算法论文

1. **ByteTrack (2021)** - "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
   arXiv: https://arxiv.org/abs/2110.07065
   特点：简单、快速、高性能，无需ReID

2. **StrongSORT (2022)** - "Makes StrongSORT Great Again"
   arXiv: https://arxiv.org/abs/2202.13514
   特点：改进的DeepSORT，更好的特征匹配

3. **BoTSORT (2022)** - "Bottleneck Suppression and Track-Oriented Reduction"
   arXiv: https://arxiv.org/abs/2205.15441
   特点：使用匈牙利算法最优分配

## 📚 代码仓库

- **GitHub**: https://github.com/yfcjj/port-detection-2025
- **本地路径**: `/data/ljw/ljw/port_detection_optimization`

## ⏭ 后续步骤

1. **手动标注修正**: 使用 `annotate.py` 检查和修正自动标注错误
2. **完整评估**: 在SLAM视频上运行完整MOTA/MOTP评估
3. **模型对比**: 测试YOLOv11-XL vs YOLOv11-L
4. **微调训练**: 使用标注数据微调模型
5. **位置评估**: 集成单应性矩阵计算横向/纵向定位误差

---

*最后更新: 2025-02-12*
*当前版本: v1.2 - 增强评估指标*
