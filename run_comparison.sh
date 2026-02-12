#!/usr/bin/env python3
"""
Quick model comparison script - handles path and environment issues
"""
import sys
import os

# Add project directory to Python path
project_dir = '/data/ljw/ljw/port_detection_optimization'
sys.path.insert(0, project_dir)
os.chdir(project_dir)

# Now run compare_models
from compare_models import MODEL_CONFIGS, compare_models, main

# Override model paths with absolute paths
for key in MODEL_CONFIGS:
    if 'path' in MODEL_CONFIGS[key]:
        MODEL_CONFIGS[key]['path'] = os.path.join(project_dir, 'data', 'models', MODEL_CONFIGS[key]['path'])

print("Running model comparison from:", project_dir)
sys.exit(0)
