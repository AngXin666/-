#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证修复结果"""

import json
import random
from pathlib import Path

# 读取配置
with open('config/yolo_model_registry.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 检查嵌套路径
nested = [m for m, info in data['models'].items() 
          if 'runs/detect/runs/detect' in info.get('model_path', '') 
          or 'runs\\detect\\runs\\detect' in info.get('model_path', '')]

print(f"配置文件中剩余嵌套路径数量: {len(nested)}")

if nested:
    print("\n仍有嵌套路径的模型:")
    for name in nested:
        print(f"  - {name}: {data['models'][name].get('model_path', '')}")
else:
    print("✓ 配置文件中无嵌套路径")

# 检查目录结构
print("\n目录结构检查:")
nested_dir = Path("models/runs/detect/runs/detect/yolo_runs")
if nested_dir.exists():
    print(f"✗ 多级目录仍然存在: {nested_dir}")
else:
    print(f"✓ 多级目录已清理")

target_dir = Path("models/runs/detect/yolo_runs")
if target_dir.exists():
    detector_count = sum(1 for d in target_dir.iterdir() if d.is_dir())
    print(f"✓ 目标目录包含 {detector_count} 个detector")
else:
    print(f"✗ 目标目录不存在: {target_dir}")

# 随机抽样检查路径
print("\n随机抽样路径检查:")
samples = random.sample(list(data['models'].items()), min(5, len(data['models'])))
for name, info in samples:
    model_path = info.get('model_path', '')
    print(f"  {name}:")
    print(f"    {model_path}")
