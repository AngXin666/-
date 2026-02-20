#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化广告页训练图片
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 广告页目录
ad_dir = Path("标注工具_完整独立版/training_data/广告页")

# 获取所有原始图片（不含增强图）
image_files = [f for f in ad_dir.glob("*.png") if "_aug_" not in f.name]
image_files = sorted(image_files)[:10]  # 只看前10张

print(f"找到 {len(image_files)} 张图片")

# 创建图表
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('广告页训练数据（前10张）', fontsize=16)

for idx, img_path in enumerate(image_files):
    row = idx // 5
    col = idx % 5
    
    # 加载图片
    img = Image.open(img_path)
    
    # 显示图片
    axes[row, col].imshow(img)
    axes[row, col].set_title(img_path.name[:30], fontsize=8)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

print("\n图片文件名:")
for img_path in image_files:
    print(f"  - {img_path.name}")
