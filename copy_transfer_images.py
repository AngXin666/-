"""
复制转账页图片到 original_annotations
"""

import shutil
from pathlib import Path

# 源目录
src_train = Path('training_data_completed/transfer_detector_20260127_181721/images/train')
src_val = Path('training_data_completed/transfer_detector_20260127_181721/images/val')
src_labels_train = Path('training_data_completed/transfer_detector_20260127_181721/labels/train')
src_labels_val = Path('training_data_completed/transfer_detector_20260127_181721/labels/val')

# 目标目录
dst_images = Path('original_annotations/transfer_detector_20260127_181721/images')
dst_labels = Path('original_annotations/transfer_detector_20260127_181721/labels')

# 复制图片
images_copied = 0
for img_file in src_train.glob('*.png'):
    shutil.copy2(img_file, dst_images / img_file.name)
    images_copied += 1

for img_file in src_val.glob('*.png'):
    shutil.copy2(img_file, dst_images / img_file.name)
    images_copied += 1

# 复制标签
labels_copied = 0
for label_file in src_labels_train.glob('*.txt'):
    shutil.copy2(label_file, dst_labels / label_file.name)
    labels_copied += 1

for label_file in src_labels_val.glob('*.txt'):
    shutil.copy2(label_file, dst_labels / label_file.name)
    labels_copied += 1

print(f"已复制 {images_copied} 张图片")
print(f"已复制 {labels_copied} 个标签")
