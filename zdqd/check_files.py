"""检查文件状态"""
from pathlib import Path
import json
import os
from datetime import datetime

# 检查原始标注文件
source_ann = Path('training_data/个人页_已登录/annotations.json')
target_ann = Path('training_data/个人页_已登录_余额积分/annotations.json')

print("文件状态检查:")
print(f"原始标注文件: {source_ann}")
print(f"  存在: {source_ann.exists()}")
if source_ann.exists():
    mtime = os.path.getmtime(source_ann)
    print(f"  修改时间: {datetime.fromtimestamp(mtime)}")
    data = json.load(open(source_ann, 'r', encoding='utf-8'))
    print(f"  图片数: {len(data)}")

print(f"\n筛选后标注文件: {target_ann}")
print(f"  存在: {target_ann.exists()}")
if target_ann.exists():
    mtime = os.path.getmtime(target_ann)
    print(f"  修改时间: {datetime.fromtimestamp(mtime)}")
    data = json.load(open(target_ann, 'r', encoding='utf-8'))
    print(f"  图片数: {len(data)}")
