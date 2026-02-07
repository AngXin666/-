"""
直接删除所有 yolo_dataset_* 训练数据集
"""

import shutil
from pathlib import Path

def delete_datasets():
    """删除所有训练数据集"""
    print("=" * 80)
    print("删除 YOLO 训练数据集")
    print("=" * 80)
    
    # 查找所有 yolo_dataset_* 文件夹
    datasets = list(Path(".").glob("yolo_dataset_*"))
    
    if not datasets:
        print("\n没有找到 yolo_dataset_* 文件夹")
        return
    
    print(f"\n找到 {len(datasets)} 个训练数据集文件夹")
    
    # 统计信息
    total_size = 0
    for dataset in datasets:
        size = sum(f.stat().st_size for f in dataset.rglob('*') if f.is_file())
        total_size += size
    
    total_size_gb = total_size / (1024 * 1024 * 1024)
    
    print(f"总大小: {total_size_gb:.2f} GB")
    
    # 开始删除
    print("\n开始删除...")
    deleted_count = 0
    
    for dataset in datasets:
        print(f"  删除: {dataset.name}")
        shutil.rmtree(dataset)
        deleted_count += 1
    
    print(f"\n✓ 已删除 {deleted_count} 个训练数据集")
    print(f"✓ 释放了 {total_size_gb:.2f} GB 磁盘空间")

if __name__ == '__main__':
    delete_datasets()
