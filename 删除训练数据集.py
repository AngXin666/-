"""
删除 yolo_dataset_* 训练数据集

这些数据集包含增强后的图片，占用大量空间
原始标注图已保存在 原始标注图/ 文件夹中
需要重新训练时可以重新生成
"""

import shutil
from pathlib import Path

def delete_yolo_datasets():
    """删除所有 yolo_dataset_* 文件夹"""
    print("=" * 80)
    print("删除 YOLO 训练数据集")
    print("=" * 80)
    
    # 查找所有 yolo_dataset_* 文件夹
    datasets = list(Path(".").glob("yolo_dataset_*"))
    
    if not datasets:
        print("\n没有找到 yolo_dataset_* 文件夹")
        return
    
    print(f"\n找到 {len(datasets)} 个训练数据集文件夹：")
    print("-" * 80)
    
    total_size = 0
    dataset_info = []
    
    for dataset in datasets:
        # 统计文件夹大小
        size = sum(f.stat().st_size for f in dataset.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        
        # 统计图片数量
        num_images = len(list(dataset.rglob('*.png'))) + len(list(dataset.rglob('*.jpg')))
        
        dataset_info.append({
            'name': dataset.name,
            'size_mb': size_mb,
            'num_images': num_images
        })
        
        total_size += size
        
        print(f"\n{dataset.name}:")
        print(f"  大小: {size_mb:.2f} MB")
        print(f"  图片: {num_images} 张")
    
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    
    print("\n" + "-" * 80)
    print(f"总大小: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")
    print(f"总图片: {sum(d['num_images'] for d in dataset_info)} 张")
    
    # 确认删除
    print("\n" + "=" * 80)
    print("⚠️  注意：")
    print("  - 这些数据集包含增强后的训练数据")
    print("  - 原始标注图已保存在 原始标注图/ 文件夹中")
    print("  - 需要重新训练时可以重新生成")
    print("  - 删除后可以节省 {:.2f} GB 磁盘空间".format(total_size_gb))
    print("=" * 80)
    
    confirm = input("\n确认删除这些训练数据集？(y/n): ")
    
    if confirm.lower() == 'y':
        print("\n开始删除...")
        for dataset in datasets:
            print(f"  删除: {dataset.name}")
            shutil.rmtree(dataset)
        
        print(f"\n✓ 已删除 {len(datasets)} 个训练数据集")
        print(f"✓ 释放了 {total_size_gb:.2f} GB 磁盘空间")
    else:
        print("\n取消删除")

if __name__ == '__main__':
    delete_yolo_datasets()
