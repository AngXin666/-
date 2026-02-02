"""
删除 training_data_completed 中的增强数据

只保留原始标注图
"""

import shutil
from pathlib import Path

def delete_augmented_data():
    """删除增强数据"""
    print("=" * 80)
    print("删除 training_data_completed 中的增强数据")
    print("=" * 80)
    
    base_dir = Path("training_data_completed")
    
    if not base_dir.exists():
        print("⚠ training_data_completed 文件夹不存在")
        return
    
    # 需要删除的文件夹
    folders_to_delete = [
        '启动页服务弹窗',  # 432 张增强数据
        '登录异常',  # 1500 张增强数据
        'transfer_detector_20260127_171430',  # 64 张（旧版本）
        '手机号码不存在',  # 0 张图片
        '用户名或密码错误弹窗',  # 0 张图片
    ]
    
    print("\n需要删除的文件夹：")
    print("-" * 80)
    
    total_images_to_delete = 0
    
    for folder_name in folders_to_delete:
        folder_path = base_dir / folder_name
        if folder_path.exists():
            images_dir = folder_path / "images"
            num_images = 0
            
            if images_dir.exists():
                num_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
            
            print(f"  {folder_name}: {num_images} 张图片")
            total_images_to_delete += num_images
        else:
            print(f"  {folder_name}: 不存在")
    
    print("-" * 80)
    print(f"总计: {total_images_to_delete} 张图片将被删除")
    
    # 删除文件夹
    print("\n开始删除...")
    deleted_count = 0
    
    for folder_name in folders_to_delete:
        folder_path = base_dir / folder_name
        if folder_path.exists():
            print(f"  删除: {folder_name}")
            shutil.rmtree(folder_path)
            deleted_count += 1
    
    print(f"\n✓ 已删除 {deleted_count} 个文件夹")
    
    # 统计保留的文件夹
    print("\n" + "=" * 80)
    print("保留的原始标注图文件夹：")
    print("=" * 80)
    
    total_images = 0
    total_labels = 0
    
    remaining_folders = []
    for folder in base_dir.iterdir():
        if folder.is_dir():
            remaining_folders.append(folder.name)
    
    for folder_name in sorted(remaining_folders):
        folder_path = base_dir / folder_name
        images_dir = folder_path / "images"
        labels_dir = folder_path / "labels"
        
        num_images = 0
        num_labels = 0
        
        if images_dir.exists():
            num_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
        
        if labels_dir.exists():
            num_labels = len(list(labels_dir.glob("*.txt")))
        
        print(f"\n{folder_name}:")
        print(f"  图片: {num_images} 张")
        print(f"  标签: {num_labels} 个")
        
        total_images += num_images
        total_labels += num_labels
    
    print("\n" + "=" * 80)
    print(f"总计: {total_images} 张原始标注图, {total_labels} 个标签")
    print("=" * 80)

if __name__ == '__main__':
    delete_augmented_data()
