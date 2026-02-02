"""
清理 training_data_completed 文件夹

删除所有增强数据，只保留原始标注图
"""

import shutil
from pathlib import Path

def clean_completed_folder():
    """清理 training_data_completed 文件夹"""
    print("=" * 80)
    print("清理 training_data_completed 文件夹")
    print("=" * 80)
    
    base_dir = Path("training_data_completed")
    
    if not base_dir.exists():
        print("⚠ training_data_completed 文件夹不存在")
        return
    
    # 需要保留的文件夹（只包含原始标注图的）
    folders_to_keep = {
        '个人页_已登录_余额积分',  # 16 张原始图
        '个人页_已登录_头像首页',  # 6 张原始图
        '个人页_未登录',  # 19 张原始图
        '温馨提示',  # 17 张原始图
        '登录页',  # 30 张原始图
        '签到弹窗',  # 11 张原始图
        '签到页',  # 20 张原始图
        '首页',  # 9 张原始图
        'transfer_detector_20260127_181721',  # 54 张原始图（转账页）
    }
    
    # 需要删除的文件夹（包含增强数据或不需要的）
    folders_to_delete = []
    
    # 扫描所有子文件夹
    for folder in base_dir.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            
            # 如果不在保留列表中，标记为删除
            if folder_name not in folders_to_keep:
                folders_to_delete.append(folder)
    
    # 删除文件夹
    print(f"\n找到 {len(folders_to_delete)} 个需要删除的文件夹：")
    for folder in folders_to_delete:
        print(f"  - {folder.name}")
    
    if folders_to_delete:
        confirm = input("\n确认删除这些文件夹？(y/n): ")
        if confirm.lower() == 'y':
            for folder in folders_to_delete:
                print(f"删除: {folder.name}")
                shutil.rmtree(folder)
            print(f"\n✓ 已删除 {len(folders_to_delete)} 个文件夹")
        else:
            print("\n取消删除")
    else:
        print("\n没有需要删除的文件夹")
    
    # 统计保留的文件夹
    print("\n" + "=" * 80)
    print("保留的文件夹：")
    print("=" * 80)
    
    total_images = 0
    total_labels = 0
    
    for folder_name in sorted(folders_to_keep):
        folder_path = base_dir / folder_name
        if folder_path.exists():
            images_dir = folder_path / "images"
            labels_dir = folder_path / "labels"
            
            num_images = 0
            num_labels = 0
            
            if images_dir.exists():
                num_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
            
            if labels_dir.exists():
                num_labels = len(list(labels_dir.glob("*.txt")))
            
            print(f"{folder_name}:")
            print(f"  图片: {num_images} 张")
            print(f"  标签: {num_labels} 个")
            
            total_images += num_images
            total_labels += num_labels
    
    print("\n" + "=" * 80)
    print(f"总计: {total_images} 张图片, {total_labels} 个标签")
    print("=" * 80)

if __name__ == '__main__':
    clean_completed_folder()
