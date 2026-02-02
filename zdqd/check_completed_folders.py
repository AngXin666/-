"""
检查 training_data_completed 文件夹中的标注图数量
"""
import os
from pathlib import Path

def count_images_in_folder(folder_path):
    """统计文件夹中的图片数量"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return 0
    
    # 检查是否有 images 子文件夹（YOLO格式）
    images_train = folder / 'images' / 'train'
    images_val = folder / 'images' / 'val'
    
    if images_train.exists() or images_val.exists():
        # YOLO 格式
        train_count = len(list(images_train.glob('*.png'))) + len(list(images_train.glob('*.jpg'))) if images_train.exists() else 0
        val_count = len(list(images_val.glob('*.png'))) + len(list(images_val.glob('*.jpg'))) if images_val.exists() else 0
        return train_count + val_count
    else:
        # 直接在文件夹中
        return len(list(folder.glob('*.png'))) + len(list(folder.glob('*.jpg')))

def main():
    """主函数"""
    completed_dir = Path('training_data_completed')
    
    if not completed_dir.exists():
        print("training_data_completed 文件夹不存在！")
        return
    
    print("=" * 80)
    print("training_data_completed 文件夹统计")
    print("=" * 80)
    
    folders_50_plus = []
    folders_less_50 = []
    
    # 遍历所有子文件夹
    for folder in sorted(completed_dir.iterdir()):
        if folder.is_dir():
            image_count = count_images_in_folder(folder)
            
            folder_info = {
                'name': folder.name,
                'count': image_count
            }
            
            if image_count >= 50:
                folders_50_plus.append(folder_info)
            else:
                folders_less_50.append(folder_info)
    
    # 显示 50+ 图片的文件夹
    if folders_50_plus:
        print("\n【标注图 ≥ 50 张的文件夹】")
        print("-" * 80)
        print(f"{'文件夹名称':<50} {'图片数量':>10}")
        print("-" * 80)
        
        for info in folders_50_plus:
            print(f"{info['name']:<50} {info['count']:>10}")
        
        print("-" * 80)
        print(f"{'总计':<50} {len(folders_50_plus):>10} 个文件夹")
    else:
        print("\n【标注图 ≥ 50 张的文件夹】")
        print("-" * 80)
        print("没有找到标注图 ≥ 50 张的文件夹")
    
    # 显示 < 50 图片的文件夹
    if folders_less_50:
        print("\n【标注图 < 50 张的文件夹】")
        print("-" * 80)
        print(f"{'文件夹名称':<50} {'图片数量':>10}")
        print("-" * 80)
        
        for info in folders_less_50:
            print(f"{info['name']:<50} {info['count']:>10}")
        
        print("-" * 80)
        print(f"{'总计':<50} {len(folders_less_50):>10} 个文件夹")
    
    print("\n" + "=" * 80)
    print("统计完成！")
    print("=" * 80)
    print(f"标注图 ≥ 50 张: {len(folders_50_plus)} 个文件夹")
    print(f"标注图 < 50 张: {len(folders_less_50)} 个文件夹")
    print(f"总文件夹数: {len(folders_50_plus) + len(folders_less_50)} 个")

if __name__ == '__main__':
    main()
