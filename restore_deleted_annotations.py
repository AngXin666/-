"""
恢复被误删的原始标注图

从 yolo_dataset_* 文件夹中恢复原始图片和标签到 training_data_completed
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def count_files(directory):
    """统计目录中的文件数量"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def restore_from_yolo_dataset(yolo_dataset_path, target_folder_name, completed_base="training_data_completed"):
    """
    从 YOLO 数据集恢复图片和标签
    
    Args:
        yolo_dataset_path: YOLO 数据集路径（如 yolo_dataset_balance）
        target_folder_name: 目标文件夹名称（如 个人页_已登录_余额积分）
        completed_base: training_data_completed 基础路径
    """
    # 创建目标文件夹
    target_dir = Path(completed_base) / target_folder_name
    target_images_dir = target_dir / "images"
    target_labels_dir = target_dir / "labels"
    
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLO 数据集路径
    yolo_path = Path(yolo_dataset_path)
    
    # 统计信息
    stats = {
        'images_copied': 0,
        'labels_copied': 0,
        'images_skipped': 0,
        'labels_skipped': 0
    }
    
    # 从 train 和 val 文件夹中恢复
    for split in ['train', 'val']:
        # 恢复图片
        source_images = yolo_path / "images" / split
        if source_images.exists():
            for img_file in source_images.glob("*.png"):
                target_file = target_images_dir / img_file.name
                if not target_file.exists():
                    shutil.copy2(img_file, target_file)
                    stats['images_copied'] += 1
                else:
                    stats['images_skipped'] += 1
            
            for img_file in source_images.glob("*.jpg"):
                target_file = target_images_dir / img_file.name
                if not target_file.exists():
                    shutil.copy2(img_file, target_file)
                    stats['images_copied'] += 1
                else:
                    stats['images_skipped'] += 1
        
        # 恢复标签
        source_labels = yolo_path / "labels" / split
        if source_labels.exists():
            for label_file in source_labels.glob("*.txt"):
                target_file = target_labels_dir / label_file.name
                if not target_file.exists():
                    shutil.copy2(label_file, target_file)
                    stats['labels_copied'] += 1
                else:
                    stats['labels_skipped'] += 1
    
    return stats

def main():
    """主函数"""
    print("=" * 80)
    print("恢复被误删的原始标注图")
    print("=" * 80)
    
    # 定义需要恢复的数据集
    datasets_to_restore = [
        {
            'yolo_dataset': 'yolo_dataset_balance',
            'target_folder': '个人页_已登录_余额积分',
            'description': '余额积分检测模型'
        },
        {
            'yolo_dataset': 'yolo_dataset_avatar_homepage',
            'target_folder': '个人页_已登录_头像首页',
            'description': '头像和首页按钮检测模型'
        },
        {
            'yolo_dataset': 'yolo_dataset_startup_popup',
            'target_folder': '启动页服务弹窗',
            'description': '启动页服务弹窗检测模型'
        },
        {
            'yolo_dataset': 'yolo_dataset_login_exception',
            'target_folder': '登录异常',
            'description': '登录异常检测模型（合并数据集）'
        }
    ]
    
    # 总统计
    total_stats = defaultdict(int)
    
    # 恢复每个数据集
    for dataset_info in datasets_to_restore:
        print(f"\n处理: {dataset_info['description']}")
        print(f"  YOLO 数据集: {dataset_info['yolo_dataset']}")
        print(f"  目标文件夹: {dataset_info['target_folder']}")
        
        # 检查 YOLO 数据集是否存在
        if not os.path.exists(dataset_info['yolo_dataset']):
            print(f"  ⚠ YOLO 数据集不存在，跳过")
            continue
        
        # 恢复数据
        stats = restore_from_yolo_dataset(
            dataset_info['yolo_dataset'],
            dataset_info['target_folder']
        )
        
        # 更新总统计
        for key, value in stats.items():
            total_stats[key] += value
        
        # 打印统计
        print(f"  ✓ 图片复制: {stats['images_copied']} 张")
        print(f"  ✓ 标签复制: {stats['labels_copied']} 个")
        if stats['images_skipped'] > 0:
            print(f"  - 图片跳过: {stats['images_skipped']} 张（已存在）")
        if stats['labels_skipped'] > 0:
            print(f"  - 标签跳过: {stats['labels_skipped']} 个（已存在）")
    
    # 打印总统计
    print("\n" + "=" * 80)
    print("恢复完成！")
    print("=" * 80)
    print(f"总计:")
    print(f"  图片复制: {total_stats['images_copied']} 张")
    print(f"  标签复制: {total_stats['labels_copied']} 个")
    print(f"  图片跳过: {total_stats['images_skipped']} 张")
    print(f"  标签跳过: {total_stats['labels_skipped']} 个")
    
    # 验证恢复结果
    print("\n" + "=" * 80)
    print("验证恢复结果")
    print("=" * 80)
    
    for dataset_info in datasets_to_restore:
        target_dir = Path("training_data_completed") / dataset_info['target_folder']
        if target_dir.exists():
            images_dir = target_dir / "images"
            labels_dir = target_dir / "labels"
            
            num_images = count_files(images_dir) if images_dir.exists() else 0
            num_labels = count_files(labels_dir) if labels_dir.exists() else 0
            
            print(f"\n{dataset_info['target_folder']}:")
            print(f"  图片: {num_images} 张")
            print(f"  标签: {num_labels} 个")

if __name__ == '__main__':
    main()
