"""
准备个人页YOLO训练数据
从 training_data/新已登陆页 准备数据到 yolo_dataset/profile_numbers
"""

import json
import shutil
from pathlib import Path
from collections import Counter


def prepare_yolo_data():
    """准备YOLO训练数据"""
    
    print("=" * 70)
    print("准备个人页YOLO训练数据")
    print("=" * 70)
    
    # 源目录
    source_dir = Path("training_data/新已登陆页")
    annotations_file = source_dir / "annotations.json"
    
    # 目标目录
    target_dir = Path("yolo_dataset/profile_numbers")
    target_images = target_dir / "images" / "train"
    target_labels = target_dir / "labels" / "train"
    
    # 创建目录
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[源数据]")
    print(f"  目录: {source_dir}")
    
    # 检查标注文件
    if not annotations_file.exists():
        print(f"\n❌ 标注文件不存在: {annotations_file}")
        return
    
    # 加载标注
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"  标注文件: {annotations_file}")
    print(f"  图片数量: {len(annotations)}")
    
    # 统计标注类别
    all_classes = []
    for img_path, anns in annotations.items():
        for ann in anns:
            all_classes.append(ann['class'])
    
    class_counts = Counter(all_classes)
    print(f"\n[标注类别统计]")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # 类别映射（根据你的说明）
    # "确认按钮" 区域包含昵称和ID
    # "其他" 区域包含余额、积分、抵扣劵、优惠劵
    
    print(f"\n⚠️  当前标注需要细化：")
    print(f"  - '确认按钮' 区域需要分别标注：昵称文本、用户ID")
    print(f"  - '其他' 区域需要分别标注：余额数字、积分数字、抵扣劵数字、优惠劵数字")
    print(f"\n建议：")
    print(f"  1. 在标注工具中打开 '新已登陆页' 类别")
    print(f"  2. 重新标注，使用正确的类别：")
    print(f"     - 昵称文本（第23项）")
    print(f"     - 用户ID（第24项）")
    print(f"     - 余额数字（第21项）")
    print(f"     - 积分数字（第22项）")
    print(f"     - 抵扣劵数字（第15项）")
    print(f"     - 优惠劵数字（第16项）")
    print(f"  3. 保存后重新运行此脚本")
    
    # 检查是否有正确的标注
    valid_classes = {'昵称文本', '用户ID', '余额数字', '积分数字', '抵扣劵数字', '优惠劵数字'}
    has_valid = any(cls in valid_classes for cls in all_classes)
    
    if not has_valid:
        print(f"\n❌ 没有找到有效的标注类别")
        print(f"   请先在标注工具中重新标注")
        return
    
    # 类别ID映射
    class_to_id = {
        '余额数字': 0,
        '积分数字': 1,
        '抵扣劵数字': 2,
        '优惠劵数字': 3,
        '昵称文本': 4,
        '用户ID': 5,
    }
    
    print(f"\n[开始转换]")
    converted_count = 0
    skipped_count = 0
    
    for img_path_str, anns in annotations.items():
        img_path = Path(img_path_str)
        
        # 检查图片是否存在
        if not img_path.exists():
            print(f"  ⚠ 图片不存在: {img_path.name}")
            skipped_count += 1
            continue
        
        # 过滤有效标注
        valid_anns = [ann for ann in anns if ann['class'] in class_to_id]
        
        if not valid_anns:
            skipped_count += 1
            continue
        
        # 获取图片尺寸
        from PIL import Image
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # 转换为YOLO格式
        yolo_lines = []
        for ann in valid_anns:
            class_id = class_to_id[ann['class']]
            
            # 计算中心点和宽高（归一化）
            x_center = ((ann['x1'] + ann['x2']) / 2) / img_width
            y_center = ((ann['y1'] + ann['y2']) / 2) / img_height
            width = (ann['x2'] - ann['x1']) / img_width
            height = (ann['y2'] - ann['y1']) / img_height
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # 复制图片
        target_img = target_images / img_path.name
        shutil.copy2(img_path, target_img)
        
        # 保存标注
        target_label = target_labels / (img_path.stem + ".txt")
        with open(target_label, 'w', encoding='utf-8') as f:
            f.writelines(yolo_lines)
        
        converted_count += 1
    
    print(f"\n✅ 转换完成！")
    print(f"  成功: {converted_count} 张")
    print(f"  跳过: {skipped_count} 张")
    print(f"  保存位置: {target_dir}")
    
    if converted_count > 0:
        print(f"\n下一步：")
        print(f"  1. 运行: python split_dataset.py")
        print(f"  2. 运行: python train_profile_numbers_yolo.py")


if __name__ == '__main__':
    prepare_yolo_data()
