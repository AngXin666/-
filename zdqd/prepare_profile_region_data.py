"""
准备个人页区域检测YOLO训练数据
检测两个大区域：
1. 确认按钮区域（包含昵称和ID）
2. 其他区域（包含余额、积分、抵扣劵、优惠劵）
"""

import json
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image


def prepare_region_data():
    """准备区域检测训练数据"""
    
    print("=" * 70)
    print("准备个人页区域检测YOLO训练数据")
    print("=" * 70)
    
    # 源目录
    source_dir = Path("training_data/新已登陆页")
    annotations_file = source_dir / "annotations.json"
    
    # 目标目录
    target_dir = Path("yolo_dataset/profile_regions")
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
    
    # 类别映射
    class_to_id = {
        '确认按钮': 0,  # 昵称和ID区域
        '其他': 1,      # 余额、积分、抵扣劵、优惠劵区域
    }
    
    print(f"\n[类别映射]")
    print(f"  0: 确认按钮区域（昵称+ID）")
    print(f"  1: 数据区域（余额+积分+抵扣劵+优惠劵）")
    
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
        if converted_count % 10 == 0:
            print(f"  已处理: {converted_count}/{len(annotations)}")
    
    print(f"\n✅ 转换完成！")
    print(f"  成功: {converted_count} 张")
    print(f"  跳过: {skipped_count} 张")
    print(f"  保存位置: {target_dir}")
    
    # 创建 data.yaml
    data_yaml = target_dir / "data.yaml"
    yaml_content = f"""# 个人页区域检测数据集配置
# Profile Regions Detection Dataset

path: {target_dir.absolute()}
train: images/train
val: images/val

# 类别定义
nc: 2
names:
  0: 确认按钮区域
  1: 数据区域

# 说明：
# - 确认按钮区域：包含昵称和用户ID
# - 数据区域：包含余额、积分、抵扣劵、优惠劵
# - 检测到区域后，使用OCR识别具体内容
"""
    
    with open(data_yaml, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"  配置文件: {data_yaml}")
    
    if converted_count > 0:
        print(f"\n下一步：")
        print(f"  1. 分割数据集: python split_dataset.py --dataset profile_regions")
        print(f"  2. 开始训练: python train_profile_regions_yolo.py")


if __name__ == '__main__':
    prepare_region_data()
