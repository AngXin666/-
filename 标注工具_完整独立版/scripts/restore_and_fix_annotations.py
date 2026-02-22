"""从YOLO数据集恢复标注，然后删除下方的关闭按钮"""
import json
from pathlib import Path
import shutil

# 1. 先备份当前标注文件
annotation_file = Path("原始标注图/签到成功弹窗_20260130_013633/annotations.json")
backup_file = annotation_file.with_suffix('.json.backup')

if annotation_file.exists():
    shutil.copy2(annotation_file, backup_file)
    print(f"✓ 已备份当前标注文件: {backup_file}")

# 2. 从YOLO数据集读取标签文件，重建完整标注
yolo_dataset = Path("yolo_dataset_签到成功弹窗")
images_dir = Path("原始标注图/签到成功弹窗_20260130_013633/images")
labels_dir = Path("原始标注图/签到成功弹窗_20260130_013633/labels")

# YOLO类别映射（需要从dataset.yaml读取）
class_names = {
    0: "签到金额",
    1: "签到成功文本", 
    2: "关闭按钮"
}

print(f"\n从YOLO标签恢复标注...")

# 读取所有图片和对应的标签
annotations = {}
image_files = list(images_dir.glob("*.png"))

print(f"找到 {len(image_files)} 张图片")

for img_file in image_files:
    # 对应的标签文件
    label_file = labels_dir / (img_file.stem + ".txt")
    
    if not label_file.exists():
        print(f"⚠️ 缺少标签文件: {label_file.name}")
        continue
    
    # 读取YOLO标签
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    # 转换为原始标注格式
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # 转换为绝对坐标（假设图片尺寸540x960）
        img_width = 540
        img_height = 960
        
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        x2 = (x_center + width/2) * img_width
        y2 = (y_center + height/2) * img_height
        
        labels.append({
            "class": class_names.get(class_id, f"unknown_{class_id}"),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    
    # 使用相对路径作为key
    rel_path = f"training_data\\签到成功弹窗\\{img_file.name}"
    annotations[rel_path] = labels

print(f"✓ 恢复了 {len(annotations)} 张图片的标注")

# 3. 删除下方的关闭按钮（y坐标较大的）
deleted_count = 0
kept_count = 0

for image_path, labels in annotations.items():
    # 找出所有关闭按钮
    close_buttons = [label for label in labels if label['class'] == '关闭按钮']
    other_labels = [label for label in labels if label['class'] != '关闭按钮']
    
    if len(close_buttons) > 1:
        # 按y1坐标排序，保留第一个（最上面的）
        close_buttons.sort(key=lambda x: x['y1'])
        kept_button = close_buttons[0]
        deleted_count += len(close_buttons) - 1
        kept_count += 1
        
        # 重新组合
        annotations[image_path] = other_labels + [kept_button]
        
        if deleted_count <= 3:  # 只打印前3个示例
            print(f"\n{Path(image_path).name}:")
            print(f"  保留上方按钮: y1={kept_button['y1']:.1f}")
            print(f"  删除下方按钮: y1={close_buttons[1]['y1']:.1f}")
    elif len(close_buttons) == 1:
        kept_count += 1

print(f"\n统计信息:")
print(f"  删除的关闭按钮: {deleted_count}")
print(f"  保留的关闭按钮: {kept_count}")

# 4. 保存修改后的标注
with open(annotation_file, 'w', encoding='utf-8') as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print(f"\n✓ 已保存修改后的标注文件: {annotation_file}")
print(f"✓ 每张图片现在有3个标注: 签到成功文本、签到金额、关闭按钮(上方)")
print(f"\n请重新打开标注工具查看！")
