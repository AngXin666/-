"""删除签到成功弹窗标注中的下方关闭按钮标注，保留上方的"""
import json
from pathlib import Path

# 读取原始标注文件
annotation_file = Path("原始标注图/签到成功弹窗_20260130_013633/annotations.json")

print(f"读取标注文件: {annotation_file}")

with open(annotation_file, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

print(f"原始标注数量: {len(annotations)} 张图片")

# 统计信息
total_annotations = 0
deleted_count = 0
kept_count = 0

# 删除下方的关闭按钮标注（y坐标较大的）
for image_path, labels in annotations.items():
    total_annotations += len(labels)
    
    # 找出所有关闭按钮标注
    close_buttons = [label for label in labels if label['class'] == '关闭按钮']
    other_labels = [label for label in labels if label['class'] != '关闭按钮']
    
    # 如果有多个关闭按钮，保留y坐标最小的（最上面的）
    if len(close_buttons) > 1:
        # 按y1坐标排序，保留第一个（最上面的）
        close_buttons.sort(key=lambda x: x['y1'])
        kept_button = close_buttons[0]
        deleted_count += len(close_buttons) - 1
        kept_count += 1
        
        # 重新组合标注
        annotations[image_path] = other_labels + [kept_button]
        
        print(f"\n{image_path}:")
        print(f"  保留上方按钮: y1={kept_button['y1']:.1f}")
        for i, btn in enumerate(close_buttons[1:], 1):
            print(f"  删除下方按钮{i}: y1={btn['y1']:.1f}")
    elif len(close_buttons) == 1:
        kept_count += 1
        annotations[image_path] = labels
    else:
        annotations[image_path] = labels

print(f"\n统计信息:")
print(f"  总标注数: {total_annotations}")
print(f"  删除的关闭按钮: {deleted_count}")
print(f"  保留的关闭按钮: {kept_count}")

# 保存修改后的标注文件
output_file = annotation_file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print(f"\n✓ 已保存修改后的标注文件: {output_file}")
print(f"✓ 已删除下方的关闭按钮标注")
print(f"✓ 保留了上方的关闭按钮标注")
print(f"\n现在可以使用标注工具重新标注更大的关闭按钮区域了！")
