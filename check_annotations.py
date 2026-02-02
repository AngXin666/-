"""检查标注文件"""
import json

# 读取标注文件
with open("training_data/转账页/annotations.json", 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# 统计已标注的图片
annotated_count = 0
for img_path, labels in annotations.items():
    if len(labels) > 0:
        annotated_count += 1

print(f"总图片数: {len(annotations)}")
print(f"已标注的图片数: {annotated_count}")
print(f"未标注的图片数: {len(annotations) - annotated_count}")

# 显示前5个已标注的图片
print("\n前5个已标注的图片:")
count = 0
for img_path, labels in annotations.items():
    if len(labels) > 0:
        print(f"  {img_path}: {len(labels)}个标注")
        count += 1
        if count >= 5:
            break
