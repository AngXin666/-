"""删除转账页目录中未标注的图片"""
import json
from pathlib import Path

# 读取标注文件
with open("training_data/转账页/annotations.json", 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# 获取所有已标注的图片路径
annotated_images = set()
for img_path, labels in annotations.items():
    if len(labels) > 0:
        # 转换为Path对象并获取文件名
        annotated_images.add(Path(img_path).name)

print(f"已标注的图片数: {len(annotated_images)}")

# 获取目录中所有图片
all_images = list(Path("training_data/转账页").glob("*.png"))
print(f"目录中的图片总数: {len(all_images)}")

# 找出未标注的图片
unlabeled_images = []
for img_path in all_images:
    if img_path.name not in annotated_images:
        unlabeled_images.append(img_path)

print(f"未标注的图片数: {len(unlabeled_images)}")

if len(unlabeled_images) > 0:
    print("\n未标注的图片（前10个）:")
    for img in unlabeled_images[:10]:
        print(f"  {img.name}")
    
    # 删除未标注的图片
    confirm = input(f"\n确认删除 {len(unlabeled_images)} 张未标注的图片? (y/n): ")
    if confirm.lower() == 'y':
        for img_path in unlabeled_images:
            img_path.unlink()
            print(f"已删除: {img_path.name}")
        print(f"\n✓ 已删除 {len(unlabeled_images)} 张未标注的图片")
    else:
        print("取消删除")
else:
    print("\n✓ 没有未标注的图片需要删除")
