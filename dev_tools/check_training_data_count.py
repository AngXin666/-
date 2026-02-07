"""检查训练数据数量"""
from pathlib import Path

training_data_dir = Path("标注工具_完整独立版/training_data")

print("="*80)
print("训练数据统计")
print("="*80)

total_images = 0
class_counts = []

for class_dir in sorted(training_data_dir.iterdir()):
    if not class_dir.is_dir():
        continue
    
    images = list(class_dir.glob("*.png"))
    count = len(images)
    total_images += count
    
    class_counts.append((class_dir.name, count))

# 按数量排序
class_counts.sort(key=lambda x: x[1])

print(f"\n总图片数: {total_images}")
print(f"类别数: {len(class_counts)}")

print(f"\n各类别图片数量（从少到多）:")
for class_name, count in class_counts:
    if count < 50:
        status = "❌ 偏少"
    elif count < 60:
        status = "⚠️  一般"
    else:
        status = "✓ 充足"
    print(f"  {status} {class_name}: {count}张")

print("\n" + "="*80)
