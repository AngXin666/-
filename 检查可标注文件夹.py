"""
检查标注工具能看到哪些文件夹
"""

from pathlib import Path

data_dir = Path("training_data")

print("=" * 80)
print("可标注的文件夹列表")
print("=" * 80)

categories = []
for item in sorted(data_dir.iterdir()):
    if item.is_dir():
        png_count = len(list(item.glob("*.png")))
        if png_count > 0:
            categories.append((item.name, png_count))

print(f"\n找到 {len(categories)} 个可标注的文件夹：\n")

for i, (name, count) in enumerate(categories, 1):
    print(f"{i:2d}. {name:30s} - {count:3d} 张图片")

print("\n" + "=" * 80)
print(f"总计: {len(categories)} 个文件夹")
