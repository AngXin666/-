"""检查训练数据数量"""
from pathlib import Path

data_dir = Path('training_data')
folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.endswith('_temp_augmented')])

print(f"{'文件夹':<30} {'图片数量':>10}")
print("=" * 45)

for folder in folders:
    count = len(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    print(f"{folder.name:<30} {count:>10}")
