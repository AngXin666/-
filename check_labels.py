from pathlib import Path
from collections import Counter

counts = Counter()
label_dir = Path('yolo_dataset_pages/首页/labels/train')

for f in label_dir.glob('*.txt'):
    with open(f, 'r') as file:
        for line in file:
            if line.strip():
                cls = int(line.split()[0])
                counts[cls] += 1

print('训练集标签统计:')
for k, v in sorted(counts.items()):
    print(f'  类别 {k}: {v} 个')
