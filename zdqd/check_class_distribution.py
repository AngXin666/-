"""检查类别分布"""
import json
from collections import Counter

data = json.load(open('training_data/个人页_已登录_余额积分/annotations.json', 'r', encoding='utf-8'))

all_classes = []
for img_path, anns in data.items():
    img_classes = [ann.get('label') or ann.get('class') for ann in anns]
    all_classes.extend(img_classes)

counts = Counter(all_classes)
print('各类别标注总数:')
for k, v in sorted(counts.items()):
    print(f'  {k}: {v}')

print(f'\n总图片数: {len(data)}')
print(f'总标注数: {sum(counts.values())}')
print(f'平均每张图片标注数: {sum(counts.values()) / len(data):.1f}')

# 检查每张图片的类别组合
print('\n图片类别组合统计:')
combos = []
for anns in data.values():
    img_classes = tuple(sorted([ann.get('label') or ann.get('class') for ann in anns]))
    combos.append(img_classes)

combo_counts = Counter(combos)
for combo, count in combo_counts.most_common():
    print(f'  {combo}: {count}张')
