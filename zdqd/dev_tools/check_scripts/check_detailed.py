"""详细检查每张图片的标注"""
import json
from collections import Counter

data = json.load(open('training_data/个人页_已登录_余额积分/annotations.json', 'r', encoding='utf-8'))

# 统计每张图片的类别组合
print("每张图片的标注情况:")
print("=" * 80)

# 按标注数量分组
by_count = {}
for img_path, anns in data.items():
    count = len(anns)
    if count not in by_count:
        by_count[count] = []
    
    classes = sorted([ann.get('label') or ann.get('class') for ann in anns])
    by_count[count].append((img_path, classes))

for count in sorted(by_count.keys()):
    imgs = by_count[count]
    print(f"\n标注{count}个元素的图片: {len(imgs)}张")
    
    # 统计类别组合
    combos = Counter([tuple(classes) for _, classes in imgs])
    for combo, cnt in combos.most_common():
        print(f"  {combo}: {cnt}张")
        # 显示第一张图片的文件名
        for img_path, classes in imgs:
            if tuple(classes) == combo:
                from pathlib import Path
                print(f"    示例: {Path(img_path).name}")
                break

print("\n" + "=" * 80)
print(f"总计: {len(data)}张图片")
print(f"\n有4个元素的图片数: {len(by_count.get(4, []))}")
