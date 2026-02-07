"""检查损坏的图片"""
from pathlib import Path
from PIL import Image

dataset_dir = Path("page_classifier_dataset")
corrupted_files = []

print("检查所有图片...")
for class_dir in sorted(dataset_dir.iterdir()):
    if not class_dir.is_dir():
        continue
    
    print(f"\n检查类别: {class_dir.name}")
    class_corrupted = []
    
    for img_path in class_dir.glob("*.png"):
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            class_corrupted.append(str(img_path))
            print(f"  损坏: {img_path.name}")
    
    if class_corrupted:
        corrupted_files.extend(class_corrupted)
        print(f"  ✗ 发现 {len(class_corrupted)} 个损坏文件")
    else:
        print(f"  ✓ 无损坏文件")

print(f"\n{'='*60}")
print(f"总计发现 {len(corrupted_files)} 个损坏文件")
if corrupted_files:
    print("\n是否删除这些损坏文件？(y/n)")
