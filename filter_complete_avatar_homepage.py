"""
筛选同时包含头像和首页按钮的图片
"""
import json
from pathlib import Path
import shutil

def filter_complete_data():
    """筛选同时包含头像和首页按钮的图片"""
    
    source_dir = Path("training_data/个人页_已登录_头像首页")
    annotations_file = source_dir / "annotations.json"
    
    # 读取标注
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print("=" * 80)
    print("筛选同时包含头像和首页按钮的图片")
    print("=" * 80)
    
    # 筛选完整数据
    complete_data = {}
    incomplete_images = []
    
    for image_path, boxes in annotations.items():
        classes = [box['class'] for box in boxes]
        
        # 检查是否同时包含头像和首页按钮
        has_avatar = '头像' in classes
        has_homepage = '首页按钮' in classes
        
        if has_avatar and has_homepage and len(boxes) == 2:
            complete_data[image_path] = boxes
        else:
            incomplete_images.append((image_path, classes))
    
    print(f"\n完整数据: {len(complete_data)}张（同时包含头像和首页按钮）")
    print(f"不完整数据: {len(incomplete_images)}张")
    
    if incomplete_images:
        print("\n不完整的图片:")
        for img, classes in incomplete_images:
            img_name = Path(img).name
            print(f"  {img_name}: {classes}")
    
    # 保存筛选后的标注
    filtered_annotations_file = source_dir / "annotations_complete.json"
    with open(filtered_annotations_file, 'w', encoding='utf-8') as f:
        json.dump(complete_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 筛选完成")
    print(f"  完整标注保存到: {filtered_annotations_file}")
    print(f"  共 {len(complete_data)} 张图片")
    print("=" * 80)
    
    return len(complete_data)

if __name__ == "__main__":
    filter_complete_data()
