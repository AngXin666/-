"""
清理未标注的图片（可选）
"""
from pathlib import Path

def clean_unlabeled_images():
    """删除没有标注文件的图片"""
    training_data = Path('training_data')
    
    deleted_count = 0
    kept_count = 0
    
    print("=" * 70)
    print("清理未标注图片")
    print("=" * 70)
    print("\n扫描 training_data 目录...")
    
    for category_dir in training_data.iterdir():
        if not category_dir.is_dir():
            continue
        
        page_type = category_dir.name
        page_deleted = 0
        page_kept = 0
        
        # 查找所有图片
        for img_file in category_dir.glob("*.png"):
            label_file = img_file.with_suffix(".txt")
            
            if not label_file.exists():
                # 没有标注文件，删除图片
                print(f"  删除: {img_file.name}")
                img_file.unlink()
                deleted_count += 1
                page_deleted += 1
            else:
                kept_count += 1
                page_kept += 1
        
        if page_deleted > 0:
            print(f"\n{page_type}: 删除 {page_deleted} 张，保留 {page_kept} 张")
    
    print("\n" + "=" * 70)
    print("清理完成")
    print("=" * 70)
    print(f"\n删除: {deleted_count} 张未标注图片")
    print(f"保留: {kept_count} 张已标注图片")
    
    if deleted_count == 0:
        print("\n✅ 没有未标注图片，无需清理")
    else:
        print(f"\n✅ 已清理 {deleted_count} 张未标注图片")

if __name__ == "__main__":
    # 确认操作
    print("⚠️  警告：此操作将删除所有未标注的图片！")
    print("   （没有对应 .txt 文件的 .png 图片）")
    confirm = input("\n确认删除？(yes/no): ")
    
    if confirm.lower() == 'yes':
        clean_unlabeled_images()
    else:
        print("\n已取消操作")
