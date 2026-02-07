"""
删除所有标注图片，保留原始图片
标注图片的特征：文件名包含 " - 副本"
"""
import os
import shutil

def delete_annotation_files():
    """删除 annotation_check 文件夹中所有带"副本"的标注图片"""
    annotation_dir = "annotation_check"
    
    if not os.path.exists(annotation_dir):
        print(f"错误：找不到 {annotation_dir} 文件夹")
        return
    
    deleted_count = 0
    kept_count = 0
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(annotation_dir):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                
                # 检查是否是标注图片（包含"副本"）
                if " - 副本" in file:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        if deleted_count % 100 == 0:
                            print(f"已删除 {deleted_count} 个标注文件...")
                    except Exception as e:
                        print(f"删除失败: {file_path}, 错误: {e}")
                else:
                    # 原始图片，保留
                    kept_count += 1
    
    print(f"\n删除完成！")
    print(f"删除的标注图片: {deleted_count} 个")
    print(f"保留的原始图片: {kept_count} 个")

if __name__ == "__main__":
    print("开始删除标注图片...")
    print("=" * 50)
    delete_annotation_files()
    print("=" * 50)
