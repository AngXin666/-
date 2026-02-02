"""
删除 training_data 文件夹中的所有增强图片
包括：
1. 副本文件（文件名包含"副本"）
2. _augmented 文件夹
"""
import os
import shutil
from pathlib import Path

def is_copy_file(filename):
    """判断是否为副本文件"""
    return '副本' in filename

def delete_copy_files_in_folder(folder_path):
    """删除指定文件夹中的所有副本文件"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return 0, 0
    
    deleted_count = 0
    kept_count = 0
    
    # 遍历文件夹中的所有文件
    for file_path in folder.iterdir():
        if file_path.is_file():
            if is_copy_file(file_path.name):
                # 删除副本文件
                file_path.unlink()
                deleted_count += 1
            else:
                kept_count += 1
    
    return deleted_count, kept_count

def main():
    """主函数"""
    training_data_dir = Path('training_data')
    
    if not training_data_dir.exists():
        print("training_data 文件夹不存在！")
        return
    
    print("=" * 80)
    print("开始删除所有增强图片...")
    print("=" * 80)
    
    total_deleted = 0
    total_kept = 0
    deleted_folders = []
    
    # 遍历所有子文件夹
    for folder in sorted(training_data_dir.iterdir()):
        if folder.is_dir():
            # 删除 _augmented 文件夹
            if folder.name.endswith('_augmented'):
                print(f"\n删除增强文件夹: {folder.name}")
                try:
                    shutil.rmtree(folder)
                    deleted_folders.append(folder.name)
                    print(f"  ✓ 已删除整个文件夹")
                except Exception as e:
                    print(f"  ✗ 删除失败: {e}")
            else:
                # 删除文件夹中的副本文件
                print(f"\n处理文件夹: {folder.name}")
                deleted, kept = delete_copy_files_in_folder(folder)
                
                if deleted > 0:
                    total_deleted += deleted
                    total_kept += kept
                    print(f"  删除副本: {deleted} 个")
                    print(f"  保留原始: {kept} 个")
                elif kept > 0:
                    total_kept += kept
                    print(f"  保留原始: {kept} 个（无副本）")
                else:
                    print(f"  空文件夹")
    
    print("\n" + "=" * 80)
    print("删除完成！")
    print("=" * 80)
    print(f"删除副本文件: {total_deleted} 个")
    print(f"保留原始文件: {total_kept} 个")
    print(f"删除增强文件夹: {len(deleted_folders)} 个")
    if deleted_folders:
        for folder_name in deleted_folders:
            print(f"  - {folder_name}")

if __name__ == '__main__':
    main()
