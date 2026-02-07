"""
删除 training_data 文件夹中的所有副本文件
保留原始文件
"""
import os
from pathlib import Path

def is_copy_file(filename):
    """判断是否为副本文件"""
    # 副本文件的特征：包含 "副本" 或 " - 副本"
    return '副本' in filename

def delete_copy_files_in_folder(folder_path):
    """删除指定文件夹中的所有副本文件"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"文件夹不存在: {folder}")
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
                if deleted_count <= 5:  # 只显示前5个
                    print(f"  删除: {file_path.name}")
            else:
                kept_count += 1
    
    return deleted_count, kept_count

def main():
    """主函数"""
    training_data_dir = Path('training_data')
    
    if not training_data_dir.exists():
        print("training_data 文件夹不存在！")
        return
    
    print("开始删除副本文件...")
    print("=" * 60)
    
    total_deleted = 0
    total_kept = 0
    
    # 遍历所有子文件夹
    for folder in training_data_dir.iterdir():
        if folder.is_dir():
            print(f"\n处理文件夹: {folder.name}")
            print("-" * 60)
            
            deleted, kept = delete_copy_files_in_folder(folder)
            total_deleted += deleted
            total_kept += kept
            
            if deleted > 5:
                print(f"  ... 还有 {deleted - 5} 个文件")
            
            print(f"  删除: {deleted} 个文件")
            print(f"  保留: {kept} 个文件")
    
    print("\n" + "=" * 60)
    print("删除完成！")
    print("=" * 60)
    print(f"总共删除: {total_deleted} 个副本文件")
    print(f"总共保留: {total_kept} 个原始文件")

if __name__ == '__main__':
    main()
