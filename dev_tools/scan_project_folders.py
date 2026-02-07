"""
扫描整个项目文件夹，检查每个文件夹存放的内容
"""
import os
from pathlib import Path
from collections import defaultdict

def get_folder_info(folder_path):
    """获取文件夹信息"""
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        return None
    
    # 统计文件类型
    file_types = defaultdict(int)
    total_files = 0
    total_size = 0
    subfolder_count = 0
    
    try:
        for item in folder.iterdir():
            if item.is_file():
                total_files += 1
                ext = item.suffix.lower()
                file_types[ext if ext else '(无扩展名)'] += 1
                try:
                    total_size += item.stat().st_size
                except:
                    pass
            elif item.is_dir():
                subfolder_count += 1
    except PermissionError:
        return None
    
    return {
        'name': folder.name,
        'total_files': total_files,
        'total_size': total_size,
        'subfolder_count': subfolder_count,
        'file_types': dict(file_types)
    }

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_folder_description(folder_name, info):
    """根据文件夹名称和内容推断用途"""
    file_types = info['file_types']
    
    # 根据文件夹名称判断
    if 'training_data' in folder_name:
        if '_completed' in folder_name:
            return "已完成训练的数据集"
        elif '_augmented' in folder_name:
            return "数据增强后的训练数据"
        else:
            return "原始训练数据（标注图）"
    
    elif 'yolo_dataset' in folder_name:
        return "YOLO 格式的数据集"
    
    elif 'yolo_runs' in folder_name or 'runs' in folder_name:
        return "YOLO 训练运行结果"
    
    elif folder_name in ['src', 'tests']:
        return "源代码文件"
    
    elif folder_name == 'config':
        return "配置文件"
    
    elif folder_name in ['login_cache', 'runtime_data']:
        return "运行时数据"
    
    elif 'screenshot' in folder_name:
        return "截图文件"
    
    elif folder_name in ['build', 'dist']:
        return "构建输出"
    
    elif folder_name == 'reports':
        return "报告文件"
    
    elif folder_name == 'data':
        return "数据文件"
    
    elif folder_name in ['dev_tools', 'templates']:
        return "开发工具/模板"
    
    # 根据文件类型判断
    elif '.py' in file_types:
        return "Python 脚本"
    elif '.png' in file_types or '.jpg' in file_types:
        if '.txt' in file_types:
            return "标注图片（图片+标签）"
        else:
            return "图片文件"
    elif '.pt' in file_types or '.h5' in file_types:
        return "模型文件"
    elif '.yaml' in file_types or '.json' in file_types:
        return "配置/数据文件"
    
    return "其他"

def main():
    """主函数"""
    project_root = Path('.')
    
    print("=" * 100)
    print("项目文件夹扫描报告")
    print("=" * 100)
    
    # 获取所有一级子文件夹
    folders = []
    for item in sorted(project_root.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            info = get_folder_info(item)
            if info:
                info['description'] = get_folder_description(item.name, info)
                folders.append(info)
    
    # 按文件数量排序
    folders.sort(key=lambda x: x['total_files'], reverse=True)
    
    # 显示结果
    print(f"\n{'文件夹名称':<35} {'文件数':>8} {'大小':>12} {'子文件夹':>10} {'用途描述':<30}")
    print("-" * 100)
    
    total_files = 0
    total_size = 0
    
    for info in folders:
        print(f"{info['name']:<35} "
              f"{info['total_files']:>8} "
              f"{format_size(info['total_size']):>12} "
              f"{info['subfolder_count']:>10} "
              f"{info['description']:<30}")
        
        total_files += info['total_files']
        total_size += info['total_size']
    
    print("-" * 100)
    print(f"{'总计':<35} {total_files:>8} {format_size(total_size):>12}")
    
    # 显示详细的文件类型统计
    print("\n" + "=" * 100)
    print("重点文件夹详细信息")
    print("=" * 100)
    
    important_folders = [
        'training_data', 'training_data_completed',
        'yolo_dataset', 'yolo_dataset_transfer',
        'src', 'tests'
    ]
    
    for folder_name in important_folders:
        folder_path = project_root / folder_name
        if folder_path.exists():
            info = get_folder_info(folder_path)
            if info:
                print(f"\n【{folder_name}】")
                print(f"  文件数: {info['total_files']}")
                print(f"  大小: {format_size(info['total_size'])}")
                print(f"  子文件夹: {info['subfolder_count']}")
                print(f"  文件类型:")
                for ext, count in sorted(info['file_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"    {ext}: {count} 个")

if __name__ == '__main__':
    main()
