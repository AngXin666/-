"""
详细扫描 training_data 和 training_data_completed 文件夹
"""
import os
from pathlib import Path

def count_files_by_type(folder_path):
    """统计文件夹中各类型文件的数量"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return {}
    
    counts = {
        'png': 0,
        'jpg': 0,
        'txt': 0,
        'json': 0,
        'yaml': 0,
        'pt': 0,
        'other': 0
    }
    
    # 递归统计所有文件
    try:
        for item in folder.rglob('*'):
            if item.is_file():
                ext = item.suffix.lower()
                if ext == '.png':
                    counts['png'] += 1
                elif ext in ['.jpg', '.jpeg']:
                    counts['jpg'] += 1
                elif ext == '.txt':
                    counts['txt'] += 1
                elif ext == '.json':
                    counts['json'] += 1
                elif ext == '.yaml':
                    counts['yaml'] += 1
                elif ext == '.pt':
                    counts['pt'] += 1
                else:
                    counts['other'] += 1
    except:
        pass
    
    return counts

def analyze_training_folder(folder_path):
    """分析训练数据文件夹"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return []
    
    results = []
    
    for subfolder in sorted(folder.iterdir()):
        if subfolder.is_dir():
            counts = count_files_by_type(subfolder)
            total = sum(counts.values())
            
            # 判断文件夹类型
            if counts['png'] > 0 or counts['jpg'] > 0:
                images = counts['png'] + counts['jpg']
                labels = counts['txt']
                
                if labels > 0:
                    folder_type = "标注数据"
                else:
                    folder_type = "图片数据"
                
                results.append({
                    'name': subfolder.name,
                    'type': folder_type,
                    'images': images,
                    'labels': labels,
                    'total': total,
                    'has_model': counts['pt'] > 0
                })
            elif counts['pt'] > 0:
                results.append({
                    'name': subfolder.name,
                    'type': "模型文件",
                    'images': 0,
                    'labels': 0,
                    'total': total,
                    'has_model': True
                })
            elif total > 0:
                results.append({
                    'name': subfolder.name,
                    'type': "其他",
                    'images': 0,
                    'labels': 0,
                    'total': total,
                    'has_model': False
                })
    
    return results

def main():
    """主函数"""
    print("=" * 100)
    print("训练数据文件夹详细扫描")
    print("=" * 100)
    
    # 扫描 training_data
    print("\n【training_data - 原始训练数据】")
    print("-" * 100)
    print(f"{'文件夹名称':<35} {'类型':<12} {'图片':>8} {'标签':>8} {'总文件':>10}")
    print("-" * 100)
    
    training_data = analyze_training_folder('training_data')
    
    total_images = 0
    total_labels = 0
    
    for item in training_data:
        print(f"{item['name']:<35} "
              f"{item['type']:<12} "
              f"{item['images']:>8} "
              f"{item['labels']:>8} "
              f"{item['total']:>10}")
        
        total_images += item['images']
        total_labels += item['labels']
    
    print("-" * 100)
    print(f"{'总计':<35} {'':12} {total_images:>8} {total_labels:>8}")
    
    # 扫描 training_data_completed
    print("\n\n【training_data_completed - 已完成训练数据】")
    print("-" * 100)
    print(f"{'文件夹名称':<35} {'类型':<12} {'图片':>8} {'标签':>8} {'总文件':>10} {'模型':<6}")
    print("-" * 100)
    
    completed_data = analyze_training_folder('training_data_completed')
    
    total_images_c = 0
    total_labels_c = 0
    
    for item in completed_data:
        model_mark = "✓" if item['has_model'] else ""
        print(f"{item['name']:<35} "
              f"{item['type']:<12} "
              f"{item['images']:>8} "
              f"{item['labels']:>8} "
              f"{item['total']:>10} "
              f"{model_mark:<6}")
        
        total_images_c += item['images']
        total_labels_c += item['labels']
    
    print("-" * 100)
    print(f"{'总计':<35} {'':12} {total_images_c:>8} {total_labels_c:>8}")
    
    # 总结
    print("\n" + "=" * 100)
    print("总结")
    print("=" * 100)
    print(f"training_data 文件夹数: {len(training_data)} 个")
    print(f"training_data 图片总数: {total_images} 张")
    print(f"training_data 标签总数: {total_labels} 个")
    print(f"\ntraining_data_completed 文件夹数: {len(completed_data)} 个")
    print(f"training_data_completed 图片总数: {total_images_c} 张")
    print(f"training_data_completed 标签总数: {total_labels_c} 个")

if __name__ == '__main__':
    main()
