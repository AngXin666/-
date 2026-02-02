"""
检查已完成训练的training_data文件夹
找出可以安全删除的文件夹（已保存原始图 AND 已注册模型）
"""
import json
from pathlib import Path


def normalize_name(name):
    """标准化名称，用于匹配"""
    # 移除时间戳后缀
    if '_20' in name:
        name = name.split('_20')[0]
    # 移除特殊后缀
    name = name.replace('_temp_augmented', '')
    return name.strip()


def main():
    # 读取模型注册表
    registry_file = Path("yolo_model_registry.json")
    if not registry_file.exists():
        print("❌ 找不到模型注册表")
        return
    
    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # 获取已注册的模型（从models字典中）
    registered_models = set()
    for model_key, model_info in registry.get('models', {}).items():
        page_type = model_info.get('page_type', '')
        if page_type:
            registered_models.add(normalize_name(page_type))
    
    # 也检查顶层的模型注册（如coupon_detector）
    if 'coupon_detector' in registry:
        registered_models.add('我的优惠劵')
    if 'category_page' in registry:
        registered_models.add('分类页')
    
    print(f"📝 已注册的模型 ({len(registered_models)}个):")
    for model in sorted(registered_models):
        print(f"  - {model}")
    
    # 获取已保存原始图的页面类型
    original_dir = Path("原始标注图")
    saved_originals = set()
    if original_dir.exists():
        for folder in original_dir.iterdir():
            if folder.is_dir():
                saved_originals.add(normalize_name(folder.name))
    
    print(f"\n📦 已保存原始图的页面 ({len(saved_originals)}个):")
    for page in sorted(saved_originals):
        print(f"  - {page}")
    
    # 获取training_data中的所有文件夹
    training_data_dir = Path("training_data")
    training_folders = []
    if training_data_dir.exists():
        for folder in training_data_dir.iterdir():
            if folder.is_dir():
                training_folders.append(folder.name)
    
    print(f"\n📂 training_data中的文件夹 ({len(training_folders)}个):")
    for folder in sorted(training_folders):
        print(f"  - {folder}")
    
    # 分析哪些可以删除
    print(f"\n{'='*60}")
    print(f"分析结果")
    print(f"{'='*60}\n")
    
    can_delete = []
    cannot_delete = []
    
    for folder in training_folders:
        normalized = normalize_name(folder)
        
        # 跳过临时增强文件夹
        if '_temp_augmented' in folder:
            can_delete.append({
                'folder': folder,
                'reason': '临时增强文件夹（可以直接删除）'
            })
            continue
        
        has_original = normalized in saved_originals
        has_model = normalized in registered_models
        
        if has_original and has_model:
            can_delete.append({
                'folder': folder,
                'reason': '✓ 已保存原始图 + ✓ 已注册模型'
            })
        elif not has_original and not has_model:
            cannot_delete.append({
                'folder': folder,
                'reason': '✗ 未保存原始图 + ✗ 未注册模型'
            })
        elif not has_original:
            cannot_delete.append({
                'folder': folder,
                'reason': '✗ 未保存原始图（但已注册模型）'
            })
        elif not has_model:
            cannot_delete.append({
                'folder': folder,
                'reason': '✗ 未注册模型（但已保存原始图）'
            })
    
    # 打印可以删除的文件夹
    print(f"✅ 可以安全删除的文件夹 ({len(can_delete)}个):\n")
    for item in can_delete:
        print(f"  📁 {item['folder']}")
        print(f"     原因: {item['reason']}\n")
    
    # 打印不能删除的文件夹
    if cannot_delete:
        print(f"\n⚠️  不能删除的文件夹 ({len(cannot_delete)}个):\n")
        for item in cannot_delete:
            print(f"  📁 {item['folder']}")
            print(f"     原因: {item['reason']}\n")
    
    # 生成删除命令
    if can_delete:
        print(f"\n{'='*60}")
        print(f"删除命令")
        print(f"{'='*60}\n")
        print(f"如需删除这些文件夹，可以运行以下命令：\n")
        
        for item in can_delete:
            folder_path = f"training_data/{item['folder']}"
            print(f'rmdir /s /q "{folder_path}"')
        
        print(f"\n或者使用Python脚本批量删除：")
        print(f"python delete_completed_training_folders.py")
    
    # 保存报告
    report_path = Path("training_data_cleanup_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("training_data 清理报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"生成时间: {Path(__file__).stat().st_mtime}\n\n")
        
        f.write(f"可以安全删除的文件夹 ({len(can_delete)}个):\n\n")
        for item in can_delete:
            f.write(f"  - {item['folder']}\n")
            f.write(f"    {item['reason']}\n\n")
        
        if cannot_delete:
            f.write(f"\n不能删除的文件夹 ({len(cannot_delete)}个):\n\n")
            for item in cannot_delete:
                f.write(f"  - {item['folder']}\n")
                f.write(f"    {item['reason']}\n\n")
    
    print(f"\n📄 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
