"""
审查YOLO模型注册表中的所有路径
"""
import json
import os

def check_model_paths():
    """检查所有模型路径"""
    
    registry_path = 'models/yolo_model_registry.json'
    
    with open(registry_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    models = data.get('models', {})
    
    print(f"=" * 80)
    print(f"YOLO模型路径审查")
    print(f"=" * 80)
    print(f"\n总模型数: {len(models)}")
    
    # 分类统计
    issues = {
        '重复路径': [],
        '使用反斜杠': [],
        '路径格式异常': [],
        '文件不存在': [],
        '正常': []
    }
    
    for key, model in models.items():
        path = model.get('model_path', '')
        
        # 检查重复路径
        if 'runs/detect/runs/detect' in path or 'runs\\detect\\runs\\detect' in path:
            issues['重复路径'].append((key, path))
        # 检查反斜杠
        elif '\\' in path:
            issues['使用反斜杠'].append((key, path))
        # 检查路径格式
        elif not path.startswith('runs/detect/') and not path.startswith('yolo_runs/'):
            issues['路径格式异常'].append((key, path))
        else:
            # 检查文件是否存在
            full_path = os.path.join('models', path)
            if not os.path.exists(full_path):
                issues['文件不存在'].append((key, path))
            else:
                issues['正常'].append((key, path))
    
    # 输出结果
    print(f"\n" + "=" * 80)
    print("审查结果:")
    print("=" * 80)
    
    for category, items in issues.items():
        if items:
            print(f"\n【{category}】 ({len(items)}个)")
            for key, path in items[:10]:  # 只显示前10个
                print(f"  - {key}")
                print(f"    路径: {path}")
            if len(items) > 10:
                print(f"  ... 还有 {len(items) - 10} 个")
    
    # 总结
    print(f"\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    total_issues = sum(len(v) for k, v in issues.items() if k != '正常')
    print(f"✓ 正常: {len(issues['正常'])}个")
    print(f"✗ 有问题: {total_issues}个")
    
    if total_issues == 0:
        print(f"\n🎉 所有模型路径都正常！")
    else:
        print(f"\n⚠️ 需要修复 {total_issues} 个模型的路径")

if __name__ == "__main__":
    check_model_paths()
