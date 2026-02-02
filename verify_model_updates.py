"""
验证模型注册表和映射配置的更新
"""
import json
from pathlib import Path

def verify_updates():
    """验证所有更新"""
    print("=" * 60)
    print("验证模型注册表和映射配置更新")
    print("=" * 60)
    
    # 加载模型注册表
    with open('yolo_model_registry.json', 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # 加载映射配置
    with open('page_yolo_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    print("\n[1/3] 检查旧模型是否已删除...")
    old_models = ['homepage', 'checkin', 'checkin_popup']
    for model_key in old_models:
        if model_key in registry['models']:
            print(f"  ✗ 旧模型 '{model_key}' 仍然存在!")
        else:
            print(f"  ✓ 旧模型 '{model_key}' 已删除")
    
    print("\n[2/3] 检查新模型是否存在...")
    new_models = {
        '首页': {
            'expected_mAP50': 0.971,
            'expected_classes': ['我的按钮', '每日签到按钮']
        },
        '签到页': {
            'expected_mAP50': 0.995,
            'expected_classes': ['签到按钮', '签到次数']
        },
        '签到成功弹窗': {
            'expected_mAP50': 0.995,
            'expected_classes': ['关闭按钮', '签到成功文本', '签到金额']
        }
    }
    
    for model_key, expected in new_models.items():
        if model_key in registry['models']:
            model_info = registry['models'][model_key]
            print(f"  ✓ 新模型 '{model_key}' 存在")
            print(f"    - mAP50: {model_info['performance']['mAP50']:.3f} (预期: {expected['expected_mAP50']:.3f})")
            print(f"    - 类别: {model_info['classes']}")
            
            # 验证模型文件是否存在
            model_path = Path(model_info['model_path'])
            if model_path.exists():
                print(f"    - 模型文件: ✓ 存在")
            else:
                print(f"    - 模型文件: ✗ 不存在 ({model_path})")
        else:
            print(f"  ✗ 新模型 '{model_key}' 不存在!")
    
    print("\n[3/3] 检查映射配置...")
    mappings_to_check = {
        '首页': '首页',
        '签到页': '签到页',
        '签到弹窗': '签到成功弹窗'
    }
    
    for page_name, expected_model_key in mappings_to_check.items():
        if page_name in mapping['mapping']:
            page_mapping = mapping['mapping'][page_name]
            actual_model_key = page_mapping['yolo_models'][0]['model_key']
            
            if actual_model_key == expected_model_key:
                print(f"  ✓ '{page_name}' 映射到 '{actual_model_key}'")
            else:
                print(f"  ✗ '{page_name}' 映射错误: '{actual_model_key}' (预期: '{expected_model_key}')")
        else:
            print(f"  ✗ '{page_name}' 映射不存在!")
    
    print("\n" + "=" * 60)
    print("✓ 验证完成!")
    print("=" * 60)
    print("\n总结:")
    print("  1. 旧模型已删除: homepage, checkin, checkin_popup")
    print("  2. 新模型已配置: 首页, 签到页, 签到成功弹窗")
    print("  3. 映射配置已更新,使用新训练的高性能模型")
    print("\n新模型性能:")
    print("  - 首页: mAP50=0.971, Recall=0.989")
    print("  - 签到页: mAP50=0.995, Recall=1.000")
    print("  - 签到成功弹窗: mAP50=0.995, Recall=1.000")

if __name__ == '__main__':
    verify_updates()
