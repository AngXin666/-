"""
更新性能指标为0的模型
Update models with zero performance metrics
"""

import json
from pathlib import Path
from ultralytics import YOLO


def test_and_update_model(model_key, model_info, registry_path="yolo_model_registry.json"):
    """测试模型并更新性能指标"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_key}")
    print(f"{'='*60}")
    
    model_path = model_info['model_path']
    
    # 检查模型文件是否存在
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    print(f"✓ 模型文件存在: {model_path}")
    
    # 查找对应的数据集配置文件
    # 从原始数据路径推断数据集路径
    original_data_path = model_info.get('original_data_path', '')
    if not original_data_path:
        print(f"❌ 未找到原始数据路径")
        return False
    
    # 从原始数据路径提取页面类型
    page_type = Path(original_data_path).name.split('_')[0]
    
    # 尝试查找数据集配置文件
    possible_dataset_paths = [
        f"yolo_dataset_{page_type}/dataset.yaml",
        f"yolo_dataset_{page_type.lower()}/dataset.yaml",
        f"yolo_dataset_{model_key}/dataset.yaml",
    ]
    
    dataset_yaml = None
    for path in possible_dataset_paths:
        if Path(path).exists():
            dataset_yaml = path
            break
    
    if not dataset_yaml:
        print(f"❌ 未找到数据集配置文件")
        print(f"   尝试过的路径:")
        for path in possible_dataset_paths:
            print(f"   - {path}")
        return False
    
    print(f"✓ 找到数据集配置: {dataset_yaml}")
    
    # 加载模型
    try:
        print(f"\n加载模型...")
        model = YOLO(model_path)
        print(f"✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 测试模型
    try:
        print(f"\n开始测试模型...")
        results = model.val(data=dataset_yaml, verbose=False)
        
        # 提取性能指标
        performance = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr)
        }
        
        print(f"\n✓ 测试完成:")
        print(f"  mAP50: {performance['mAP50']:.3f}")
        print(f"  mAP50-95: {performance['mAP50-95']:.3f}")
        print(f"  Precision: {performance['precision']:.3f}")
        print(f"  Recall: {performance['recall']:.3f}")
        
        # 更新注册表
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        registry['models'][model_key]['performance'] = performance
        
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 注册表已更新")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("更新性能指标为0的模型")
    print("=" * 60)
    
    # 读取注册表
    registry_path = "yolo_model_registry.json"
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # 查找性能指标为0的模型
    zero_performance_models = []
    for model_key, model_info in registry['models'].items():
        performance = model_info.get('performance', {})
        if all(v == 0.0 for v in performance.values()):
            zero_performance_models.append((model_key, model_info))
    
    if not zero_performance_models:
        print("\n✓ 所有模型的性能指标都已正确记录")
        return
    
    print(f"\n找到 {len(zero_performance_models)} 个性能指标为0的模型:")
    for model_key, _ in zero_performance_models:
        print(f"  - {model_key}")
    
    # 测试并更新每个模型
    success_count = 0
    for model_key, model_info in zero_performance_models:
        if test_and_update_model(model_key, model_info, registry_path):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"更新完成: {success_count}/{len(zero_performance_models)} 个模型成功更新")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
