"""
验证注册表更新结果
"""
import json

# 加载注册表
with open('yolo_model_registry.json', 'r', encoding='utf-8') as f:
    registry = json.load(f)

# 检查精准度为0的模型
models_with_zero = [
    name 
    for name, info in registry['models'].items() 
    if info['performance']['mAP50'] == 0.0
]

print("=" * 60)
print("验证注册表更新结果")
print("=" * 60)
print(f"\n精准度为0的模型数量: {len(models_with_zero)}")

if models_with_zero:
    print(f"仍有精准度为0的模型: {', '.join(models_with_zero)}")
else:
    print("✓ 所有模型都有性能指标")

# 显示已更新的模型
updated_models = ['分类页', '搜索页', '积分页', '文章页', '钱包页', '个人页广告', '首页异常代码弹窗']
print(f"\n已更新的模型:")
for name in updated_models:
    if name in registry['models']:
        perf = registry['models'][name]['performance']
        print(f"  - {name}:")
        print(f"      mAP50: {perf['mAP50']:.3f}")
        print(f"      Precision: {perf['precision']:.3f}")
        print(f"      Recall: {perf['recall']:.3f}")

print(f"\n" + "=" * 60)
