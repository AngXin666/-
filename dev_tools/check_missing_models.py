"""检查缺失的YOLO模型文件"""
import json
import os

# 读取注册表
with open('models/yolo_model_registry.json', 'r', encoding='utf-8') as f:
    registry = json.load(f)

print("检查YOLO模型文件...")
print("=" * 60)

missing_models = []
existing_models = []

for model_key, model_info in registry['models'].items():
    model_path = model_info.get('model_path')
    if not model_path:
        continue
    
    # 添加models/前缀
    full_path = os.path.join('models', model_path)
    
    if os.path.exists(full_path):
        existing_models.append((model_key, full_path))
        print(f"✓ {model_key}")
    else:
        missing_models.append((model_key, full_path))
        print(f"✗ {model_key} -> {full_path}")

print("\n" + "=" * 60)
print(f"总计: {len(registry['models'])} 个模型")
print(f"存在: {len(existing_models)} 个")
print(f"缺失: {len(missing_models)} 个")

if missing_models:
    print("\n缺失的模型:")
    for model_key, path in missing_models:
        print(f"  - {model_key}: {path}")
