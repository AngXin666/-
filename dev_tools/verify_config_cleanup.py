#!/usr/bin/env python3
"""验证config注册表清理结果"""
import json

# 读取config注册表
with open('config/yolo_model_registry.json', 'r', encoding='utf-8') as f:
    config_data = json.load(f)

# 读取models注册表
with open('models/yolo_model_registry.json', 'r', encoding='utf-8') as f:
    models_data = json.load(f)

print("=" * 60)
print("注册表对比")
print("=" * 60)

config_models = config_data['models']
models_models = models_data['models']

print(f"\nconfig/yolo_model_registry.json: {len(config_models)} 个模型")
print(f"models/yolo_model_registry.json: {len(models_models)} 个模型")

# 检查自动注册模型
config_auto = [k for k, v in config_models.items() if v.get('auto_registered')]
models_auto = [k for k, v in models_models.items() if v.get('auto_registered')]

print(f"\nconfig 自动注册模型: {len(config_auto)} 个")
if config_auto:
    for key in config_auto:
        print(f"  - {key}")

print(f"\nmodels 自动注册模型: {len(models_auto)} 个")
if models_auto:
    for key in models_auto:
        print(f"  - {key}")

# 检查差异
config_keys = set(config_models.keys())
models_keys = set(models_models.keys())

only_in_config = config_keys - models_keys
only_in_models = models_keys - config_keys

if only_in_config:
    print(f"\n只在 config 中的模型: {len(only_in_config)} 个")
    for key in only_in_config:
        print(f"  - {key}")

if only_in_models:
    print(f"\n只在 models 中的模型: {len(only_in_models)} 个")
    for key in only_in_models:
        print(f"  - {key}")

if not only_in_config and not only_in_models and len(config_auto) == 0 and len(models_auto) == 0:
    print("\n✅ 两个注册表完全一致，没有自动注册模型")
else:
    print("\n⚠️ 两个注册表存在差异")

print("=" * 60)
