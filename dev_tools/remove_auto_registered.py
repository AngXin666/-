import json

# 读取注册表
with open('models/yolo_model_registry.json', 'r', encoding='utf-8') as f:
    registry = json.load(f)

# 找出所有自动注册的模型
auto_keys = [k for k, v in registry['models'].items() if v.get('auto_registered')]

print(f"发现 {len(auto_keys)} 个自动注册模型:")
for key in auto_keys:
    print(f"  - {key}")

# 删除
for key in auto_keys:
    del registry['models'][key]

# 保存
with open('models/yolo_model_registry.json', 'w', encoding='utf-8') as f:
    json.dump(registry, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已删除 {len(auto_keys)} 个自动注册模型")
print(f"剩余模型数: {len(registry['models'])}")
