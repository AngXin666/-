"""列出所有YOLO模型"""
import json

with open('yolo_model_registry.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("YOLO模型注册表 - 所有已训练模型")
print("=" * 80)
print()

models = data['models']
print(f"总计: {len(models)} 个模型\n")

for i, (key, value) in enumerate(models.items(), 1):
    name = value.get('name', value.get('model_name', key))
    page_type = value.get('page_type', '未知')
    classes = value.get('classes', [])
    
    print(f"{i:2d}. 模型ID: {key}")
    print(f"    名称: {name}")
    print(f"    页面类型: {page_type}")
    print(f"    检测类别: {', '.join(classes)}")
    print()
