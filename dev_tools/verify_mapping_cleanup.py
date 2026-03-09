#!/usr/bin/env python3
"""验证映射文件清理结果"""
import json

# 读取映射文件
with open('models/page_yolo_mapping.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)

# 检查签到页映射
checkin = mapping['mapping']['签到页']
print("签到页映射:")
print(f"  模型数量: {len(checkin['yolo_models'])}")
for m in checkin['yolo_models']:
    print(f"  - {m['model_key']} (优先级{m['priority']})")

# 检查首页映射
home = mapping['mapping']['首页']
print("\n首页映射:")
print(f"  模型数量: {len(home['yolo_models'])}")
for m in home['yolo_models']:
    print(f"  - {m['model_key']} (优先级{m['priority']})")

# 统计空映射的页面
empty_pages = [page for page, config in mapping['mapping'].items() if not config.get('yolo_models')]
print(f"\n空映射的页面数: {len(empty_pages)}")
if empty_pages:
    print("空映射的页面:")
    for page in empty_pages[:10]:  # 只显示前10个
        print(f"  - {page}")
    if len(empty_pages) > 10:
        print(f"  ... 还有 {len(empty_pages) - 10} 个")
