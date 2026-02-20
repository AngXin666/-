"""
测试YOLO模型加载
"""
import json
from pathlib import Path
from ultralytics import YOLO

print("="*60)
print("测试YOLO模型加载")
print("="*60)

# 读取模型注册表
registry_path = Path("config/yolo_model_registry.json")
with open(registry_path, 'r', encoding='utf-8') as f:
    registry = json.load(f)

# 测试首页模型
print("\n[1] 测试首页模型...")
home_model_info = registry['models']['首页']
model_path = home_model_info['model_path']
print(f"  模型路径: {model_path}")

# 检查文件是否存在
full_path = Path("models") / model_path
print(f"  完整路径: {full_path}")
print(f"  文件存在: {full_path.exists()}")

if full_path.exists():
    try:
        print("\n[2] 加载YOLO模型...")
        model = YOLO(str(full_path))
        print(f"  ✓ 模型加载成功")
        print(f"  ✓ 类别: {model.names}")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n  ✗ 模型文件不存在")
    print(f"\n  尝试查找模型文件...")
    
    # 搜索模型文件
    models_dir = Path("models")
    found_files = list(models_dir.rglob("*首页*detector2*/weights/best.pt"))
    
    if found_files:
        print(f"  找到 {len(found_files)} 个可能的模型文件:")
        for f in found_files:
            rel_path = f.relative_to(models_dir)
            print(f"    - {rel_path}")
            
            # 尝试加载第一个找到的模型
            if len(found_files) == 1:
                print(f"\n[3] 尝试加载找到的模型...")
                try:
                    model = YOLO(str(f))
                    print(f"  ✓ 模型加载成功")
                    print(f"  ✓ 类别: {model.names}")
                    print(f"\n  建议更新配置文件中的路径为: {rel_path}")
                except Exception as e:
                    print(f"  ✗ 模型加载失败: {e}")
    else:
        print(f"  ✗ 未找到任何首页模型文件")

print("\n" + "="*60)
