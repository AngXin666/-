"""从Git历史中恢复缺失的YOLO模型文件"""
import json
import os
import subprocess

# 读取注册表
with open('models/yolo_model_registry.json', 'r', encoding='utf-8') as f:
    registry = json.load(f)

print("从Git历史中恢复缺失的YOLO模型文件...")
print("=" * 60)

missing_models = []
restored_models = []
failed_models = []

for model_key, model_info in registry['models'].items():
    model_path = model_info.get('model_path')
    if not model_path:
        continue
    
    # 添加models/前缀
    full_path = os.path.join('models', model_path)
    
    if not os.path.exists(full_path):
        missing_models.append((model_key, full_path))

print(f"发现 {len(missing_models)} 个缺失的模型文件")
print()

# 从Git历史中恢复
for model_key, full_path in missing_models:
    print(f"恢复 {model_key}...")
    try:
        # 使用git checkout从历史中恢复文件
        result = subprocess.run(
            ['git', 'checkout', '49fe07e', '--', full_path],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            # 验证文件是否恢复成功
            if os.path.exists(full_path):
                restored_models.append((model_key, full_path))
                print(f"  ✓ 恢复成功")
            else:
                failed_models.append((model_key, full_path, "文件未创建"))
                print(f"  ✗ 恢复失败：文件未创建")
        else:
            failed_models.append((model_key, full_path, result.stderr))
            print(f"  ✗ 恢复失败：{result.stderr}")
    except Exception as e:
        failed_models.append((model_key, full_path, str(e)))
        print(f"  ✗ 恢复失败：{e}")

print("\n" + "=" * 60)
print(f"恢复完成:")
print(f"  成功: {len(restored_models)} 个")
print(f"  失败: {len(failed_models)} 个")

if failed_models:
    print("\n失败的模型:")
    for model_key, path, error in failed_models:
        print(f"  - {model_key}: {error}")
