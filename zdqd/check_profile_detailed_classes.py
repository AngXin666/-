"""
检查 profile_detailed 模型的实际类别名称
"""
from ultralytics import YOLO

model_path = "runs/detect/runs/detect/profile_detailed_detector/weights/best.pt"

print("=" * 60)
print("检查 profile_detailed 模型的类别名称")
print("=" * 60)

try:
    model = YOLO(model_path)
    print(f"\n模型路径: {model_path}")
    print(f"类别数量: {len(model.names)}")
    print(f"\n实际类别名称:")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")
    
    print("\n" + "=" * 60)
    print("注册表中记录的类别名称:")
    print("=" * 60)
    registry_classes = [
        "首页按钮",
        "我的按钮",
        "抵扣券数字",
        "优惠券数字",
        "余额数字",
        "积分数字",
        "昵称文字",
        "ID文字"
    ]
    for idx, name in enumerate(registry_classes):
        print(f"  {idx}: {name}")
    
    print("\n" + "=" * 60)
    print("对比结果:")
    print("=" * 60)
    
    all_match = True
    for idx in range(len(registry_classes)):
        actual = model.names.get(idx, "N/A")
        expected = registry_classes[idx]
        match = "✓" if actual == expected else "✗"
        if actual != expected:
            all_match = False
        print(f"  {match} 类别{idx}: 实际='{actual}' vs 预期='{expected}'")
    
    if all_match:
        print("\n✓ 所有类别名称匹配！")
    else:
        print("\n❌ 类别名称不匹配！需要重新训练模型或更新注册表。")
        
except Exception as e:
    print(f"\n❌ 加载模型失败: {e}")
    import traceback
    traceback.print_exc()
