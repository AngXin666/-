"""
测试YOLO模型加载和检测
Test YOLO Model Loading and Detection
"""

import sys
from pathlib import Path
import os
import json

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("❌ PIL未安装")
    sys.exit(1)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("❌ ultralytics未安装")
    sys.exit(1)


def test_yolo_models():
    """测试YOLO模型"""
    print("=" * 70)
    print("测试YOLO模型加载和检测")
    print("=" * 70)
    
    # 加载YOLO模型注册表
    print("\n[1] 加载YOLO模型注册表...")
    registry_path = 'yolo_model_registry.json'
    
    if not os.path.exists(registry_path):
        alt_path = os.path.join('models', registry_path)
        if os.path.exists(alt_path):
            registry_path = alt_path
        else:
            print(f"❌ 注册表文件不存在: {registry_path}")
            return
    
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    models_info = registry.get('models', {})
    print(f"✓ 注册表加载成功，共 {len(models_info)} 个模型")
    
    # 检查目标模型
    print("\n[2] 检查个人页相关的YOLO模型...")
    target_models = ['profile_logged', 'balance']
    
    for model_key in target_models:
        print(f"\n检查模型: {model_key}")
        print("-" * 70)
        
        if model_key not in models_info:
            print(f"  ❌ 模型未在注册表中: {model_key}")
            continue
        
        model_info = models_info[model_key]
        model_path = model_info.get('model_path')
        classes = model_info.get('classes', [])
        
        print(f"  模型名称: {model_info.get('name')}")
        print(f"  模型路径: {model_path}")
        print(f"  检测类别: {classes}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            # 尝试在models目录查找
            alt_path = os.path.join('models', model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"  ✓ 在models目录找到: {model_path}")
            else:
                print(f"  ❌ 模型文件不存在: {model_path}")
                continue
        else:
            print(f"  ✓ 模型文件存在")
        
        # 尝试加载模型
        try:
            print(f"  正在加载模型...")
            model = YOLO(model_path)
            print(f"  ✓ 模型加载成功")
            
            # 检查模型类别
            model_classes = list(model.names.values())
            print(f"  模型类别数量: {len(model_classes)}")
            print(f"  模型类别: {model_classes}")
            
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            continue
    
    # 使用测试图片进行检测
    print("\n[3] 使用测试图片进行YOLO检测...")
    print("=" * 70)
    
    # 查找测试图片
    test_image_path = None
    test_dirs = [
        '原始标注图/个人页_已登录_余额积分/images',
        '原始标注图/个人页_已登录_头像首页/images',
    ]
    
    for img_dir in test_dirs:
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image_path = os.path.join(img_dir, images[0])
                break
    
    if not test_image_path:
        print("❌ 未找到测试图片")
        return
    
    print(f"测试图片: {test_image_path}")
    
    # 加载图片
    image = Image.open(test_image_path)
    print(f"图片尺寸: {image.size}")
    
    # 测试每个模型
    for model_key in target_models:
        print(f"\n测试模型: {model_key}")
        print("-" * 70)
        
        if model_key not in models_info:
            continue
        
        model_info = models_info[model_key]
        model_path = model_info.get('model_path')
        
        # 检查路径
        if not os.path.exists(model_path):
            alt_path = os.path.join('models', model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                print(f"  ❌ 模型文件不存在")
                continue
        
        try:
            # 加载模型
            model = YOLO(model_path)
            
            # 执行检测
            print(f"  执行检测（置信度阈值=0.25）...")
            results = model.predict(image, conf=0.25, verbose=False)
            
            # 解析结果
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                print(f"  检测到 {len(boxes)} 个目标:")
                
                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        # 提取信息
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        
                        print(f"    {i+1}. {class_name}")
                        print(f"       - 置信度: {conf:.2%}")
                        print(f"       - 边界框: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
                    
                    print(f"  ✅ YOLO检测成功")
                else:
                    print(f"  ⚠️ 未检测到任何目标（可能置信度过低）")
            else:
                print(f"  ⚠️ 检测结果为空")
        
        except Exception as e:
            print(f"  ❌ 检测失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == '__main__':
    test_yolo_models()
