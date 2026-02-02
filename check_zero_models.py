"""
检查精准度为0的模型是否真的训练过
"""
import json
import os
from pathlib import Path
from ultralytics import YOLO

def check_zero_models():
    """检查精准度为0的模型"""
    print("=" * 60)
    print("检查精准度为0的YOLO模型")
    print("=" * 60)
    
    # 加载注册表
    with open('yolo_model_registry.json', 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # 找出精准度为0的模型
    models_with_zero = [
        (name, info) 
        for name, info in registry['models'].items() 
        if info['performance']['mAP50'] == 0.0
    ]
    
    print(f"\n找到 {len(models_with_zero)} 个精准度为0的模型\n")
    
    for name, info in models_with_zero:
        print(f"\n{'='*60}")
        print(f"模型: {name}")
        print(f"{'='*60}")
        
        model_path = info.get('model_path', '')
        print(f"模型路径: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"  ❌ 模型文件不存在")
            continue
        
        print(f"  ✓ 模型文件存在")
        
        # 尝试加载模型
        try:
            model = YOLO(model_path)
            print(f"  ✓ 模型加载成功")
            print(f"  类别: {model.names}")
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            continue
        
        # 检查原始数据
        original_data_path = info.get('original_data_path', '')
        if original_data_path:
            original_dir = Path(original_data_path)
            if original_dir.exists():
                images_dir = original_dir / "images"
                labels_dir = original_dir / "labels"
                
                if images_dir.exists() and labels_dir.exists():
                    image_count = len(list(images_dir.glob("*.png")))
                    label_count = len(list(labels_dir.glob("*.txt")))
                    print(f"  ✓ 原始数据存在: {image_count} 张图片, {label_count} 个标签")
                    
                    # 随机测试1张图片
                    images = list(images_dir.glob("*.png"))
                    if images:
                        test_img = images[0]
                        print(f"\n  测试图片: {test_img.name}")
                        
                        try:
                            results = model.predict(str(test_img), conf=0.25, verbose=False)
                            if results and len(results) > 0:
                                result = results[0]
                                boxes = result.boxes
                                
                                if len(boxes) > 0:
                                    print(f"    ✓ 检测到 {len(boxes)} 个目标")
                                    for box in boxes:
                                        cls_id = int(box.cls[0].cpu().numpy())
                                        conf = float(box.conf[0].cpu().numpy())
                                        class_name = model.names[cls_id]
                                        print(f"      - {class_name}: 置信度={conf:.2f}")
                                else:
                                    print(f"    ⚠️ 未检测到目标")
                            else:
                                print(f"    ⚠️ 预测失败")
                        except Exception as e:
                            print(f"    ❌ 测试失败: {e}")
                else:
                    print(f"  ❌ 原始数据目录结构不完整")
            else:
                print(f"  ❌ 原始数据目录不存在: {original_data_path}")
        else:
            print(f"  ⚠️ 注册表中没有原始数据路径")
        
        # 检查YOLO数据集
        dataset_name = f"yolo_dataset_{name}"
        dataset_dir = Path(dataset_name)
        if dataset_dir.exists():
            print(f"  ✓ YOLO数据集存在: {dataset_name}")
            
            train_images = list((dataset_dir / "images" / "train").glob("*.png"))
            val_images = list((dataset_dir / "images" / "val").glob("*.png"))
            print(f"    训练集: {len(train_images)} 张")
            print(f"    验证集: {len(val_images)} 张")
        else:
            print(f"  ⚠️ YOLO数据集不存在: {dataset_name}")
            print(f"     (可能训练后已删除)")
    
    print(f"\n{'='*60}")
    print(f"检查完成")
    print(f"{'='*60}")

if __name__ == '__main__':
    check_zero_models()
