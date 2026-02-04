"""
测试 avatar_homepage YOLO模型
测试原始标注图，检查模型是否能正确识别首页按钮和优惠券按钮
"""

import os
import sys
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

def test_avatar_homepage_model():
    """测试 avatar_homepage 模型"""
    
    # 切换到zdqd目录
    os.chdir("zdqd")
    
    # 模型路径
    model_path = "models/runs/detect/runs/detect/yolo_runs/avatar_homepage_detector/train/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    print(f"✓ 加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 原始标注图目录
    test_dirs = [
        "yolo_dataset/images/val",  # 验证集
        "yolo_dataset/images/train",  # 训练集（测试前10张）
    ]
    
    # 查找测试图片
    test_images = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            test_images.extend(images[:10])  # 每个目录最多10张
    
    if not test_images:
        print(f"❌ 未找到测试图片")
        return
    
    print(f"\n找到 {len(test_images)} 张测试图片")
    
    # 测试图片
    total_images = 0
    detected_home = 0
    detected_avatar = 0
    detected_other = 0
    
    print(f"\n{'='*60}")
    print(f"开始测试")
    print(f"{'='*60}")
    
    for i, img_path in enumerate(test_images, 1):
        try:
            # 加载图片
            image = Image.open(img_path)
            
            # YOLO检测（降低置信度阈值）
            results = model.predict(image, conf=0.25, verbose=False)
            
            total_images += 1
            
            # 分析检测结果
            detected_classes = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    conf = float(box.conf[0])
                    detected_classes.append((class_name, conf))
                    
                    if "首页" in class_name:
                        detected_home += 1
                    elif "头像" in class_name:
                        detected_avatar += 1
                    else:
                        detected_other += 1
            
            # 输出结果
            img_name = os.path.basename(img_path)
            if detected_classes:
                print(f"  [{i}] {img_name}")
                for class_name, conf in detected_classes:
                    print(f"      - {class_name}: {conf:.2%}")
            else:
                print(f"  [{i}] {img_name} - 未检测到任何对象")
                
        except Exception as e:
            print(f"  [{i}] {os.path.basename(img_path)} - 错误: {e}")
    
    # 统计结果
    print(f"\n{'='*60}")
    print(f"测试统计")
    print(f"{'='*60}")
    print(f"总图片数: {total_images}")
    print(f"检测到首页按钮: {detected_home} 次")
    print(f"检测到头像: {detected_avatar} 次")
    print(f"检测到其他对象: {detected_other} 次")
    
    # 分析结果
    if detected_home == 0:
        print(f"\n⚠️ 警告: 模型没有检测到任何首页按钮")
        print(f"   可能的原因:")
        print(f"   1. 置信度阈值太高（当前0.25）")
        print(f"   2. 模型训练数据不足")
        print(f"   3. 测试图片与训练数据差异较大")
        print(f"   4. 测试数据集不是avatar_homepage模型的训练数据")
    else:
        print(f"\n✓ 模型能够检测到首页按钮")
        print(f"  检测率: {detected_home}/{total_images} = {detected_home/total_images*100:.1f}%")

if __name__ == "__main__":
    test_avatar_homepage_model()
