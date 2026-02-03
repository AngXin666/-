"""直接测试YOLO和OCR - 看它们识别了什么"""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from rapidocr import RapidOCR

print("=" * 80)
print("直接测试YOLO和OCR")
print("=" * 80)

# 1. 加载YOLO模型
print("\n[1] 加载YOLO模型...")
yolo_registry_path = Path("yolo_model_registry.json")
if not yolo_registry_path.exists():
    print("  ✗ YOLO模型注册表不存在")
    sys.exit(1)

import json
with open(yolo_registry_path, 'r', encoding='utf-8') as f:
    registry = json.load(f)

# 查找profile_logged模型
profile_model_info = registry.get('models', {}).get('profile_logged')

if not profile_model_info:
    print("  ✗ 未找到profile_logged模型")
    sys.exit(1)

model_path = Path(profile_model_info['model_path'])
if not model_path.exists():
    print(f"  ✗ 模型文件不存在: {model_path}")
    sys.exit(1)

print(f"  ✓ 加载模型: {model_path}")
yolo_model = YOLO(str(model_path))

# 2. 初始化OCR
print("\n[2] 初始化OCR...")
ocr = RapidOCR()
print("  ✓ OCR初始化完成")

# 3. 查找测试截图
print("\n[3] 查找测试截图...")
screenshot_dir = Path("checkin_screenshots/20260202")
if not screenshot_dir.exists():
    print(f"  ✗ 截图目录不存在: {screenshot_dir}")
    sys.exit(1)

screenshots = sorted(screenshot_dir.glob("*.png"))
print(f"  ✓ 找到 {len(screenshots)} 个截图")

# 4. 测试前5个截图
test_screenshots = screenshots[:5]

print("\n" + "=" * 80)
print("开始测试")
print("=" * 80)

# 创建输出目录
output_dir = Path("debug_yolo_ocr_output")
output_dir.mkdir(exist_ok=True)

for i, screenshot_path in enumerate(test_screenshots, 1):
    print(f"\n{'=' * 80}")
    print(f"截图 {i}: {screenshot_path.name}")
    print(f"{'=' * 80}")
    
    # 读取图片
    image = cv2.imread(str(screenshot_path))
    if image is None:
        print(f"  ✗ 无法读取图片")
        continue
    
    h, w = image.shape[:2]
    print(f"  图片尺寸: {w}x{h}")
    
    # ========== 测试YOLO ==========
    print(f"\n  [YOLO检测]")
    try:
        results = yolo_model(image, conf=0.3, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                print(f"    ✓ 检测到 {len(boxes)} 个区域:")
                
                # 创建可视化图片
                vis_image = image.copy()
                
                for j, box in enumerate(boxes, 1):
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id] if result.names else f"class_{cls_id}"
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    print(f"      区域 {j}: {class_name}")
                    print(f"        置信度: {confidence:.3f}")
                    print(f"        位置: ({x1}, {y1}) -> ({x2}, {y2})")
                    print(f"        尺寸: {x2-x1}x{y2-y1}")
                    
                    # 绘制检测框
                    color = (0, 255, 0)  # 绿色
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加标签
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(vis_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 保存可视化结果
                output_path = output_dir / f"yolo_{screenshot_path.name}"
                cv2.imwrite(str(output_path), vis_image)
                print(f"    可视化已保存: {output_path}")
            else:
                print(f"    ✗ 未检测到任何区域")
        else:
            print(f"    ✗ YOLO返回空结果")
    except Exception as e:
        print(f"    ✗ YOLO检测失败: {e}")
    
    # ========== 测试OCR ==========
    print(f"\n  [OCR识别]")
    try:
        ocr_result = ocr(image)
        
        if ocr_result and ocr_result[0]:
            texts = [item[1] for item in ocr_result[0]]
            boxes = [item[0] for item in ocr_result[0]]
            
            print(f"    ✓ 识别到 {len(texts)} 个文本:")
            print(f"\n    前30个文本（按识别顺序）:")
            
            for j, (text, box) in enumerate(zip(texts[:30], boxes[:30]), 1):
                # 计算文本位置
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                print(f"      {j:2d}. '{text}'")
                print(f"          位置: x={x_min:.0f}-{x_max:.0f}, y={y_min:.0f}-{y_max:.0f}")
            
            # 创建可视化图片
            vis_image = image.copy()
            
            for j, (text, box) in enumerate(zip(texts, boxes), 1):
                # 绘制文本框
                points = np.array(box, dtype=np.int32)
                cv2.polylines(vis_image, [points], True, (0, 0, 255), 2)
                
                # 添加序号
                x_min = int(min([p[0] for p in box]))
                y_min = int(min([p[1] for p in box]))
                cv2.putText(vis_image, str(j), (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # 保存可视化结果
            output_path = output_dir / f"ocr_{screenshot_path.name}"
            cv2.imwrite(str(output_path), vis_image)
            print(f"\n    可视化已保存: {output_path}")
            
            # 分析可疑的文本
            print(f"\n    [分析可疑文本]")
            
            # 查找单字
            single_chars = [t for t in texts if len(t) == 1]
            if single_chars:
                print(f"      单字文本 ({len(single_chars)}个): {single_chars[:10]}")
            
            # 查找6字符大小写混合
            six_char_mixed = []
            for text in texts:
                if len(text) == 6:
                    has_lower = any(c.islower() for c in text)
                    has_upper = any(c.isupper() for c in text)
                    if has_lower and has_upper:
                        six_char_mixed.append(text)
            
            if six_char_mixed:
                print(f"      6字符大小写混合 ({len(six_char_mixed)}个): {six_char_mixed[:5]}")
            
        else:
            print(f"    ✗ OCR未识别到任何文本")
    except Exception as e:
        print(f"    ✗ OCR识别失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
print(f"\n可视化结果保存在: {output_dir}/")
print("  yolo_*.png - YOLO检测结果（绿色框）")
print("  ocr_*.png  - OCR识别结果（红色框+序号）")
