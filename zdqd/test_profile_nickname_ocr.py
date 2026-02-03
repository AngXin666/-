"""测试个人资料页昵称识别 - 使用实际截图"""
import sys
import os
import asyncio

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src')

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# 导入profile_reader
from profile_reader import ProfileReader
from model_manager import ModelManager
from ocr_thread_pool import OCRThreadPool

async def test_nickname_recognition():
    """测试昵称识别"""
    
    print("=" * 80)
    print("测试个人资料页昵称识别")
    print("=" * 80)
    
    # 初始化组件
    print("\n初始化组件...")
    model_manager = ModelManager()
    ocr_pool = OCRThreadPool(max_workers=2)
    profile_reader = ProfileReader(model_manager, ocr_pool)
    
    # 查找最近的签到截图（虽然不是个人资料页，但可以测试OCR）
    screenshot_dir = Path("checkin_screenshots/20260202")
    if not screenshot_dir.exists():
        print(f"截图目录不存在: {screenshot_dir}")
        return
    
    screenshots = sorted(screenshot_dir.glob("*.png"))
    print(f"找到 {len(screenshots)} 个截图")
    
    # 测试前3个截图
    test_screenshots = screenshots[:3]
    
    print("\n" + "=" * 80)
    print("开始测试...")
    print("=" * 80)
    
    for i, screenshot_path in enumerate(test_screenshots, 1):
        print(f"\n{'=' * 80}")
        print(f"截图 {i}: {screenshot_path.name}")
        print(f"{'=' * 80}")
        
        # 读取图片
        image = cv2.imread(str(screenshot_path))
        if image is None:
            print(f"  ✗ 无法读取图片")
            continue
        
        print(f"  图片尺寸: {image.shape[1]}x{image.shape[0]}")
        
        try:
            # 测试YOLO检测
            print("\n  [1] 测试YOLO检测...")
            profile_model = model_manager.get_model('profile_logged')
            results = profile_model(image, conf=0.3, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"      ✓ YOLO检测到 {len(result.boxes)} 个区域")
                    for j, box in enumerate(result.boxes, 1):
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id] if result.names else f"class_{cls_id}"
                        confidence = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        print(f"        区域 {j}: {class_name} (置信度: {confidence:.3f})")
                        print(f"                位置: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
                else:
                    print(f"      ✗ YOLO未检测到任何区域")
            else:
                print(f"      ✗ YOLO未检测到任何区域")
            
            # 测试全屏OCR
            print("\n  [2] 测试全屏OCR...")
            ocr_result = await ocr_pool.recognize(image, timeout=5.0)
            
            if ocr_result and ocr_result.texts:
                print(f"      ✓ OCR识别到 {len(ocr_result.texts)} 个文本")
                print(f"      前20个文本:")
                for j, text in enumerate(ocr_result.texts[:20], 1):
                    # 获取文本位置
                    if ocr_result.boxes and j <= len(ocr_result.boxes):
                        box = ocr_result.boxes[j-1]
                        x_min = min(box[0][0], box[1][0], box[2][0], box[3][0])
                        y_min = min(box[0][1], box[1][1], box[2][1], box[3][1])
                        print(f"        {j:2d}. '{text}' (位置: x={x_min:.0f}, y={y_min:.0f})")
                    else:
                        print(f"        {j:2d}. '{text}'")
                
                # 测试昵称提取
                print("\n  [3] 测试昵称提取...")
                nickname = profile_reader._extract_nickname(ocr_result.texts)
                if nickname:
                    print(f"      ✓ 提取到昵称: '{nickname}'")
                else:
                    print(f"      ✗ 未能提取昵称")
            else:
                print(f"      ✗ OCR未识别到任何文本")
            
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 关闭OCR线程池
    ocr_pool.shutdown()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_nickname_recognition())
