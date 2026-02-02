"""
测试OCR定位协议勾选框
直接在当前登录页面测试
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.screen_capture import ScreenCapture
from src.ocr_thread_pool import get_ocr_pool
import cv2
import numpy as np

async def test():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    screen = ScreenCapture(adb)
    ocr_pool = get_ocr_pool()
    
    print("="*60)
    print("OCR定位协议勾选框测试")
    print("="*60)
    print("\n请确保当前在登录页面\n")
    
    # 截取屏幕
    print("【步骤1】截取屏幕...")
    img = await screen.capture('127.0.0.1:5555')
    if img is None:
        print("❌ 截图失败")
        return
    print("✓ 截图成功")
    
    # 转换为PIL图像
    from PIL import Image
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # OCR识别
    print("\n【步骤2】OCR识别文字...")
    ocr_result = await ocr_pool.recognize(pil_img, timeout=10.0)
    
    if not ocr_result.texts:
        print("❌ OCR未识别到任何文字")
        return
    
    print(f"✓ 识别到 {len(ocr_result.texts)} 个文字块")
    
    # 查找协议相关文字
    print("\n【步骤3】查找协议文字...")
    keywords = ["我已阅读", "用户协议", "隐私政策", "已阅读", "接受"]
    
    found = False
    for i, (text, box) in enumerate(zip(ocr_result.texts, ocr_result.boxes)):
        # 检查是否包含关键词
        for keyword in keywords:
            if keyword in text:
                print(f"\n✓ 找到文字: '{text}'")
                print(f"  包含关键词: '{keyword}'")
                
                # box 是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                
                text_x = int(min(x_coords))
                text_y = int(min(y_coords))
                text_w = int(max(x_coords) - min(x_coords))
                text_h = int(max(y_coords) - min(y_coords))
                
                print(f"  文字位置: ({text_x}, {text_y}), 大小: ({text_w}x{text_h})")
                
                # 计算勾选框位置（在文字左侧约15像素）
                checkbox_x = text_x - 15
                checkbox_y = text_y + text_h // 2
                
                print(f"  → 推算勾选框位置: ({checkbox_x}, {checkbox_y})")
                
                # 计算文字点击位置
                text_click_x = text_x + text_w // 2
                text_click_y = text_y + text_h // 2
                
                print(f"  → 推算文字点击位置: ({text_click_x}, {text_click_y})")
                
                found = True
                
                # 测试点击文字区域
                print(f"\n【步骤4】测试点击文字区域...")
                print(f"点击坐标: ({text_click_x}, {text_click_y})")
                await adb.tap('127.0.0.1:5555', text_click_x, text_click_y)
                print("✓ 已点击，请观察屏幕是否勾选")
                
                await asyncio.sleep(2)
                
                # 再测试点击勾选框
                print(f"\n【步骤5】测试点击勾选框...")
                print(f"点击坐标: ({checkbox_x}, {checkbox_y})")
                await adb.tap('127.0.0.1:5555', checkbox_x, checkbox_y)
                print("✓ 已点击，请观察屏幕是否勾选")
                
                break
        
        if found:
            break
    
    if not found:
        print("\n❌ 未找到协议文字")
        print("\n识别到的所有文字：")
        for i, text in enumerate(ocr_result.texts[:20]):  # 只显示前20个
            print(f"  {i+1}. {text}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

asyncio.run(test())
