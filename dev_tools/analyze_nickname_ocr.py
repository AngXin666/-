"""分析昵称OCR识别问题"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
from rapidocr import RapidOCR

# 初始化OCR
ocr = RapidOCR()

# 分析前几张截图（个人页截图）
screenshot_dir = "checkin_screenshots/20260208"

for i in range(1, 6):  # 分析前5张截图
    screenshot_path = f"{screenshot_dir}/{i}.png"
    
    if not os.path.exists(screenshot_path):
        continue
    
    print(f"\n{'='*60}")
    print(f"分析截图: {screenshot_path}")
    print(f"{'='*60}")
    
    # 读取图片
    image = Image.open(screenshot_path)
    
    # 全屏OCR识别
    try:
        result = ocr(image)
        
        if result:
            print(f"\nOCR结果类型: {type(result)}")
            print(f"OCR结果属性: {dir(result)}")
            
            if hasattr(result, 'txts'):
                print(f"\nOCR识别到 {len(result.txts)} 个文本:")
                for j, text in enumerate(result.txts):
                    print(f"  [{j}] '{text}'")
                    
                    # 特别标记可疑的文本
                    if '1' in text or '0' in text:
                        print(f"      ⚠️ 包含数字！")
        else:
            print("OCR识别失败")
    except Exception as e:
        print(f"OCR识别出错: {e}")
        import traceback
        traceback.print_exc()

