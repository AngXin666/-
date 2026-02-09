"""查找个人页截图并分析OCR"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
from rapidocr import RapidOCR

# 初始化OCR
ocr = RapidOCR()

screenshot_dir = "checkin_screenshots/20260208"

# 遍历所有截图，找到个人页（包含"余额"、"积分"等关键字）
print("正在扫描截图，查找个人页...")

profile_screenshots = []

for i in range(1, 166):
    screenshot_path = f"{screenshot_dir}/{i}.png"
    
    if not os.path.exists(screenshot_path):
        continue
    
    # 读取图片
    image = Image.open(screenshot_path)
    
    # OCR识别
    try:
        result = ocr(image)
        
        if result and hasattr(result, 'txts') and result.txts:
            texts = ' '.join(result.txts)
            
            # 判断是否是个人页（包含余额、积分等关键字）
            if '余额' in texts and '积分' in texts:
                profile_screenshots.append((i, screenshot_path, result.txts))
                print(f"  找到个人页截图: {i}.png")
                
                # 只找前10张
                if len(profile_screenshots) >= 10:
                    break
    except Exception as e:
        pass

print(f"\n共找到 {len(profile_screenshots)} 张个人页截图\n")

# 分析这些个人页截图
for i, (num, path, texts) in enumerate(profile_screenshots[:5]):  # 只分析前5张
    print(f"\n{'='*60}")
    print(f"分析截图 {num}.png")
    print(f"{'='*60}")
    
    print(f"\nOCR识别到的所有文本:")
    for j, text in enumerate(texts):
        print(f"  [{j}] '{text}'")
        
        # 标记可疑文本
        text_clean = text.replace(' ', '').replace('\t', '')
        if text_clean.isdigit() and len(text_clean) <= 3:
            print(f"      ⚠️ 可疑的数字组合！")
        elif text in ['1 0', '10', '1', '0']:
            print(f"      ⚠️⚠️⚠️ 这就是问题文本！")
