"""只测试OCR - 看它识别出了什么"""
import sys
import os

# 设置UTF-8编码
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from pathlib import Path
from rapidocr import RapidOCR

print("=" * 80)
print("测试OCR识别")
print("=" * 80)

# 初始化OCR
print("\n[1] 初始化OCR...")
ocr = RapidOCR()
print("  ✓ OCR初始化完成")

# 查找测试截图
print("\n[2] 查找测试截图...")
screenshot_dir = Path("checkin_screenshots/20260202")
if not screenshot_dir.exists():
    print(f"  ✗ 截图目录不存在: {screenshot_dir}")
    sys.exit(1)

screenshots = sorted(screenshot_dir.glob("*.png"))
print(f"  ✓ 找到 {len(screenshots)} 个截图")

# 测试前10个截图
test_screenshots = screenshots[:10]

print("\n" + "=" * 80)
print("开始测试")
print("=" * 80)

# 创建输出目录
output_dir = Path("debug_ocr_output")
output_dir.mkdir(exist_ok=True)

# 统计数据
all_single_chars = []
all_six_char_mixed = []

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
    
    # OCR识别
    print(f"\n  [OCR识别]")
    try:
        ocr_result = ocr(image)
        
        if ocr_result:
            # RapidOCR返回的是RapidOCROutput对象，有texts和boxes属性
            if hasattr(ocr_result, 'texts') and ocr_result.texts:
                texts = ocr_result.texts
                boxes = ocr_result.boxes if hasattr(ocr_result, 'boxes') else []
            elif isinstance(ocr_result, (list, tuple)) and len(ocr_result) > 0:
                # 旧版API
                texts = [item[1] for item in ocr_result[0]]
                boxes = [item[0] for item in ocr_result[0]]
            else:
                print(f"    ✗ OCR未识别到任何文本")
                continue
            
            print(f"    ✓ 识别到 {len(texts)} 个文本")
            print(f"\n    前30个文本（按识别顺序）:")
            
            for j, text in enumerate(texts[:30], 1):
                # 如果有boxes信息，显示位置
                if boxes and j <= len(boxes):
                    box = boxes[j-1]
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        print(f"      {j:2d}. '{text}'")
                        print(f"          位置: x={x_min:.0f}-{x_max:.0f}, y={y_min:.0f}-{y_max:.0f}")
                    else:
                        print(f"      {j:2d}. '{text}'")
                else:
                    print(f"      {j:2d}. '{text}'")
            
            # 分析可疑的文本
            print(f"\n    [分析可疑文本]")
            
            # 查找单字
            single_chars = [t for t in texts if len(t) == 1]
            if single_chars:
                print(f"      单字文本 ({len(single_chars)}个): {single_chars}")
                all_single_chars.extend(single_chars)
            
            # 查找6字符大小写混合
            six_char_mixed = []
            for text in texts:
                if len(text) == 6:
                    has_lower = any(c.islower() for c in text)
                    has_upper = any(c.isupper() for c in text)
                    if has_lower and has_upper:
                        six_char_mixed.append(text)
            
            if six_char_mixed:
                print(f"      6字符大小写混合 ({len(six_char_mixed)}个): {six_char_mixed}")
                all_six_char_mixed.extend(six_char_mixed)
            
            # 创建可视化图片（如果有boxes信息）
            if boxes:
                vis_image = image.copy()
                
                for j, (text, box) in enumerate(zip(texts, boxes), 1):
                    if not isinstance(box, (list, tuple)) or len(box) < 4:
                        continue
                    
                    # 绘制文本框
                    points = np.array(box, dtype=np.int32)
                    
                    # 根据文本类型使用不同颜色
                    if len(text) == 1:
                        color = (0, 0, 255)  # 红色 - 单字
                    elif len(text) == 6:
                        has_lower = any(c.islower() for c in text)
                        has_upper = any(c.isupper() for c in text)
                        if has_lower and has_upper:
                            color = (255, 0, 255)  # 紫色 - 6字符大小写混合
                        else:
                            color = (0, 255, 0)  # 绿色 - 普通文本
                    else:
                        color = (0, 255, 0)  # 绿色 - 普通文本
                    
                    cv2.polylines(vis_image, [points], True, color, 2)
                    
                    # 添加序号
                    x_min = int(min([p[0] for p in box]))
                    y_min = int(min([p[1] for p in box]))
                    cv2.putText(vis_image, str(j), (x_min, y_min - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 保存可视化结果
                output_path = output_dir / f"ocr_{screenshot_path.name}"
                cv2.imwrite(str(output_path), vis_image)
                print(f"\n    可视化已保存: {output_path}")
            
        else:
            print(f"    ✗ OCR未识别到任何文本")
    except Exception as e:
        print(f"    ✗ OCR识别失败: {e}")
        import traceback
        traceback.print_exc()

# 统计总结
print("\n" + "=" * 80)
print("统计总结")
print("=" * 80)

if all_single_chars:
    from collections import Counter
    single_char_counter = Counter(all_single_chars)
    print(f"\n单字文本统计（共{len(all_single_chars)}个）:")
    for char, count in single_char_counter.most_common(10):
        print(f"  '{char}': {count}次")

if all_six_char_mixed:
    print(f"\n6字符大小写混合统计（共{len(all_six_char_mixed)}个）:")
    for text in all_six_char_mixed[:20]:
        print(f"  {text}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
print(f"\n可视化结果保存在: {output_dir}/")
print("  红色框 = 单字文本")
print("  紫色框 = 6字符大小写混合")
print("  绿色框 = 普通文本")
