"""在运行的模拟器上测试OCR"""
import sys
import os
import asyncio

# 设置UTF-8编码
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src')

import cv2
import numpy as np
from pathlib import Path
from rapidocr import RapidOCR
from adb_bridge import ADBBridge
from PIL import Image
from io import BytesIO

async def test_live_ocr():
    """测试实时OCR"""
    
    print("=" * 80)
    print("测试实时OCR（从模拟器）")
    print("=" * 80)
    
    # 初始化ADB
    print("\n[1] 初始化ADB...")
    adb = ADBBridge()
    
    # 获取设备列表
    devices = await adb.list_devices()
    if not devices:
        print("  ✗ 没有找到运行的模拟器")
        print("  提示: 请先启动模拟器")
        return
    
    print(f"  ✓ 找到 {len(devices)} 个设备:")
    for i, device in enumerate(devices, 1):
        print(f"    {i}. {device}")
    
    # 使用第一个设备
    device_id = devices[0]
    print(f"\n  使用设备: {device_id}")
    
    # 初始化OCR
    print("\n[2] 初始化OCR...")
    ocr = RapidOCR()
    print("  ✓ OCR初始化完成")
    
    # 创建输出目录
    output_dir = Path("debug_live_ocr_output")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("开始测试")
    print("=" * 80)
    print("\n提示: 请确保模拟器显示的是个人资料页面")
    print("按回车开始截图和OCR测试...")
    input()
    
    # 截图
    print("\n[3] 从模拟器截图...")
    screenshot_data = await adb.screencap(device_id)
    if not screenshot_data:
        print("  ✗ 截图失败")
        return
    
    # 转换为图片
    pil_image = Image.open(BytesIO(screenshot_data))
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    print(f"  ✓ 截图成功: {w}x{h}")
    
    # 保存原始截图
    original_path = output_dir / "original.png"
    cv2.imwrite(str(original_path), image)
    print(f"  原始截图已保存: {original_path}")
    
    # OCR识别
    print("\n[4] OCR识别...")
    try:
        ocr_result = ocr(image)
        
        if ocr_result:
            # RapidOCR返回的是RapidOCROutput对象
            if hasattr(ocr_result, 'texts') and ocr_result.texts:
                texts = ocr_result.texts
                boxes = ocr_result.boxes if hasattr(ocr_result, 'boxes') else []
            else:
                print(f"  ✗ OCR未识别到任何文本")
                return
            
            print(f"  ✓ 识别到 {len(texts)} 个文本")
            print(f"\n  所有识别的文本（按顺序）:")
            print("  " + "=" * 76)
            
            for j, text in enumerate(texts, 1):
                # 如果有boxes信息，显示位置
                if boxes and j <= len(boxes):
                    box = boxes[j-1]
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        # 标记可疑文本
                        flag = ""
                        if len(text) == 1:
                            flag = " [单字]"
                        elif len(text) == 6:
                            has_lower = any(c.islower() for c in text)
                            has_upper = any(c.isupper() for c in text)
                            if has_lower and has_upper:
                                flag = " [6字符大小写混合]"
                        
                        print(f"  {j:3d}. '{text}'{flag}")
                        print(f"       位置: x={x_min:3.0f}-{x_max:3.0f}, y={y_min:3.0f}-{y_max:3.0f}")
                    else:
                        print(f"  {j:3d}. '{text}'")
                else:
                    print(f"  {j:3d}. '{text}'")
            
            # 分析可疑的文本
            print("\n" + "=" * 80)
            print("可疑文本分析")
            print("=" * 80)
            
            # 查找单字
            single_chars = [t for t in texts if len(t) == 1]
            if single_chars:
                print(f"\n单字文本 ({len(single_chars)}个):")
                from collections import Counter
                single_char_counter = Counter(single_chars)
                for char, count in single_char_counter.most_common():
                    print(f"  '{char}': {count}次")
            else:
                print(f"\n单字文本: 无")
            
            # 查找6字符大小写混合
            six_char_mixed = []
            for text in texts:
                if len(text) == 6:
                    has_lower = any(c.islower() for c in text)
                    has_upper = any(c.isupper() for c in text)
                    if has_lower and has_upper:
                        six_char_mixed.append(text)
            
            if six_char_mixed:
                print(f"\n6字符大小写混合 ({len(six_char_mixed)}个):")
                for text in six_char_mixed:
                    print(f"  {text}")
            else:
                print(f"\n6字符大小写混合: 无")
            
            # 查找"ID"关键字位置
            print(f"\n查找'ID'关键字:")
            id_found = False
            for j, text in enumerate(texts):
                if 'ID' in text or 'id' in text.lower():
                    print(f"  位置 {j+1}: '{text}'")
                    id_found = True
                    
                    # 显示ID前面的3个文本
                    if j >= 1:
                        print(f"\n  ID前面的文本:")
                        for k in range(max(0, j-3), j):
                            print(f"    {k+1}. '{texts[k]}'")
            
            if not id_found:
                print(f"  未找到ID关键字")
            
            # 创建可视化图片
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
                        thickness = 3
                    elif len(text) == 6:
                        has_lower = any(c.islower() for c in text)
                        has_upper = any(c.isupper() for c in text)
                        if has_lower and has_upper:
                            color = (255, 0, 255)  # 紫色 - 6字符大小写混合
                            thickness = 3
                        else:
                            color = (0, 255, 0)  # 绿色 - 普通文本
                            thickness = 2
                    elif 'ID' in text or 'id' in text.lower():
                        color = (255, 255, 0)  # 青色 - ID
                        thickness = 3
                    else:
                        color = (0, 255, 0)  # 绿色 - 普通文本
                        thickness = 2
                    
                    cv2.polylines(vis_image, [points], True, color, thickness)
                    
                    # 添加序号
                    x_min = int(min([p[0] for p in box]))
                    y_min = int(min([p[1] for p in box]))
                    cv2.putText(vis_image, str(j), (x_min, y_min - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 保存可视化结果
                output_path = output_dir / "ocr_visualization.png"
                cv2.imwrite(str(output_path), vis_image)
                print(f"\n可视化已保存: {output_path}")
                print("\n图例:")
                print("  红色框 = 单字文本")
                print("  紫色框 = 6字符大小写混合")
                print("  青色框 = ID文本")
                print("  绿色框 = 普通文本")
            
        else:
            print(f"  ✗ OCR未识别到任何文本")
    except Exception as e:
        print(f"  ✗ OCR识别失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_live_ocr())
