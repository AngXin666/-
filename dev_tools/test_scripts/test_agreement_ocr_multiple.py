"""
多次测试OCR识别并计算勾选框坐标
验证坐标计算的稳定性和准确性
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.screen_capture import ScreenCapture
from src.ocr_thread_pool import get_ocr_pool
from PIL import Image
import cv2
import win32api
import win32con
import time

async def test_ocr_calculate_coord(screen, ocr_pool, device_id, test_num):
    """单次测试：OCR识别并计算勾选框坐标"""
    print(f"\n{'='*60}")
    print(f"【第 {test_num} 次识别】")
    print(f"{'='*60}")
    
    try:
        # 1. 截取屏幕
        print("截取屏幕...")
        img = await screen.capture(device_id)
        if img is None:
            print("❌ 截图失败")
            return None
        
        # 2. 转换为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 3. OCR识别
        print("OCR识别文字...")
        ocr_result = await ocr_pool.recognize(pil_img, timeout=5.0)
        
        if not ocr_result.texts:
            print("❌ OCR未识别到文字")
            return None
        
        print(f"✓ 识别到 {len(ocr_result.texts)} 个文字块")
        
        # 4. 查找协议文字
        keywords = ["我已阅读", "用户协议", "隐私政策", "已阅读"]
        
        for text, box in zip(ocr_result.texts, ocr_result.boxes):
            for keyword in keywords:
                if keyword in text:
                    # 计算勾选框位置
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    
                    text_x = int(min(x_coords))
                    text_y = int(min(y_coords))
                    text_h = int(max(y_coords) - min(y_coords))
                    
                    checkbox_x = text_x - 15
                    checkbox_y = text_y + text_h // 2
                    
                    print(f"✓ 找到文字: '{text}'")
                    print(f"  文字位置: ({text_x}, {text_y})")
                    print(f"  计算勾选框位置: ({checkbox_x}, {checkbox_y})")
                    
                    return (checkbox_x, checkbox_y)
        
        print("❌ 未找到协议文字")
        return None
        
    except Exception as e:
        print(f"❌ 识别异常: {e}")
        return None

async def main():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    screen = ScreenCapture(adb)
    ocr_pool = get_ocr_pool()
    device_id = '127.0.0.1:5555'
    
    print("="*60)
    print("多次测试OCR识别并验证勾选框坐标")
    print("="*60)
    print("\n测试说明：")
    print("1. 程序会多次OCR识别协议文字")
    print("2. 每次计算勾选框坐标并点击")
    print("3. 你用鼠标确认是否勾选成功")
    print("   - 已勾选：按鼠标左键")
    print("   - 未勾选：按鼠标右键")
    print("   - 跳过：按 ESC 键")
    print("\n请确保当前在登录页面\n")
    
    input("按回车开始测试...")
    
    # 测试5次
    test_count = 5
    coords_list = []
    success_count = 0
    expected_state = False  # 初始状态：未勾选
    
    for i in range(test_count):
        # OCR识别并计算坐标
        coord = await test_ocr_calculate_coord(screen, ocr_pool, device_id, i + 1)
        
        if coord is None:
            print("⏭️  本次识别失败，跳过")
            continue
        
        coords_list.append(coord)
        
        # 点击勾选框
        print(f"\n点击坐标: {coord}")
        print(f"期望状态切换: {'未勾选 → 已勾选' if not expected_state else '已勾选 → 未勾选'}")
        await adb.tap(device_id, coord[0], coord[1])
        await asyncio.sleep(0.3)
        
        print("\n请观察屏幕并确认：")
        print(f"  当前应该是: {'已勾选' if not expected_state else '未勾选'}")
        print("  - 已勾选？按鼠标左键")
        print("  - 未勾选？按鼠标右键")
        print("  - 跳过？按 ESC 键")
        
        # 等待用户确认
        actual_state = None
        while True:
            if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & 0x8000:
                actual_state = True  # 已勾选
                time.sleep(0.5)
                break
            
            if win32api.GetAsyncKeyState(win32con.VK_RBUTTON) & 0x8000:
                actual_state = False  # 未勾选
                time.sleep(0.5)
                break
            
            if win32api.GetAsyncKeyState(win32con.VK_ESCAPE) & 0x8000:
                actual_state = None  # 跳过
                time.sleep(0.5)
                break
            
            await asyncio.sleep(0.05)
        
        # 判断结果
        if actual_state is None:
            print("⏭️  跳过")
        elif actual_state == (not expected_state):
            print(f"✅ 成功！状态正确切换")
            success_count += 1
            expected_state = actual_state
        else:
            print(f"❌ 失败！状态没有切换")
        
        # 等待1秒再进行下一次测试
        if i < test_count - 1:
            print("\n等待1秒后进行下一次识别...")
            await asyncio.sleep(1)
    
    # 统计结果
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    
    if coords_list:
        print(f"\n识别到的坐标（共 {len(coords_list)} 次）：")
        for i, coord in enumerate(coords_list, 1):
            print(f"  {i}. {coord}")
        
        # 计算平均坐标
        if len(coords_list) > 1:
            avg_x = sum(x for x, y in coords_list) // len(coords_list)
            avg_y = sum(y for x, y in coords_list) // len(coords_list)
            print(f"\n平均坐标: ({avg_x}, {avg_y})")
            
            # 计算坐标偏差
            max_x_diff = max(abs(x - avg_x) for x, y in coords_list)
            max_y_diff = max(abs(y - avg_y) for y, y in coords_list)
            print(f"最大偏差: X={max_x_diff}px, Y={max_y_diff}px")
    
    print(f"\n总测试次数: {len(coords_list)}")
    print(f"成功次数: {success_count}")
    print(f"失败次数: {len(coords_list) - success_count}")
    
    if len(coords_list) > 0:
        print(f"成功率: {success_count / len(coords_list) * 100:.1f}%")
        
        if success_count == len(coords_list):
            print("\n✅ 所有测试都成功！OCR定位非常稳定可靠")
        elif success_count >= len(coords_list) * 0.8:
            print("\n⚠️  大部分测试成功，OCR定位基本可靠")
        else:
            print("\n❌ 成功率较低，需要优化坐标计算方法")

asyncio.run(main())
