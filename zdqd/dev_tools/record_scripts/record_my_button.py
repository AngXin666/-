"""
记录"我的"按钮坐标工具
用于在首页截图上标记"我的"按钮的准确位置
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.screen_capture import ScreenCapture
from src.ocr_thread_pool import get_ocr_pool


async def find_my_button():
    """使用 OCR 查找"我的"按钮位置"""
    
    from src.emulator_controller import EmulatorController
    
    print("=" * 60)
    print("记录'我的'按钮坐标工具")
    print("=" * 60)
    
    # 自动检测模拟器
    print(f"\n1. 自动检测模拟器...")
    found_emulators = EmulatorController.detect_all_emulators()
    
    if not found_emulators:
        print(f"❌ 未检测到模拟器")
        print(f"   请确保 MuMu 模拟器正在运行")
        return
    
    emulator_type, emulator_path = found_emulators[0]
    print(f"✓ 检测到模拟器: {emulator_path}")
    
    # 初始化控制器
    controller = EmulatorController(emulator_path)
    adb_path = controller.get_adb_path()
    
    if not adb_path:
        print(f"❌ 未找到 ADB 路径")
        return
    
    print(f"✓ ADB 路径: {adb_path}")
    
    # 获取 ADB 端口
    adb_port = await controller.get_adb_port(0)
    device_id = f"127.0.0.1:{adb_port}"
    print(f"✓ 设备 ID: {device_id}")
    
    # 初始化
    adb = ADBBridge(adb_path)
    screen_capture = ScreenCapture(adb)
    ocr_pool = get_ocr_pool()
    
    # 连接设备
    print(f"\n2. 连接设备...")
    connected = await adb.connect(device_id)
    if not connected:
        print(f"❌ 连接失败")
        return
    print(f"✓ 连接成功")
    
    # 截图
    print(f"\n3. 截取当前屏幕")
    screenshot_np = await screen_capture.capture(device_id)
    if screenshot_np is None:
        print(f"❌ 截图失败")
        return
    
    # 转换为 PIL Image
    from PIL import Image
    screenshot = Image.fromarray(screenshot_np)
    print(f"✓ 截图成功: {screenshot.width}x{screenshot.height}")
    
    # 保存截图
    screenshot_path = "my_button_screenshot.png"
    screenshot.save(screenshot_path)
    print(f"✓ 截图已保存: {screenshot_path}")
    
    # OCR 识别
    print(f"\n4. 使用 OCR 识别'我的'按钮")
    print(f"   搜索关键词: ['我的', '个人中心', '我']")
    
    # 识别整个屏幕
    ocr_result = await ocr_pool.recognize(screenshot, timeout=10.0)
    
    if not ocr_result.texts:
        print(f"\n   ❌ OCR 未识别到任何文本")
        return
    
    print(f"\n   识别到 {len(ocr_result.texts)} 个文本区域:")
    print(f"   {'序号':<6} {'文本':<15} {'位置(x,y)':<20} {'置信度':<10}")
    print(f"   {'-'*60}")
    
    my_button_candidates = []
    
    for i, (text, box, confidence) in enumerate(zip(ocr_result.texts, ocr_result.boxes, ocr_result.scores), 1):
        # 计算中心点
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        center_x = int(sum(x_coords) / len(x_coords))
        center_y = int(sum(y_coords) / len(y_coords))
        
        # 显示所有识别结果
        print(f"   {i:<6} {text:<15} ({center_x}, {center_y}){' '*5} {confidence:.2f}")
        
        # 查找"我的"相关文本
        if any(keyword in text for keyword in ['我的', '我', '个人', '中心']):
            my_button_candidates.append({
                'text': text,
                'center': (center_x, center_y),
                'box': box,
                'confidence': confidence
            })
    
    # 显示候选结果
    if my_button_candidates:
        print(f"\n5. 找到 {len(my_button_candidates)} 个'我的'按钮候选位置:")
        print(f"   {'序号':<6} {'文本':<15} {'中心坐标':<20} {'置信度':<10}")
        print(f"   {'-'*60}")
        
        for i, candidate in enumerate(my_button_candidates, 1):
            text = candidate['text']
            center = candidate['center']
            confidence = candidate['confidence']
            print(f"   {i:<6} {text:<15} {center}{' '*10} {confidence:.2f}")
        
        # 推荐最佳候选（底部导航栏区域 y > 850）
        bottom_candidates = [c for c in my_button_candidates if c['center'][1] > 850]
        
        if bottom_candidates:
            print(f"\n6. 推荐坐标（底部导航栏区域）:")
            best = max(bottom_candidates, key=lambda c: c['confidence'])
            print(f"   文本: {best['text']}")
            print(f"   坐标: {best['center']}")
            print(f"   置信度: {best['confidence']:.2f}")
            print(f"\n   建议在 navigator.py 中更新:")
            print(f"   TAB_MY = {best['center']}  # '{best['text']}' 按钮")
        else:
            print(f"\n6. ⚠️ 未在底部导航栏区域找到'我的'按钮")
            print(f"   所有候选位置的 y 坐标都小于 850")
    else:
        print(f"\n5. ❌ 未找到'我的'按钮")
        print(f"   请检查:")
        print(f"   - 是否在首页")
        print(f"   - 底部导航栏是否可见")
        print(f"   - OCR 识别是否准确")
    
    print(f"\n" + "=" * 60)
    print(f"完成！请查看 {screenshot_path} 确认位置")
    print(f"=" * 60)


if __name__ == "__main__":
    asyncio.run(find_my_button())
