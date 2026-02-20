"""
完整测试：关闭弹窗 → 检测签到按钮 → 点击
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager
from src.navigator import Navigator
from src.daily_checkin import DailyCheckin

async def main():
    print("="*60)
    print("完整测试：签到按钮点击")
    print("="*60)
    
    device_id = "127.0.0.1:16384"
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path=adb_path)
    
    # 初始化模型
    print("\n[1] 初始化模型...")
    model_manager = ModelManager.get_instance()
    model_manager.initialize_all_models(adb)
    detector = model_manager.get_page_detector_integrated()
    print(f"  ✓ 检测器: {type(detector).__name__}")
    
    # 初始化导航器和签到处理器
    print("\n[2] 初始化导航器和签到处理器...")
    navigator = Navigator(adb, detector)
    checkin = DailyCheckin(adb, detector, navigator)
    print("  ✓ 初始化完成")
    
    # 检测当前页面
    print("\n[3] 检测当前页面...")
    page_result = await detector.detect_page(device_id, use_cache=False, detect_elements=True)
    if page_result:
        print(f"  ✓ 当前页面: {page_result.state.value} (置信度: {page_result.confidence:.2%})")
        
        # 如果有弹窗，先关闭
        if page_result.elements:
            print(f"  ✓ 检测到 {len(page_result.elements)} 个元素")
            for element in page_result.elements:
                if "确认" in element.class_name or "关闭" in element.class_name:
                    print(f"  → 点击关闭按钮: {element.center}")
                    await adb.tap(device_id, element.center[0], element.center[1])
                    await asyncio.sleep(1)
                    break
    
    # 导航到首页
    print("\n[4] 导航到首页...")
    success = await navigator.navigate_to_home(device_id)
    if success:
        print("  ✓ 已到达首页")
    else:
        print("  ✗ 无法导航到首页")
        return
    
    # 再次检测页面
    print("\n[5] 确认当前在首页...")
    page_result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
    if page_result:
        print(f"  ✓ 当前页面: {page_result.state.value} (置信度: {page_result.confidence:.2%})")
    
    # 查找签到按钮
    print("\n[6] 查找签到按钮...")
    button_pos = await checkin._find_checkin_button(device_id)
    if button_pos:
        print(f"  ✓ 找到签到按钮: {button_pos}")
        
        # 验证坐标合理性
        x, y = button_pos
        x_min, x_max, y_min, y_max = checkin.CHECKIN_BUTTON_VALID_RANGE
        if x_min <= x <= x_max and y_min <= y <= y_max:
            print(f"  ✓ 坐标合理性验证通过")
        else:
            print(f"  ⚠️ 坐标不合理（超出范围 {checkin.CHECKIN_BUTTON_VALID_RANGE}）")
    else:
        print("  ✗ 未找到签到按钮")
        return
    
    # 点击签到按钮
    print("\n[7] 点击签到按钮...")
    await adb.tap(device_id, button_pos[0], button_pos[1])
    print("  ✓ 已点击")
    
    await asyncio.sleep(2)
    
    # 检测点击后的页面
    print("\n[8] 检测点击后的页面...")
    page_result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
    if page_result:
        print(f"  ✓ 当前页面: {page_result.state.value} (置信度: {page_result.confidence:.2%})")
        
        if "签到" in page_result.state.value or page_result.state.value == "checkin":
            print("\n" + "="*60)
            print("✓✓✓ 测试成功：签到按钮点击正常工作！")
            print("="*60)
        else:
            print("\n" + "="*60)
            print(f"⚠️ 点击后未进入签到页，当前在: {page_result.state.value}")
            print("可能需要调整按钮坐标或检查YOLO模型")
            print("="*60)
    else:
        print("  ✗ 页面检测失败")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
