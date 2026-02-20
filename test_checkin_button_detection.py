"""
测试签到按钮检测逻辑（假设已在首页）
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
    print("测试：签到按钮检测逻辑")
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
    print("\n[2] 初始化签到处理器...")
    navigator = Navigator(adb, detector)
    checkin = DailyCheckin(adb, detector, navigator)
    print("  ✓ 初始化完成")
    
    # 检测当前页面
    print("\n[3] 检测当前页面...")
    page_result = await detector.detect_page(device_id, use_cache=False, detect_elements=True)
    if page_result:
        print(f"  ✓ 当前页面: {page_result.state.value} (置信度: {page_result.confidence:.2%})")
        if page_result.elements:
            print(f"  ✓ 检测到 {len(page_result.elements)} 个元素:")
            for element in page_result.elements:
                print(f"    - {element.class_name}: {element.center} (置信度: {element.confidence:.2%})")
    
    # 测试按钮检测逻辑
    print("\n[4] 测试 _find_checkin_button 方法...")
    button_pos = await checkin._find_checkin_button(device_id)
    if button_pos:
        print(f"  ✓ 找到签到按钮: {button_pos}")
        
        # 验证坐标合理性
        x, y = button_pos
        x_min, x_max, y_min, y_max = checkin.CHECKIN_BUTTON_VALID_RANGE
        print(f"  ✓ 合理范围: {checkin.CHECKIN_BUTTON_VALID_RANGE}")
        if x_min <= x <= x_max and y_min <= y <= y_max:
            print(f"  ✓ 坐标合理性验证通过")
        else:
            print(f"  ⚠️ 坐标不合理（超出范围）")
    else:
        print("  ✗ 未找到签到按钮")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
