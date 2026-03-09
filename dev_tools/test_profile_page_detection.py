"""
测试个人页检测 - 验证登录专用模型能否识别个人未登录页
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager


async def main():
    """主函数"""
    print("=" * 60)
    print("测试个人页检测 - 登录专用模型")
    print("=" * 60)
    
    # 初始化模拟器控制器和ADB
    from src.emulator_controller import EmulatorController
    controller = EmulatorController()
    adb_path = controller.get_adb_path()
    
    if not adb_path:
        print("❌ 无法找到 ADB 路径")
        return
    
    print(f"✓ ADB 路径: {adb_path}")
    
    adb = ADBBridge(adb_path=adb_path)
    
    # 使用设备ID
    device_id = "127.0.0.1:16448"
    print(f"✓ 使用设备: {device_id}\n")
    
    # 获取登录专用检测器
    model_manager = ModelManager.get_instance()
    
    # 加载所有模型
    print("正在加载模型...")
    stats = model_manager.initialize_all_models(
        adb_bridge=adb,
        log_callback=print
    )
    print(f"\n✓ 模型加载完成")
    print(f"  已加载模型: {stats['models_loaded']}")
    print(f"  加载时间: {stats['total_time']:.2f}秒\n")
    
    # 获取登录专用检测器
    login_detector = model_manager.get_login_detector()
    print(f"登录专用检测器: {login_detector}")
    
    if not login_detector:
        print("❌ 登录专用模型未加载")
        return
    
    print("✓ 登录专用模型已加载\n")
    
    # 截图并保存
    print("正在截图...")
    screenshot_data = await adb.screencap(device_id)
    
    if screenshot_data:
        # 保存截图
        screenshot_path = project_root / "test_profile_screenshot.png"
        with open(screenshot_path, 'wb') as f:
            f.write(screenshot_data)
        print(f"✓ 截图已保存: {screenshot_path}\n")
    else:
        print("❌ 截图失败")
        return
    
    # 使用登录专用模型检测
    print("使用登录专用模型检测页面...")
    result = await login_detector.detect_page(device_id, use_cache=False)
    
    print("\n" + "=" * 60)
    print("登录专用模型检测结果:")
    print("=" * 60)
    print(f"页面状态: {result.state.value}")
    print(f"置信度: {result.confidence:.2%}")
    print(f"详细信息: {result.details}")
    print(f"检测方法: {result.detection_method}")
    print(f"检测耗时: {result.detection_time*1000:.0f}ms")
    print("=" * 60)
    
    # 使用通用分类器检测
    print("\n使用通用分类器检测页面...")
    general_classifier = model_manager.get_general_classifier()
    
    if general_classifier:
        # 获取YOLO识别器
        detector = model_manager.get_page_detector_integrated()
        if detector:
            general_result = await detector.detect_page(device_id, use_cache=False, use_ocr=False, use_dl=True)
            
            print("\n" + "=" * 60)
            print("通用分类器检测结果:")
            print("=" * 60)
            print(f"页面状态: {general_result.state.value}")
            if hasattr(general_result, 'confidence'):
                print(f"置信度: {general_result.confidence:.2%}")
            if hasattr(general_result, 'details'):
                print(f"详细信息: {general_result.details}")
            if hasattr(general_result, 'detection_method'):
                print(f"检测方法: {general_result.detection_method}")
            if hasattr(general_result, 'detection_time'):
                print(f"检测耗时: {general_result.detection_time*1000:.0f}ms")
            print("=" * 60)
            
            # 对比结果
            print("\n" + "=" * 60)
            print("结果对比:")
            print("=" * 60)
            if result.state == general_result.state:
                print(f"✓ 两个模型识别结果一致: {result.state.value}")
            else:
                print(f"❌ 两个模型识别结果不一致:")
                print(f"  登录专用模型: {result.state.value}")
                print(f"  通用分类器: {general_result.state.value}")
            print("=" * 60)
    else:
        print("⚠️ 通用分类器未加载")
    
    # 判断结果
    if result.state.value == "profile":
        print("\n✓ 登录专用模型识别为个人页未登录")
    elif result.state.value == "profile_logged":
        print("\n⚠️ 登录专用模型识别为个人页已登录")
    elif result.state.value == "home":
        print("\n❌ 登录专用模型识别为首页")
    else:
        print(f"\n⚠️ 登录专用模型识别为其他页面: {result.state.value}")


if __name__ == "__main__":
    asyncio.run(main())
