"""
诊断页面检测问题
"""
import asyncio
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager


async def main():
    """主函数"""
    print("=" * 60)
    print("页面检测诊断工具")
    print("=" * 60)
    
    # 1. 初始化ADB
    print("\n[步骤1] 初始化ADB...")
    adb = ADBBridge()
    
    # 2. 连接设备
    print("\n[步骤2] 连接设备...")
    device_id = "127.0.0.1:16384"
    await adb.connect(device_id)
    print(f"✓ 已连接: {device_id}")
    
    # 3. 初始化模型管理器
    print("\n[步骤3] 初始化模型管理器...")
    model_manager = ModelManager.get_instance()
    
    # 4. 初始化所有模型
    print("\n[步骤4] 初始化所有模型...")
    try:
        stats = model_manager.initialize_all_models(
            adb_bridge=adb,
            log_callback=lambda msg: print(f"  [模型加载] {msg}")
        )
        print(f"✓ 模型初始化完成")
        print(f"  - 成功: {stats.get('success', 'N/A')}")
        if 'errors' in stats and stats['errors']:
            print(f"  - 错误:")
            for error in stats['errors']:
                print(f"    * {error}")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 加载页面检测器
    print("\n[步骤5] 获取整合检测器...")
    try:
        detector = model_manager.get_page_detector_integrated()
        if detector:
            print("✓ 整合检测器已加载")
            
            # 调试：检查分类器是否真的加载了
            print(f"\n[调试信息]")
            print(f"  - _classifier_model: {detector._classifier_model is not None}")
            print(f"  - _classes: {detector._classes is not None}")
            if detector._classes:
                print(f"  - 类别数量: {len(detector._classes)}")
            print(f"  - _device: {detector._device}")
            print(f"  - _verbose: {detector._verbose}")
            
            # 启用详细日志
            print(f"\n[启用详细日志]")
            detector.set_verbose(True)
        else:
            print("✗ 整合检测器加载失败")
            return
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 测试页面检测
    print("\n[步骤6] 测试页面检测...")
    print("正在截图并检测...")
    
    try:
        # 检测当前页面
        result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
        
        print("\n" + "=" * 60)
        print("检测结果:")
        print("=" * 60)
        print(f"页面状态: {result.state.value}")
        print(f"中文名称: {result.state.chinese_name}")
        print(f"置信度: {result.confidence:.2%}")
        print(f"检测方法: {result.detection_method}")
        print(f"检测耗时: {result.detection_time:.3f}秒")
        print(f"详细信息: {result.details}")
        
        if result.elements:
            print(f"\n检测到 {len(result.elements)} 个元素:")
            for elem in result.elements:
                print(f"  - {elem.class_name} (置信度: {elem.confidence:.2%})")
        
        print("=" * 60)
        
        # 7. 测试多次检测
        print("\n[步骤7] 连续检测5次，观察稳定性...")
        for i in range(5):
            result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
            print(f"  第{i+1}次: {result.state.value} (置信度: {result.confidence:.2%})")
            await asyncio.sleep(0.5)
        
    except Exception as e:
        print(f"\n✗ 检测失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n诊断完成！")


if __name__ == "__main__":
    asyncio.run(main())
