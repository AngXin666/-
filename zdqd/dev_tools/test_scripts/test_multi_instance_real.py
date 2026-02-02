"""
真实多实例测试 - 模拟实际运行场景
测试 ADB + 截图 + OCR 的完整流程
"""

import asyncio
import time
from src.adb_bridge import ADBBridge
from src.page_detector_hybrid import PageDetectorHybrid

async def test_single_instance(instance_id, device_id, adb_path):
    """测试单个实例的完整流程"""
    print(f"\n[实例{instance_id}] 开始测试...")
    
    try:
        # 创建 ADB 连接
        adb = ADBBridge(adb_path)
        await adb.connect(device_id)
        print(f"[实例{instance_id}] ✓ ADB 连接成功")
        
        # 创建页面检测器
        def log_callback(msg):
            print(f"[实例{instance_id}] {msg}")
        
        detector = PageDetectorHybrid(adb, log_callback=log_callback)
        print(f"[实例{instance_id}] ✓ 检测器创建成功")
        
        # 执行5次检测（模拟实际运行）
        for i in range(5):
            start_time = time.time()
            
            # 检测页面（包含截图 + OCR）
            result = await detector.detect_page(device_id, use_ocr=True, use_template=True)
            
            elapsed = time.time() - start_time
            print(f"[实例{instance_id}] 第{i+1}次检测: {result.state.value}, 耗时: {elapsed:.2f}秒")
            
            # 短暂等待
            await asyncio.sleep(0.5)
        
        print(f"[实例{instance_id}] ✓ 测试完成")
        return True
        
    except Exception as e:
        print(f"[实例{instance_id}] ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_concurrent_instances(num_instances=3):
    """测试多个实例并发运行"""
    print(f"\n{'='*60}")
    print(f"测试 {num_instances} 个实例并发运行")
    print(f"{'='*60}")
    
    # 检测可用的设备
    from src.emulator_controller import EmulatorController
    
    # 自动检测模拟器
    found = EmulatorController.detect_all_emulators()
    if not found:
        print("❌ 未检测到模拟器")
        return
    
    emulator_type, emulator_path = found[0]
    print(f"使用模拟器: {emulator_path}")
    
    controller = EmulatorController(emulator_path)
    adb_path = controller.get_adb_path()
    
    # 获取运行中的实例
    running_instances = await controller.get_running_instances()
    if len(running_instances) < num_instances:
        print(f"❌ 运行中的实例不足，需要 {num_instances} 个，实际 {len(running_instances)} 个")
        print(f"请先启动足够的模拟器实例")
        return
    
    print(f"检测到 {len(running_instances)} 个运行中的实例")
    
    # 为每个实例创建任务
    tasks = []
    for i, instance_id in enumerate(running_instances[:num_instances]):
        adb_port = await controller.get_adb_port(instance_id)
        device_id = f"127.0.0.1:{adb_port}"
        print(f"实例 {i+1}: device_id={device_id}")
        
        task = test_single_instance(i+1, device_id, adb_path)
        tasks.append(task)
    
    # 并发执行
    print(f"\n开始并发执行...")
    start_time = time.time()
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_elapsed = time.time() - start_time
    
    # 统计结果
    print(f"\n{'='*60}")
    print(f"测试结果")
    print(f"{'='*60}")
    print(f"总耗时: {total_elapsed:.2f}秒")
    
    success_count = sum(1 for r in results if r is True)
    print(f"成功: {success_count}/{num_instances}")
    print(f"失败: {num_instances - success_count}/{num_instances}")
    
    # 检查是否有异常
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"实例 {i+1} 异常: {result}")

async def main():
    """主函数"""
    # 测试3个实例并发
    await test_concurrent_instances(num_instances=3)

if __name__ == "__main__":
    asyncio.run(main())
