"""
快速性能测试脚本
Quick Performance Test Script

用法:
  python test_performance_quick.py [device_id] [package_name]

示例:
  python test_performance_quick.py 127.0.0.1:16384 com.ry.xmsc
"""

import asyncio
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tests.performance.test_startup_navigation_performance import PerformanceTestRunner


async def main():
    """主函数"""
    # 默认配置
    device_id = "127.0.0.1:16384"  # MuMu模拟器默认端口
    package_name = "com.ry.xmsc"
    
    # 从命令行参数获取配置
    if len(sys.argv) > 1:
        device_id = sys.argv[1]
    
    if len(sys.argv) > 2:
        package_name = sys.argv[2]
    
    print("")
    print("=" * 80)
    print("启动和导航性能快速测试")
    print("=" * 80)
    print("")
    print(f"配置:")
    print(f"  设备ID: {device_id}")
    print(f"  应用包名: {package_name}")
    print("")
    print("提示:")
    print("  - 请确保模拟器已启动")
    print("  - 请确保应用已安装")
    print("  - 测试将自动运行所有性能测试")
    print("")
    input("按 Enter 键开始测试...")
    print("")
    
    # 创建测试运行器
    runner = PerformanceTestRunner(device_id)
    
    # 连接设备
    runner.log("正在连接设备...")
    connected = await runner.adb.connect(device_id)
    
    if not connected:
        runner.log(f"✗ 无法连接到设备: {device_id}")
        runner.log("")
        runner.log("故障排查:")
        runner.log("  1. 检查模拟器是否已启动")
        runner.log("  2. 检查ADB服务是否正在运行")
        runner.log("  3. 检查设备ID是否正确")
        runner.log("  4. 尝试手动连接: adb connect " + device_id)
        runner.log("")
        return 1
    
    runner.log(f"✓ 已连接到设备: {device_id}")
    runner.log("")
    
    # 运行所有测试
    try:
        await runner.run_all_tests(package_name)
        runner.print_summary()
        
        runner.log("")
        runner.log("=" * 80)
        runner.log("测试完成！")
        runner.log("=" * 80)
        runner.log("")
        
        return 0
        
    except KeyboardInterrupt:
        runner.log("")
        runner.log("✗ 用户中断测试")
        runner.log("")
        return 1
        
    except Exception as e:
        runner.log("")
        runner.log(f"✗ 测试过程中发生错误: {e}")
        runner.log("")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
