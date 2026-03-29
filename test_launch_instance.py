#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试启动模拟器实例
Test Emulator Instance Launch
"""

import sys
import os
import asyncio
import yaml
from pathlib import Path

# 设置工作目录
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(application_path)
sys.path.insert(0, application_path)

# 导入必要的模块
from src.emulator_controller import EmulatorController
from src.adb_bridge import ADBBridge

async def test_launch_instance():
    """测试启动模拟器实例"""
    
    print("=" * 60)
    print("测试启动模拟器实例")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n[步骤1] 加载配置文件...")
    config_path = 'config.yaml'
    nox_path = None
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            nox_path = config_data.get('nox_path', '')
            print(f"✓ 配置的模拟器路径: {nox_path}")
    else:
        print("✗ 配置文件不存在")
        return False
    
    if not nox_path:
        print("✗ 未配置模拟器路径")
        return False
    
    # 2. 初始化模拟器控制器
    print("\n[步骤2] 初始化模拟器控制器...")
    try:
        emulator_controller = EmulatorController(nox_path)
        adb_path = emulator_controller.get_adb_path()
        
        if adb_path:
            print(f"✓ 找到ADB路径: {adb_path}")
        else:
            print("✗ 未找到ADB路径")
            return False
            
    except Exception as e:
        print(f"✗ 初始化模拟器控制器失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 初始化ADB桥接器
    print("\n[步骤3] 初始化ADB桥接器...")
    try:
        adb = ADBBridge(adb_path)
        print("✓ ADB桥接器初始化成功")
    except Exception as e:
        print(f"✗ 初始化ADB桥接器失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 获取当前运行的实例
    print("\n[步骤4] 检查当前运行的实例...")
    try:
        running_instances = await emulator_controller.get_running_instances()
        print(f"✓ 当前运行的实例: {running_instances}")
    except Exception as e:
        print(f"✗ 获取运行实例失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试启动实例0
    instance_to_launch = 0
    print(f"\n[步骤5] 测试启动实例 {instance_to_launch}...")
    
    if instance_to_launch in running_instances:
        print(f"✓ 实例 {instance_to_launch} 已经在运行")
    else:
        print(f"→ 正在启动实例 {instance_to_launch}...")
        try:
            success = await emulator_controller.launch_instance(instance_to_launch, timeout=120)
            
            if success:
                print(f"✓ 实例 {instance_to_launch} 启动成功")
            else:
                print(f"✗ 实例 {instance_to_launch} 启动失败")
                return False
                
        except Exception as e:
            print(f"✗ 启动实例失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 6. 等待设备启动完成
    print(f"\n[步骤6] 等待实例 {instance_to_launch} 启动完成...")
    try:
        boot_success = await emulator_controller.wait_for_boot(instance_to_launch, timeout=120)
        
        if boot_success:
            print(f"✓ 实例 {instance_to_launch} 启动完成")
        else:
            print(f"✗ 实例 {instance_to_launch} 启动超时")
            return False
            
    except Exception as e:
        print(f"✗ 等待启动失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. 验证ADB连接
    print(f"\n[步骤7] 验证ADB连接...")
    try:
        adb_port = emulator_controller._get_adb_port(instance_to_launch)
        device_id = f"127.0.0.1:{adb_port}"
        
        # 连接设备
        connect_success = await adb.connect(device_id)
        
        if connect_success:
            print(f"✓ 实例 {instance_to_launch} ADB连接成功 ({device_id})")
        else:
            print(f"✗ 实例 {instance_to_launch} ADB连接失败")
            return False
            
    except Exception as e:
        print(f"✗ ADB连接验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 8. 测试完成
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    
    return True

def main():
    """主函数"""
    try:
        # 运行异步测试
        result = asyncio.run(test_launch_instance())
        
        if result:
            print("\n测试成功完成")
            sys.exit(0)
        else:
            print("\n测试失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
