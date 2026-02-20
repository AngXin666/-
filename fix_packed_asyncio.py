#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复打包后的asyncio模块
解压base_library.zip，修复asyncio/__init__.py，重新打包
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

def fix_asyncio_in_zip(zip_path):
    """修复zip文件中的asyncio/__init__.py"""
    print(f"\n修复 {zip_path} 中的asyncio模块...")
    
    if not os.path.exists(zip_path):
        print(f"  ✗ 找不到文件: {zip_path}")
        return False
    
    # 创建临时目录
    temp_dir = "temp_base_library"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # 1. 解压zip
        print("  [1/4] 解压base_library.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print("  ✓ 解压完成")
        
        # 2. 检查asyncio/__init__.py
        asyncio_init = os.path.join(temp_dir, 'asyncio', '__init__.py')
        if not os.path.exists(asyncio_init):
            print(f"  ✗ 找不到asyncio/__init__.py")
            return False
        
        print("  [2/4] 读取原始asyncio/__init__.py...")
        with open(asyncio_init, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 显示前30行
        lines = original_content.split('\n')
        print(f"  原始文件前30行:")
        for i, line in enumerate(lines[:30], 1):
            print(f"    {i:2d}: {line}")
        
        # 3. 创建修复后的内容
        print("  [3/4] 创建修复后的asyncio/__init__.py...")
        fixed_content = '''"""asyncio - 修复PyInstaller打包问题"""

__all__ = (
    'AbstractEventLoop', 'AbstractEventLoopPolicy',
    'AbstractServer', 'BaseEventLoop', 'BaseProtocol',
    'BaseTransport', 'CancelledError', 'Future', 'InvalidStateError',
    'Protocol', 'StreamReader', 'StreamWriter', 'Task', 'TimeoutError',
    'Transport', 'create_subprocess_exec', 'create_subprocess_shell',
    'create_task', 'current_task', 'ensure_future', 'gather',
    'get_event_loop', 'get_event_loop_policy', 'get_running_loop',
    'iscoroutine', 'iscoroutinefunction', 'new_event_loop',
    'run', 'run_coroutine_threadsafe', 'set_event_loop',
    'set_event_loop_policy', 'sleep', 'wait', 'wait_for',
    'Lock', 'Event', 'Condition', 'Semaphore', 'BoundedSemaphore',
    'Queue', 'PriorityQueue', 'LifoQueue', 'QueueEmpty', 'QueueFull',
)

import sys

# 【关键修复】先导入所有子模块，避免引用未定义的名称
try:
    from . import base_events
    from . import coroutines
    from . import events
    from . import exceptions
    from . import futures
    from . import locks
    from . import protocols
    from . import runners
    from . import queues
    from . import streams
    from . import subprocess
    from . import tasks
    from . import transports
    
    # 然后再从子模块导入所有内容
    from .base_events import *
    from .coroutines import *
    from .events import *
    from .exceptions import *
    from .futures import *
    from .locks import *
    from .protocols import *
    from .runners import *
    from .queues import *
    from .streams import *
    from .subprocess import *
    from .tasks import *
    from .transports import *
    
    # Windows特定
    if sys.platform == 'win32':
        from . import windows_events
        from .windows_events import *
        
        # 设置默认事件循环策略
        set_event_loop_policy(WindowsProactorEventLoopPolicy())
        
except ImportError as e:
    # 如果导入失败，至少提供基本功能
    import warnings
    warnings.warn(f"asyncio import failed: {e}", ImportWarning)
'''
        
        # 写入修复后的内容
        with open(asyncio_init, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print("  ✓ asyncio/__init__.py已修复")
        
        # 4. 重新打包
        print("  [4/4] 重新打包base_library.zip...")
        # 备份原文件
        backup_path = zip_path + '.backup'
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.copy2(zip_path, backup_path)
        print(f"  ✓ 原文件已备份到: {backup_path}")
        
        # 删除原zip
        os.remove(zip_path)
        
        # 创建新zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zip_ref.write(file_path, arcname)
        
        print("  ✓ 重新打包完成")
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print("  ✓ 清理临时文件")
        
        print("\n✓ asyncio修复完成！")
        return True
        
    except Exception as e:
        print(f"\n✗ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return False

def main():
    """主函数"""
    print("="*60)
    print("  修复打包后的asyncio模块")
    print("="*60)
    
    # 检查两个可能的位置
    locations = [
        "D:/溪盟商城自动化助手_打包/溪盟商城自动化助手/_internal/base_library.zip",
        "dist/溪盟商城自动化助手/_internal/base_library.zip",
    ]
    
    zip_path = None
    for loc in locations:
        if os.path.exists(loc):
            zip_path = loc
            print(f"\n找到base_library.zip: {zip_path}")
            break
    
    if not zip_path:
        print("\n✗ 找不到base_library.zip")
        print("请确保已经运行过打包脚本")
        return False
    
    success = fix_asyncio_in_zip(zip_path)
    
    if success:
        print("\n" + "="*60)
        print("  修复完成！现在可以运行程序测试")
        print("="*60)
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
