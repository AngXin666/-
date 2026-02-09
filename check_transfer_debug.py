"""
查看转账调试日志
"""

import os
import re
from datetime import datetime

def check_transfer_debug_logs():
    """检查转账调试日志"""
    
    print("=" * 80)
    print("转账调试日志分析")
    print("=" * 80)
    print()
    
    # 检查日志文件
    log_files = [
        'logs/transfer_20260209.log',
        'logs/transfer_failure_20260209.log',
        'logs/NoxAutomation_20260209.log',
        'logs/debug_20260209.log'
    ]
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            continue
        
        print(f"\n{'=' * 80}")
        print(f"文件: {log_file}")
        print('=' * 80)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("  (空文件)")
            continue
        
        # 查找转账相关的调试信息
        debug_lines = []
        for i, line in enumerate(lines):
            # 查找调试标记
            if '[调试]' in line or '[转账]' in line or 'SmartWaiter' in line:
                debug_lines.append((i+1, line.strip()))
        
        if debug_lines:
            print(f"\n找到 {len(debug_lines)} 条相关日志:\n")
            for line_num, line in debug_lines:
                print(f"  [{line_num:4d}] {line}")
        else:
            print("\n  未找到调试信息")
    
    # 检查截图目录
    print(f"\n{'=' * 80}")
    print("检查调试截图")
    print('=' * 80)
    
    screenshot_dir = 'screenshots/transfer_debug'
    if os.path.exists(screenshot_dir):
        screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
        if screenshots:
            print(f"\n找到 {len(screenshots)} 个调试截图:\n")
            for screenshot in sorted(screenshots):
                file_path = os.path.join(screenshot_dir, screenshot)
                file_size = os.path.getsize(file_path)
                print(f"  - {screenshot} ({file_size:,} bytes)")
            
            # 打开截图目录
            print(f"\n正在打开截图目录: {screenshot_dir}")
            os.startfile(os.path.abspath(screenshot_dir))
        else:
            print("\n  目录存在但没有截图")
    else:
        print("\n  截图目录不存在")
    
    print(f"\n{'=' * 80}")
    print("分析完成")
    print('=' * 80)

if __name__ == '__main__':
    check_transfer_debug_logs()
    input("\n按回车键退出...")
