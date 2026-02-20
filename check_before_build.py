#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打包前检查脚本 - 确保所有必需文件都存在
Pre-build Check Script
"""

import os
import sys

def check_file(path, description):
    """检查文件是否存在"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists

def check_directory(path, description):
    """检查目录是否存在"""
    exists = os.path.isdir(path)
    if exists:
        file_count = sum(1 for _, _, files in os.walk(path) for _ in files)
        print(f"  ✓ {description}: {path} ({file_count} 个文件)")
    else:
        print(f"  ✗ {description}: {path}")
    return exists

def main():
    """主函数"""
    print("="*60)
    print("  打包前检查")
    print("="*60)
    
    all_ok = True
    
    # 1. 检查主程序入口
    print("\n[1] 检查主程序入口")
    all_ok &= check_file('run.py', '主程序入口')
    
    # 2. 检查runtime hooks
    print("\n[2] 检查runtime hooks")
    all_ok &= check_file('pyi_rth_hide_console.py', '隐藏控制台hook')
    all_ok &= check_file('pyi_rth_subprocess.py', 'subprocess补丁hook')
    all_ok &= check_file('pyi_rth_asyncio_fix.py', 'asyncio修复hook')
    
    # 3. 检查PyInstaller hooks
    print("\n[3] 检查PyInstaller hooks")
    all_ok &= check_file('hook-rapidocr.py', 'RapidOCR hook')
    all_ok &= check_file('hook-src.user_management_gui.py', '用户管理GUI hook')
    
    # 4. 检查asyncio补丁
    print("\n[4] 检查asyncio补丁")
    all_ok &= check_file('asyncio_init_patch.py', 'asyncio补丁文件')
    
    # 5. 检查配置文件
    print("\n[5] 检查配置文件")
    all_ok &= check_file('config.yaml', '主配置文件')
    all_ok &= check_file('model_config.json.example', '模型配置示例')
    all_ok &= check_file('transfer_config.json.example', '转账配置示例')
    all_ok &= check_file('.env.example', '环境变量示例')
    
    # 6. 检查数据目录
    print("\n[6] 检查数据目录")
    all_ok &= check_directory('config', '配置目录')
    all_ok &= check_directory('models', '模型目录')
    all_ok &= check_directory('docs', '文档目录')
    all_ok &= check_directory('src', '源代码目录')
    
    # 7. 检查关键模块
    print("\n[7] 检查关键模块")
    key_modules = [
        'src/gui.py',
        'src/adb_bridge.py',
        'src/emulator_controller.py',
        'src/auto_login.py',
        'src/daily_checkin.py',
        'src/ximeng_automation.py',
        'src/model_manager.py',
        'src/user_management_gui.py',
        'src/license_manager_simple.py',
    ]
    
    for module in key_modules:
        all_ok &= check_file(module, os.path.basename(module))
    
    # 8. 检查子模块目录
    print("\n[8] 检查子模块目录")
    all_ok &= check_directory('src/ad_detection', '广告检测模块')
    all_ok &= check_directory('src/performance', '性能优化模块')
    
    # 9. 检查打包脚本
    print("\n[9] 检查打包脚本")
    all_ok &= check_file('build_exe.py', '打包脚本')
    
    # 10. 检查Python环境
    print("\n[10] 检查Python环境")
    print(f"  Python版本: {sys.version}")
    print(f"  Python路径: {sys.executable}")
    
    # 检查PyInstaller
    try:
        import PyInstaller
        print(f"  ✓ PyInstaller版本: {PyInstaller.__version__}")
    except ImportError:
        print(f"  ✗ PyInstaller未安装")
        all_ok = False
    
    # 总结
    print("\n" + "="*60)
    if all_ok:
        print("  ✓ 所有检查通过，可以开始打包！")
        print("\n  执行打包命令:")
        print("    python build_exe.py")
    else:
        print("  ✗ 检查发现问题，请先修复后再打包")
    print("="*60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
