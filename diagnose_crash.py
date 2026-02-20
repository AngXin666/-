"""
诊断打包后程序崩溃的原因
"""
import subprocess
import sys
import time
import os

print("=" * 80)
print("打包后程序崩溃诊断")
print("=" * 80)

exe_path = r"D:\溪盟商城自动化助手_打包\溪盟商城自动化助手.exe"
log_path = r"D:\溪盟商城自动化助手_打包\startup_error.log"

# 清空旧日志
if os.path.exists(log_path):
    os.remove(log_path)
    print(f"已清空旧日志: {log_path}")

print(f"\n启动程序: {exe_path}")
print("监控10秒...")

try:
    # 启动程序
    process = subprocess.Popen(
        exe_path,
        cwd=r"D:\溪盟商城自动化助手_打包"
    )
    
    # 监控10秒
    for i in range(10):
        time.sleep(1)
        
        # 检查进程状态
        if process.poll() is not None:
            print(f"\n❌ 程序在第{i+1}秒时退出")
            print(f"退出码: {process.returncode}")
            print(f"退出码(十六进制): 0x{process.returncode & 0xFFFFFFFF:08X}")
            
            # 解释退出码
            if process.returncode == -1073741819 or (process.returncode & 0xFFFFFFFF) == 0xC0000005:
                print("\n退出码含义: ACCESS_VIOLATION (访问违规)")
                print("可能原因:")
                print("  1. DLL加载失败")
                print("  2. 多线程访问冲突")
                print("  3. 内存访问错误")
                print("  4. 深度学习库初始化失败")
            
            break
        
        print(f"  {i+1}秒: 程序运行中...")
    else:
        print("\n✅ 程序运行正常（10秒内未退出）")
        print("手动检查GUI是否显示...")
        
        # 等待用户确认
        input("\n按Enter键终止程序...")
        process.terminate()
        process.wait(timeout=5)
    
    # 读取日志
    print("\n" + "=" * 80)
    print("启动日志:")
    print("=" * 80)
    
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
            print(log_content)
    else:
        print("❌ 未找到日志文件")
    
except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
