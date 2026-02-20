"""
测试打包后的程序启动
"""
import subprocess
import sys
import time

print("=" * 60)
print("测试打包后的程序启动")
print("=" * 60)

exe_path = r"D:\溪盟商城自动化助手_打包\溪盟商城自动化助手.exe"

print(f"\n启动程序: {exe_path}")
print("等待10秒...")

try:
    # 使用subprocess启动程序，捕获输出
    process = subprocess.Popen(
        exe_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=r"D:\溪盟商城自动化助手_打包"
    )
    
    # 等待10秒
    time.sleep(10)
    
    # 检查进程状态
    if process.poll() is None:
        print("\n✅ 程序正在运行")
        print("手动检查GUI是否显示...")
        
        # 终止进程
        process.terminate()
        process.wait(timeout=5)
        print("程序已终止")
    else:
        print(f"\n❌ 程序已退出，退出码: {process.returncode}")
        
        # 读取输出
        stdout, stderr = process.communicate()
        
        if stdout:
            print("\n标准输出:")
            print(stdout)
        
        if stderr:
            print("\n标准错误:")
            print(stderr)
    
except Exception as e:
    print(f"\n❌ 启动失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
