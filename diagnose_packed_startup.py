"""诊断打包后程序启动问题的脚本"""
import sys
import os

# 在最开始设置递归深度
sys.setrecursionlimit(15000)

# 创建日志文件
log_file = "diagnose_startup.log"

def log(msg):
    """写入日志"""
    print(msg)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except:
        pass

log("=" * 60)
log("启动诊断开始")
log("=" * 60)

try:
    log(f"Python版本: {sys.version}")
    log(f"递归深度限制: {sys.getrecursionlimit()}")
    log(f"打包环境: {getattr(sys, 'frozen', False)}")
    log(f"工作目录: {os.getcwd()}")
    log(f"可执行文件路径: {sys.executable}")
    
    # 测试基础导入
    log("\n[1/10] 测试基础模块导入...")
    import traceback
    import logging
    log("  ✓ traceback, logging")
    
    log("\n[2/10] 测试subprocess patch...")
    if sys.platform == 'win32':
        import subprocess
        log("  ✓ subprocess")
    
    log("\n[3/10] 测试编码设置...")
    import io
    import locale
    log("  ✓ io, locale")
    
    log("\n[4/10] 测试torch导入...")
    import torch
    log(f"  ✓ torch {torch.__version__}")
    
    log("\n[5/10] 测试PIL导入...")
    from PIL import Image
    log("  ✓ PIL")
    
    log("\n[6/10] 测试yaml导入...")
    import yaml
    log("  ✓ yaml")
    
    log("\n[7/10] 测试src.adb_bridge导入...")
    from src.adb_bridge import ADBBridge
    log("  ✓ src.adb_bridge")
    
    log("\n[8/10] 测试src.page_state_dynamic导入...")
    from src.page_state_dynamic import PageState
    log("  ✓ src.page_state_dynamic")
    
    log("\n[9/10] 测试src.page_detector_integrated导入...")
    from src.page_detector_integrated import PageDetectorIntegrated
    log("  ✓ src.page_detector_integrated")
    
    log("\n[10/10] 测试src.model_manager导入...")
    from src.model_manager import ModelManager
    log("  ✓ src.model_manager")
    
    log("\n" + "=" * 60)
    log("所有导入测试通过！")
    log("=" * 60)
    
    # 测试创建实例
    log("\n测试创建ModelManager实例...")
    manager = ModelManager.get_instance()
    log("  ✓ ModelManager实例创建成功")
    
    log("\n诊断完成，程序正常！")
    
except Exception as e:
    log(f"\n✗ 错误: {e}")
    log("\n完整错误信息:")
    import traceback
    log(traceback.format_exc())
    
finally:
    log("\n诊断结束")
    log("=" * 60)

print(f"\n日志已保存到: {log_file}")
input("按回车键退出...")
