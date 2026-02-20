"""测试打包后程序的启动问题"""
import sys
import os

# 设置递归深度
sys.setrecursionlimit(10000)

print(f"Python版本: {sys.version}")
print(f"递归深度限制: {sys.getrecursionlimit()}")
print(f"打包环境: {getattr(sys, 'frozen', False)}")
print(f"工作目录: {os.getcwd()}")

# 测试导入
print("\n测试导入...")

try:
    print("1. 导入torch...")
    import torch
    print(f"   ✓ torch版本: {torch.__version__}")
except Exception as e:
    print(f"   ✗ torch导入失败: {e}")

try:
    print("2. 导入PIL...")
    from PIL import Image
    print("   ✓ PIL导入成功")
except Exception as e:
    print(f"   ✗ PIL导入失败: {e}")

try:
    print("3. 导入rapidocr...")
    from rapidocr_onnxruntime import RapidOCR
    print("   ✓ rapidocr导入成功")
except Exception as e:
    print(f"   ✗ rapidocr导入失败: {e}")

try:
    print("4. 导入src.adb_bridge...")
    from src.adb_bridge import ADBBridge
    print("   ✓ adb_bridge导入成功")
except Exception as e:
    print(f"   ✗ adb_bridge导入失败: {e}")
    import traceback
    traceback.print_exc()

try:
    print("5. 导入src.page_detector_integrated...")
    from src.page_detector_integrated import PageDetectorIntegrated
    print("   ✓ page_detector_integrated导入成功")
except Exception as e:
    print(f"   ✗ page_detector_integrated导入失败: {e}")
    import traceback
    traceback.print_exc()

try:
    print("6. 导入src.model_manager...")
    from src.model_manager import ModelManager
    print("   ✓ model_manager导入成功")
except Exception as e:
    print(f"   ✗ model_manager导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n所有导入测试完成")
input("按回车键退出...")
