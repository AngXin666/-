#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Testing import...")

try:
    import subprocess
    print("subprocess imported successfully")
    
    # 测试运行 PyInstaller
    result = subprocess.run(['pyinstaller', '--version'], capture_output=True, text=True)
    print("PyInstaller version:", result.stdout.strip())
    
    print("\nAll tests passed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
