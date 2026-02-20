import os
import sys

print('开始打包...')

# 基础PyInstaller命令
cmd = [
    'pyinstaller',
    '--name', '溪盟商城自动化助手',
    '--windowed',
    '--onedir',
    '--clean',
    '--noconfirm',
    '--runtime-hook', 'pyi_rth_subprocess.py',
    '--hidden-import', 'multiprocessing',
    '--hidden-import', 'yaml',
    '--hidden-import', 'cv2',
    '--hidden-import', 'PIL',
    '--hidden-import', 'numpy',
    '--hidden-import', 'rapidocr_onnxruntime',
    '--hidden-import', 'cryptography',
    '--hidden-import', 'psutil',
    '--hidden-import', 'tkinter',
    '--hidden-import', 'asyncio',
    '--hidden-import', 'sqlite3',
    '--add-data', 'config;config',
    '--add-data', 'models;models',
    '--add-data', 'config.yaml;.',
    'run.py'
]

import subprocess
result = subprocess.run(cmd)
sys.exit(result.returncode)
