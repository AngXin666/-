"""
Subprocess wrapper - 自动隐藏所有CMD窗口
这个模块会替换标准的subprocess模块
"""

import sys
import subprocess as _original_subprocess

# 导出所有原始模块的内容
from subprocess import *

# 只在Windows打包环境中修改行为
if sys.platform == 'win32':
    # 创建STARTUPINFO对象
    _STARTUPINFO = _original_subprocess.STARTUPINFO()
    _STARTUPINFO.dwFlags |= _original_subprocess.STARTF_USESHOWWINDOW
    _STARTUPINFO.wShowWindow = _original_subprocess.SW_HIDE
    _CREATE_NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW
    
    # 保存原始函数
    _original_Popen = _original_subprocess.Popen
    _original_run = _original_subprocess.run if hasattr(_original_subprocess, 'run') else None
    _original_call = _original_subprocess.call
    _original_check_call = _original_subprocess.check_call
    _original_check_output = _original_subprocess.check_output
    
    # 包装Popen
    class Popen(_original_Popen):
        def __init__(self, *args, **kwargs):
            if 'startupinfo' not in kwargs:
                kwargs['startupinfo'] = _STARTUPINFO
            if 'creationflags' not in kwargs:
                kwargs['creationflags'] = _CREATE_NO_WINDOW
            else:
                kwargs['creationflags'] |= _CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)
    
    # 包装run
    if _original_run:
        def run(*args, **kwargs):
            if 'startupinfo' not in kwargs:
                kwargs['startupinfo'] = _STARTUPINFO
            if 'creationflags' not in kwargs:
                kwargs['creationflags'] = _CREATE_NO_WINDOW
            else:
                kwargs['creationflags'] |= _CREATE_NO_WINDOW
            return _original_run(*args, **kwargs)
    
    # 包装call
    def call(*args, **kwargs):
        if 'startupinfo' not in kwargs:
            kwargs['startupinfo'] = _STARTUPINFO
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = _CREATE_NO_WINDOW
        else:
            kwargs['creationflags'] |= _CREATE_NO_WINDOW
        return _original_call(*args, **kwargs)
    
    # 包装check_call
    def check_call(*args, **kwargs):
        if 'startupinfo' not in kwargs:
            kwargs['startupinfo'] = _STARTUPINFO
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = _CREATE_NO_WINDOW
        else:
            kwargs['creationflags'] |= _CREATE_NO_WINDOW
        return _original_check_call(*args, **kwargs)
    
    # 包装check_output
    def check_output(*args, **kwargs):
        if 'startupinfo' not in kwargs:
            kwargs['startupinfo'] = _STARTUPINFO
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = _CREATE_NO_WINDOW
        else:
            kwargs['creationflags'] |= _CREATE_NO_WINDOW
        return _original_check_output(*args, **kwargs)
