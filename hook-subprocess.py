"""
PyInstaller hook for subprocess module
在打包时修改subprocess模块，注入隐藏CMD窗口的代码
"""

from PyInstaller.utils.hooks import collect_submodules, get_module_file_attribute
import os

# 收集subprocess的所有子模块
hiddenimports = collect_submodules('subprocess')

def pre_safe_import_module(api):
    """在导入subprocess之前修改其源码"""
    pass

def post_safe_import_module(api):
    """在导入subprocess之后修改其行为"""
    # 这个hook会在分析阶段运行，我们需要在运行时修改
    pass
