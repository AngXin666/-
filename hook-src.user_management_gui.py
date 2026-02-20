"""
PyInstaller hook for src.user_management_gui
强制包含用户管理GUI模块及其所有依赖
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 收集所有依赖的子模块
hiddenimports = [
    'src.user_management_gui',
    'src.user_manager',
    'src.local_db',
    'src.transfer_config',
    'src.config',
    'src.encrypted_accounts_file',
    'src.login_cache_manager',
    'src.account_cache',
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.scrolledtext',
]

# 收集数据文件（如果有）
datas = []
