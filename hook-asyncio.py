"""
PyInstaller hook for asyncio - 修复打包时的导入问题
"""

from PyInstaller.utils.hooks import collect_submodules

# 收集所有asyncio子模块
hiddenimports = collect_submodules('asyncio')

# 确保关键模块被包含
hiddenimports += [
    'asyncio.base_events',
    'asyncio.events',
    'asyncio.futures',
    'asyncio.protocols',
    'asyncio.transports',
    'asyncio.sslproto',
    'asyncio.locks',
    'asyncio.tasks',
    'asyncio.queues',
    'asyncio.streams',
    'asyncio.subprocess',
    'asyncio.windows_events',
    'asyncio.windows_utils',
    'asyncio.proactor_events',
    'asyncio.selector_events',
]
