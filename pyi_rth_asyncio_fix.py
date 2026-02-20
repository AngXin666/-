"""
PyInstaller Runtime Hook - 修复asyncio在Windows上的打包问题
Fix asyncio packaging issue on Windows
"""

import sys

# 只在Windows打包环境中执行
if sys.platform == 'win32' and getattr(sys, 'frozen', False):
    try:
        # 【关键修复】在asyncio导入前，先预加载base_events模块
        # 这样asyncio.__init__.py就能找到base_events
        import importlib
        
        # 强制按顺序加载asyncio的所有子模块
        asyncio_modules = [
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
        
        for module_name in asyncio_modules:
            try:
                importlib.import_module(module_name)
            except Exception:
                pass
        
        # 最后导入主asyncio模块
        import asyncio
        
        # 强制设置ProactorEventLoop为默认事件循环
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    except Exception as e:
        # 如果修复失败，至少不要让程序崩溃
        pass
