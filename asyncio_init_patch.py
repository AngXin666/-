"""
asyncio __init__.py 补丁 - 修复PyInstaller打包问题
"""

__all__ = (
    'AbstractEventLoop', 'AbstractEventLoopPolicy',
    'AbstractServer', 'BaseEventLoop', 'BaseProtocol',
    'BaseTransport', 'CancelledError', 'Future', 'InvalidStateError',
    'Protocol', 'StreamReader', 'StreamWriter', 'Task', 'TimeoutError',
    'Transport', 'create_subprocess_exec', 'create_subprocess_shell',
    'create_task', 'current_task', 'ensure_future', 'gather',
    'get_event_loop', 'get_event_loop_policy', 'get_running_loop',
    'iscoroutine', 'iscoroutinefunction', 'new_event_loop',
    'run', 'run_coroutine_threadsafe', 'set_event_loop',
    'set_event_loop_policy', 'sleep', 'wait', 'wait_for',
    'Lock', 'Event', 'Condition', 'Semaphore', 'BoundedSemaphore',
    'Queue', 'PriorityQueue', 'LifoQueue', 'QueueEmpty', 'QueueFull',
    'subprocess', 'create_subprocess_shell', 'create_subprocess_exec',
    'WindowsProactorEventLoopPolicy', 'WindowsSelectorEventLoopPolicy',
)

import sys

# 【关键修复】先导入所有子模块，避免引用未定义的名称
try:
    from . import base_events
    from . import coroutines
    from . import events
    from . import exceptions
    from . import futures
    from . import locks
    from . import protocols
    from . import runners
    from . import queues
    from . import streams
    from . import subprocess
    from . import tasks
    from . import taskgroups
    from . import timeouts
    from . import threads
    from . import transports
    
    # 然后再从子模块导入所有内容
    from .base_events import *
    from .coroutines import *
    from .events import *
    from .exceptions import *
    from .futures import *
    from .locks import *
    from .protocols import *
    from .runners import *
    from .queues import *
    from .streams import *
    from .subprocess import *
    from .tasks import *
    from .taskgroups import *
    from .timeouts import *
    from .threads import *
    from .transports import *
    
    # Windows特定
    if sys.platform == 'win32':
        from . import windows_events
        from . import windows_utils
        from . import proactor_events
        from . import selector_events
        from .windows_events import *
        
        # 设置默认事件循环策略
        set_event_loop_policy(WindowsProactorEventLoopPolicy())
        
except ImportError as e:
    # 如果导入失败，至少提供基本功能
    print(f"Warning: asyncio import failed: {e}")
    pass
