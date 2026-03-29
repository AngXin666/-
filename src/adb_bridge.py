"""
ADB 桥接器模块
ADB Bridge Module for Android Device Communication
"""

import asyncio
import os
import subprocess
import sys
from typing import List, Optional

# 隐藏 CMD 窗口的标志
if sys.platform == 'win32':
    STARTUPINFO = subprocess.STARTUPINFO()
    STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    STARTUPINFO.wShowWindow = subprocess.SW_HIDE
    CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
    STARTUPINFO = None
    CREATE_NO_WINDOW = 0


class ADBBridge:
    """ADB 桥接器 - 封装 ADB 命令"""
    
    # [2026-03-03] 全局步骤计数器
    _click_step_counter = 0
    
    # [2026-03-29] ADB命令超时设置（秒）
    ADB_COMMAND_TIMEOUT = 10.0  # ADB命令默认超时10秒
    ADB_SCREENCAP_TIMEOUT = 5.0  # 截图命令超时5秒
    
    def __init__(self, adb_path: Optional[str] = None):
        """初始化 ADB 桥接器
        
        Args:
            adb_path: ADB 可执行文件路径，为空则使用系统 PATH
        """
        # 优先使用传入的路径，否则使用系统PATH
        if adb_path:
            self.adb_path = adb_path
        else:
            self.adb_path = "adb"
        
        # [2026-03-29] 添加ADB服务器重启标志，避免频繁重启
        self._last_restart_time = 0
        self._restart_cooldown = 60  # 重启冷却时间60秒
    
    def _run_adb(self, *args, device_id: Optional[str] = None, timeout: float = None) -> subprocess.CompletedProcess:
        """执行 ADB 命令（隐藏 CMD 窗口）
        
        Args:
            *args: ADB命令参数
            device_id: 设备ID
            timeout: 超时时间（秒），None表示使用默认超时
        """
        cmd = [self.adb_path]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(args)
        
        # [2026-03-29] 添加超时参数，避免ADB命令卡住
        if timeout is None:
            timeout = self.ADB_COMMAND_TIMEOUT
        
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            startupinfo=STARTUPINFO,
            creationflags=CREATE_NO_WINDOW,
            timeout=timeout  # 添加超时参数
        )
    
    async def _run_adb_async(self, *args, device_id: Optional[str] = None, timeout: float = None) -> subprocess.CompletedProcess:
        """异步执行 ADB 命令
        
        Args:
            *args: ADB命令参数
            device_id: 设备ID
            timeout: 超时时间（秒），None表示使用默认超时
        
        [2026-03-29] 修复原因：使用asyncio.create_subprocess_exec真正异步执行，避免阻塞
        """
        if timeout is None:
            timeout = self.ADB_COMMAND_TIMEOUT
        
        cmd = [self.adb_path]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(args)
        
        try:
            # 使用asyncio.create_subprocess_exec真正异步执行
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                startupinfo=STARTUPINFO,
                creationflags=CREATE_NO_WINDOW
            )
            
            # 使用asyncio.wait_for控制超时
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            # 构造返回对象（兼容subprocess.CompletedProcess）
            class AsyncResult:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout.decode('utf-8', errors='ignore') if stdout else ""
                    self.stderr = stderr.decode('utf-8', errors='ignore') if stderr else ""
            
            return AsyncResult(process.returncode, stdout, stderr)
            
        except asyncio.TimeoutError:
            # 超时后杀死进程
            try:
                process.kill()
                await process.wait()
            except:
                pass
            raise subprocess.TimeoutExpired(cmd, timeout)
    
    async def connect(self, device_id: str) -> bool:
        """连接到指定设备
        
        Args:
            device_id: 设备 ID（如 127.0.0.1:62001）
            
        Returns:
            连接是否成功
        """
        try:
            result = await self._run_adb_async("connect", device_id)
            return "connected" in result.stdout.lower()
        except Exception:
            return False
    
    async def disconnect(self, device_id: str) -> bool:
        """断开设备连接"""
        try:
            result = await self._run_adb_async("disconnect", device_id)
            return result.returncode == 0
        except Exception:
            return False
    
    async def shell(self, device_id: str, command: str) -> str:
        """执行 shell 命令
        
        Args:
            device_id: 设备 ID
            command: Shell 命令
            
        Returns:
            命令输出
        
        [2026-03-29] 修复原因：添加重试机制、详细日志和自动重启ADB服务器
        """
        max_retries = 2  # 最多重试2次
        retry_delay = 0.5  # 重试间隔0.5秒
        
        for attempt in range(max_retries + 1):
            try:
                result = await self._run_adb_async("shell", command, device_id=device_id)
                return result.stdout
            except subprocess.TimeoutExpired as e:
                if attempt < max_retries:
                    # 超时后等待一下再重试
                    cmd_preview = command[:50] + "..." if len(command) > 50 else command
                    print(f"[ADB] shell 超时 (尝试 {attempt + 1}/{max_retries + 1})，命令: {cmd_preview}，{retry_delay}秒后重试...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # 重试次数用完，尝试重启ADB服务器
                    cmd_preview = command[:50] + "..." if len(command) > 50 else command
                    print(f"[ADB] shell 超时失败 (已重试{max_retries}次)，命令: {cmd_preview}")
                    
                    # 如果是简单命令（如echo test）超时，说明ADB服务器有问题，尝试重启
                    if 'echo' in command or 'test' in command:
                        print(f"[ADB] 检测到简单命令超时，尝试重启ADB服务器...")
                        await self.restart_adb_server()
                    
                    return ""
            except Exception as e:
                cmd_preview = command[:50] + "..." if len(command) > 50 else command
                print(f"[ADB] shell 异常，命令: {cmd_preview}: {e}")
                return ""

    async def tap(self, device_id: str, x: int, y: int) -> bool:
        """点击指定坐标
        
        Args:
            device_id: 设备 ID
            x: X 坐标
            y: Y 坐标
            
        Returns:
            操作是否成功
        """
        # [2026-03-29] 修复：移除调试截图逻辑，避免截图操作干扰触摸事件导致上划手势
        try:
            result = await self._run_adb_async(
                "shell", f"input tap {x} {y}",
                device_id=device_id
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def swipe(self, device_id: str, x1: int, y1: int, x2: int, y2: int, 
                    duration: int = 300) -> bool:
        """滑动操作
        
        Args:
            device_id: 设备 ID
            x1, y1: 起始坐标
            x2, y2: 结束坐标
            duration: 滑动持续时间（毫秒）
            
        Returns:
            操作是否成功
        """
        try:
            result = await self._run_adb_async(
                "shell", f"input swipe {x1} {y1} {x2} {y2} {duration}",
                device_id=device_id
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def get_screen_size(self, device_id: str) -> Optional[tuple]:
        """获取屏幕尺寸
        
        Args:
            device_id: 设备 ID
            
        Returns:
            (width, height) 或 None
        """
        try:
            result = await self._run_adb_async(
                "shell", "wm size",
                device_id=device_id
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                # 输出格式通常是: "Physical size: 1080x1920"
                if ":" in output:
                    size_part = output.split(":")[-1].strip()
                    if "x" in size_part:
                        width, height = size_part.split("x")
                        return (int(width), int(height))
            
            return None
        except Exception:
            return None
    
    async def input_text(self, device_id: str, text: str) -> bool:
        """输入文本
        
        Args:
            device_id: 设备 ID
            text: 要输入的文本
            
        Returns:
            操作是否成功
        """
        try:
            # 转义特殊字符
            escaped_text = text.replace(" ", "%s").replace("'", "\\'")
            result = await self._run_adb_async(
                "shell", f"input text '{escaped_text}'",
                device_id=device_id
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def clear_input(self, device_id: str) -> bool:
        """清空当前输入框
        
        Args:
            device_id: 设备 ID
            
        Returns:
            操作是否成功
        """
        try:
            # 全选并删除
            result1 = await self._run_adb_async(
                "shell", "input keyevent KEYCODE_CTRL_A",
                device_id=device_id
            )
            await asyncio.sleep(0.1)
            result2 = await self._run_adb_async(
                "shell", "input keyevent KEYCODE_DEL",
                device_id=device_id
            )
            return result1.returncode == 0 and result2.returncode == 0
        except Exception:
            return False
    
    async def screencap(self, device_id: str) -> bytes:
        """截取屏幕
        
        Args:
            device_id: 设备 ID
            
        Returns:
            PNG 图像数据
        """
        # [2026-03-29] 添加超时参数，避免截图命令卡住
        cmd = [self.adb_path, "-s", device_id, "exec-out", "screencap", "-p"]
        result = subprocess.run(
            cmd, 
            capture_output=True,
            startupinfo=STARTUPINFO,
            creationflags=CREATE_NO_WINDOW,
            timeout=self.ADB_SCREENCAP_TIMEOUT  # 截图命令5秒超时
        )
        return result.stdout
    
    async def screencap_to_file(self, device_id: str, local_path: str) -> bool:
        """截取屏幕并保存到文件
        
        Args:
            device_id: 设备 ID
            local_path: 本地保存路径
            
        Returns:
            操作是否成功
        """
        try:
            data = await self.screencap(device_id)
            if data:
                with open(local_path, 'wb') as f:
                    f.write(data)
                return True
            return False
        except Exception:
            return False

    async def start_app(self, device_id: str, package_name: str, 
                        activity_name: Optional[str] = None) -> bool:
        """启动应用
        
        Args:
            device_id: 设备 ID
            package_name: 应用包名
            activity_name: Activity 名称（可选）
            
        Returns:
            操作是否成功
        """
        try:
            if activity_name:
                # 如果提供了Activity名称,使用am start -n命令
                component = f"{package_name}/{activity_name}"
                cmd = f"am start -n {component}"
            else:
                # 如果没有提供Activity,尝试使用monkey命令
                # 但对于某些模拟器(如MuMu),monkey可能不工作
                # 所以我们先尝试monkey,如果失败则尝试am start
                cmd = f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
            
            # [2026-03-12] 优化日志：移除CMD控制台的DEBUG启动信息
            result = await self._run_adb_async("shell", cmd, device_id=device_id)
            
            # 检查是否成功
            if result.returncode == 0:
                return True
            
            # 如果monkey失败且没有提供Activity,尝试使用am start
            if not activity_name:
                # 尝试使用am start启动主Activity
                cmd = f"am start -W -S {package_name}"
                # [2026-03-12] 优化日志：移除CMD控制台的DEBUG启动信息
                result = await self._run_adb_async("shell", cmd, device_id=device_id)
                return result.returncode == 0
            
            return False
        except Exception as e:
            # [2026-03-12] 优化日志：移除CMD控制台的DEBUG启动信息
            return False
    
    async def stop_app(self, device_id: str, package_name: str) -> bool:
        """停止应用
        
        Args:
            device_id: 设备 ID
            package_name: 应用包名
            
        Returns:
            操作是否成功
        
        [2026-03-29] 修复原因：添加重试机制和详细日志，解决ADB命令偶发超时问题
        """
        max_retries = 2  # 最多重试2次
        retry_delay = 0.5  # 重试间隔0.5秒
        
        for attempt in range(max_retries + 1):
            try:
                result = await self._run_adb_async(
                    "shell", f"am force-stop {package_name}",
                    device_id=device_id
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired as e:
                if attempt < max_retries:
                    # 超时后等待一下再重试
                    print(f"[ADB] stop_app 超时 (尝试 {attempt + 1}/{max_retries + 1})，{retry_delay}秒后重试...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # 重试次数用完，返回失败
                    print(f"[ADB] stop_app 超时失败 (已重试{max_retries}次): {e}")
                    return False
            except Exception as e:
                print(f"[ADB] stop_app 异常: {e}")
                return False
    
    async def is_app_installed(self, device_id: str, package_name: str) -> bool:
        """检查应用是否已安装
        
        Args:
            device_id: 设备 ID
            package_name: 应用包名
            
        Returns:
            是否已安装
        """
        try:
            # 直接使用 pm list packages 然后在 Python 中过滤
            result = await self._run_adb_async(
                "shell", "pm list packages",
                device_id=device_id
            )
            if result.stdout:
                return f"package:{package_name}" in result.stdout
            return False
        except Exception:
            return False
            return False
    
    async def get_installed_packages(self, device_id: str, 
                                     keyword: Optional[str] = None) -> List[str]:
        """获取已安装应用列表
        
        Args:
            device_id: 设备 ID
            keyword: 过滤关键字（可选）
            
        Returns:
            包名列表
        """
        try:
            result = await self._run_adb_async(
                "shell", "pm list packages",
                device_id=device_id
            )
            
            packages = []
            for line in result.stdout.strip().split('\n'):
                if line.startswith("package:"):
                    pkg = line.replace("package:", "").strip()
                    if keyword is None or keyword.lower() in pkg.lower():
                        packages.append(pkg)
            
            return packages
        except Exception:
            return []
    
    async def find_package_by_name(self, device_id: str, app_name: str) -> Optional[str]:
        """通过应用名称查找包名
        
        Args:
            device_id: 设备 ID
            app_name: 应用名称关键字（如"溪盟"）
            
        Returns:
            匹配的包名，未找到返回 None
        """
        # 首先尝试在包名中搜索
        packages = await self.get_installed_packages(device_id, app_name)
        if packages:
            return packages[0]
        
        # 如果是中文名称，尝试通过 dumpsys 搜索应用标签
        try:
            # 获取所有第三方应用
            result = await self._run_adb_async(
                "shell", "pm list packages -3",
                device_id=device_id
            )
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith("package:"):
                        pkg = line.replace("package:", "").strip()
                        # 获取应用标签
                        label_result = await self._run_adb_async(
                            "shell", f"dumpsys package {pkg} | grep -A1 'labelRes'",
                            device_id=device_id
                        )
                        if app_name in (label_result.stdout or ""):
                            return pkg
        except Exception:
            pass
        
        # 常见的溪盟商城包名
        common_packages = [
            "com.ry.xmsc",      # 溪盟商城实际包名
            "com.ximeng.mall",
            "com.ximeng.shop", 
            "com.ximeng.store",
            "com.ximeng",
            "cn.ximeng.mall",
            "cn.ximeng.shop",
        ]
        
        for pkg in common_packages:
            if await self.is_app_installed(device_id, pkg):
                return pkg
        
        return None
    
    async def get_all_apps_with_labels(self, device_id: str) -> List[tuple]:
        """获取所有第三方应用及其标签
        
        Args:
            device_id: 设备 ID
            
        Returns:
            [(包名, 标签), ...] 列表
        """
        apps = []
        try:
            # 获取所有第三方应用
            result = await self._run_adb_async(
                "shell", "pm list packages -3",
                device_id=device_id
            )
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith("package:"):
                        pkg = line.replace("package:", "").strip()
                        apps.append((pkg, pkg))  # 默认标签为包名
        except Exception:
            pass
        
        return apps
    
    async def get_current_activity(self, device_id: str) -> Optional[str]:
        """获取当前前台 Activity
        
        Args:
            device_id: 设备 ID
            
        Returns:
            当前 Activity 名称
        """
        try:
            result = await self._run_adb_async(
                "shell", "dumpsys activity activities | grep mResumedActivity",
                device_id=device_id
            )
            return result.stdout.strip() if result.stdout else None
        except Exception:
            return None
    
    async def is_app_in_foreground(self, device_id: str, package_name: str) -> bool:
        """检查应用是否在前台运行
        
        Args:
            device_id: 设备 ID
            package_name: 应用包名
            
        Returns:
            bool: 应用是否在前台
        """
        try:
            # 方法1：检查当前Activity
            result = await self._run_adb_async(
                "shell", "dumpsys activity activities | grep mResumedActivity",
                device_id=device_id
            )
            if result.stdout and package_name in result.stdout:
                return True
            
            # 方法2：检查顶层窗口
            result = await self._run_adb_async(
                "shell", "dumpsys window windows | grep -E 'mCurrentFocus|mFocusedApp'",
                device_id=device_id
            )
            if result.stdout and package_name in result.stdout:
                return True
            
            return False
        except Exception:
            return False
    
    async def key_event(self, device_id: str, keycode: int) -> bool:
        """发送按键事件
        
        Args:
            device_id: 设备 ID
            keycode: 按键代码（如 KEYCODE_BACK=4）
            
        Returns:
            操作是否成功
        """
        try:
            result = await self._run_adb_async(
                "shell", f"input keyevent {keycode}",
                device_id=device_id
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def press_back(self, device_id: str) -> bool:
        """按返回键"""
        return await self.key_event(device_id, 4)
    
    async def press_home(self, device_id: str) -> bool:
        """按 Home 键"""
        return await self.key_event(device_id, 3)
    
    async def restart_adb_server(self) -> bool:
        """重启ADB服务器
        
        Returns:
            是否成功
        
        [2026-03-29] 添加原因：解决ADB服务器卡住导致命令超时的问题
        """
        import time
        
        # 检查冷却时间，避免频繁重启
        current_time = time.time()
        if current_time - self._last_restart_time < self._restart_cooldown:
            print(f"[ADB] 跳过重启（冷却中，剩余{int(self._restart_cooldown - (current_time - self._last_restart_time))}秒）")
            return False
        
        try:
            print("[ADB] 正在重启ADB服务器...")
            
            # 1. 停止ADB服务器
            result = await self._run_adb_async("kill-server", timeout=5.0)
            await asyncio.sleep(1.0)
            
            # 2. 启动ADB服务器
            result = await self._run_adb_async("start-server", timeout=10.0)
            await asyncio.sleep(2.0)
            
            # 更新重启时间
            self._last_restart_time = current_time
            
            print("[ADB] ✓ ADB服务器重启成功")
            return True
            
        except Exception as e:
            print(f"[ADB] ❌ ADB服务器重启失败: {e}")
            return False
    
    async def pull(self, device_id: str, remote_path: str, local_path: str) -> bool:
        """从设备拉取文件到本地
        
        Args:
            device_id: 设备 ID
            remote_path: 设备上的文件路径
            local_path: 本地保存路径
            
        Returns:
            操作是否成功
        """
        try:
            result = await self._run_adb_async(
                "pull", remote_path, local_path,
                device_id=device_id
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def push(self, device_id: str, local_path: str, remote_path: str) -> bool:
        """从本地推送文件到设备
        
        Args:
            device_id: 设备 ID
            local_path: 本地文件路径
            remote_path: 设备上的目标路径
            
        Returns:
            操作是否成功
        """
        try:
            result = await self._run_adb_async(
                "push", local_path, remote_path,
                device_id=device_id
            )
            return result.returncode == 0
        except Exception:
            return False
