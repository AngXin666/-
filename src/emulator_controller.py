"""
模拟器控制器模块 - 支持MuMu模拟器
Emulator Controller Module - Support MuMu Emulator
"""

import asyncio
import os
import subprocess
import sys
import time
import winreg
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .models import EmulatorInstance, InstanceStatus

# Windows API导入
if sys.platform == 'win32':
    import win32gui
    import win32con

# 隐藏 CMD 窗口的标志
if sys.platform == 'win32':
    STARTUPINFO = subprocess.STARTUPINFO()
    STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    STARTUPINFO.wShowWindow = subprocess.SW_HIDE
    CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
else:
    STARTUPINFO = None
    CREATE_NO_WINDOW = 0


class EmulatorType(Enum):
    """模拟器类型"""
    MUMU = "mumu"         # MuMu模拟器
    UNKNOWN = "unknown"


class EmulatorController:
    """MuMu模拟器控制器"""
    
    # MuMu模拟器路径
    MUMU_PATHS = [
        # MuMu 12.0 新版本路径
        r"D:\Program Files\Netease\MuMuPlayer-12.0\shell",
        r"C:\Program Files\Netease\MuMuPlayer-12.0\shell",
        r"D:\Program Files (x86)\Netease\MuMuPlayer-12.0\shell",
        r"C:\Program Files (x86)\Netease\MuMuPlayer-12.0\shell",
        # MuMu nx_main 路径
        r"D:\Program Files\Netease\MuMu\nx_main",
        r"C:\Program Files\Netease\MuMu\nx_main",
        r"D:\Program Files (x86)\Netease\MuMu\nx_main",
        r"C:\Program Files (x86)\Netease\MuMu\nx_main",
        # MuMu 旧版本路径
        r"D:\MuMu\emulator\nemu\vmonitor\bin",
        r"C:\MuMu\emulator\nemu\vmonitor\bin",
    ]
    
    # 注册表路径
    MUMU_REGISTRY = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Netease\MuMuPlayer-12.0"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Netease\MuMuPlayer-12.0"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Netease\MuMuPlayer-12.0"),
    ]
    
    def __init__(self, emulator_path: Optional[str] = None, max_retry: int = 3):
        """初始化模拟器控制器
        
        Args:
            emulator_path: 模拟器路径，为空则自动检测
            max_retry: 最大重试次数
        """
        self.emulator_path = emulator_path
        self.max_retry = max_retry
        self.emulator_type = EmulatorType.UNKNOWN
        self._console_path: Optional[str] = None
        self._adb_path: Optional[str] = None
        
        if emulator_path:
            self._detect_emulator_type(emulator_path)
        else:
            self._auto_detect()
    
    def _detect_emulator_type(self, path: str) -> None:
        """根据路径检测模拟器类型"""
        path_lower = path.lower()
        
        # 检测MuMu模拟器
        if "mumu" in path_lower or "netease" in path_lower:
            self.emulator_type = EmulatorType.MUMU
            self.emulator_path = path
            
            # 查找 MuMu 的可执行文件（优先 MuMuNxDevice.exe）
            for console_name in ["MuMuNxDevice.exe", "MuMuNxMain.exe", "MuMuManager.exe"]:
                console_path = os.path.join(path, console_name)
                if os.path.exists(console_path):
                    self._console_path = console_path
                    break
            
            # 智能查找 adb.exe（支持多种路径结构）
            self._find_adb(path)
    
    def _find_adb(self, base_path: str) -> None:
        """智能查找 adb.exe
        
        支持多种 MuMu 安装路径结构：
        - 直接在 base_path 下
        - 在 base_path/shell 下
        - 在 base_path/../shell 下
        - 在 base_path/../../shell 下
        - 递归搜索（最多3层）
        
        Args:
            base_path: 基础路径
        """
        # 可能的 adb.exe 路径（按优先级排序）
        possible_paths = [
            # 1. 直接在当前目录
            os.path.join(base_path, "adb.exe"),
            # 2. 在 shell 子目录
            os.path.join(base_path, "shell", "adb.exe"),
            # 3. 在上级目录的 shell
            os.path.join(base_path, "..", "shell", "adb.exe"),
            # 4. 在上上级目录的 shell
            os.path.join(base_path, "..", "..", "shell", "adb.exe"),
            # 5. 在 nx_device/12.0/shell（常见的 MuMu 12 结构）
            os.path.join(base_path, "..", "..", "nx_device", "12.0", "shell", "adb.exe"),
            # 6. 在 nx_main 同级的 nx_device
            os.path.join(base_path, "..", "nx_device", "12.0", "shell", "adb.exe"),
        ]
        
        # 尝试所有可能的路径
        for adb_path in possible_paths:
            abs_path = os.path.abspath(adb_path)
            if os.path.exists(abs_path):
                self._adb_path = abs_path
                print(f"[ADB] 找到 ADB: {abs_path}")
                return
        
        # 如果还没找到，递归搜索（最多3层）
        print(f"[ADB] 在常见路径未找到，开始递归搜索...")
        found_path = self._search_adb_recursive(base_path, max_depth=3)
        if found_path:
            self._adb_path = found_path
            print(f"[ADB] 递归搜索找到 ADB: {found_path}")
        else:
            print(f"[ADB] 警告：未找到 adb.exe，部分功能可能无法使用")
    
    def _search_adb_recursive(self, base_path: str, max_depth: int = 3, current_depth: int = 0) -> Optional[str]:
        """递归搜索 adb.exe
        
        Args:
            base_path: 搜索起始路径
            max_depth: 最大搜索深度
            current_depth: 当前深度
            
        Returns:
            找到的 adb.exe 路径，未找到返回 None
        """
        if current_depth > max_depth:
            return None
        
        try:
            # 检查当前目录
            adb_path = os.path.join(base_path, "adb.exe")
            if os.path.exists(adb_path):
                return os.path.abspath(adb_path)
            
            # 递归搜索子目录
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    # 跳过一些明显不相关的目录
                    skip_dirs = ['temp', 'cache', 'log', 'backup', 'download']
                    if any(skip in item.lower() for skip in skip_dirs):
                        continue
                    
                    result = self._search_adb_recursive(item_path, max_depth, current_depth + 1)
                    if result:
                        return result
        except (PermissionError, OSError):
            pass
        
        return None

    def _auto_detect(self) -> None:
        """自动检测模拟器"""
        # 检测MuMu模拟器
        path = self._detect_mumu()
        if path:
            self._detect_emulator_type(path)
            return
    
    
    def _scan_directory_for_mumu(self, base_path: str, max_depth: int = 3) -> Optional[str]:
        """递归扫描目录查找MuMu模拟器可执行文件
        
        Args:
            base_path: 基础路径
            max_depth: 最大扫描深度
            
        Returns:
            找到的路径,如果没找到返回None
        """
        # MuMu可执行文件列表(按优先级排序)
        # 优先查找 nx_main 目录（主控制台）
        mumu_executables = ["MuMuNxMain.exe", "MuMuManager.exe", "adb.exe"]
        
        def scan_recursive(current_path: str, depth: int) -> Optional[str]:
            """递归扫描函数"""
            if depth > max_depth:
                return None
            
            try:
                # 检查当前目录是否有MuMu可执行文件
                for exe in mumu_executables:
                    exe_path = os.path.join(current_path, exe)
                    if os.path.exists(exe_path):
                        return current_path
                
                # 递归扫描子目录
                for item in os.listdir(current_path):
                    item_path = os.path.join(current_path, item)
                    if os.path.isdir(item_path):
                        # 跳过一些明显不相关的目录
                        skip_dirs = ['$', 'temp', 'cache', 'log', 'backup']
                        if any(skip in item.lower() for skip in skip_dirs):
                            continue
                        
                        result = scan_recursive(item_path, depth + 1)
                        if result:
                            return result
            except (PermissionError, OSError):
                pass
            
            return None
        
        return scan_recursive(base_path, 0)
    
    def _scan_directory_for_mumu_device(self, base_path: str, max_depth: int = 4) -> Optional[str]:
        """递归扫描目录查找 MuMuNxDevice.exe
        
        Args:
            base_path: 基础路径
            max_depth: 最大扫描深度
            
        Returns:
            找到 MuMuNxDevice.exe 的目录路径
        """
        def scan_recursive(current_path: str, depth: int) -> Optional[str]:
            """递归扫描函数"""
            if depth > max_depth:
                return None
            
            try:
                # 检查当前目录是否有 MuMuNxDevice.exe
                device_exe = os.path.join(current_path, "MuMuNxDevice.exe")
                if os.path.exists(device_exe):
                    return current_path
                
                # 递归扫描子目录
                for item in os.listdir(current_path):
                    item_path = os.path.join(current_path, item)
                    if os.path.isdir(item_path):
                        # 跳过一些明显不相关的目录
                        skip_dirs = ['$', 'temp', 'cache', 'log', 'backup']
                        if any(skip in item.lower() for skip in skip_dirs):
                            continue
                        
                        result = scan_recursive(item_path, depth + 1)
                        if result:
                            return result
            except (PermissionError, OSError):
                pass
            
            return None
        
        return scan_recursive(base_path, 0)
    def _detect_mumu(self) -> Optional[str]:
        """检测MuMu模拟器路径
        
        返回包含 MuMuNxDevice.exe 的路径（nx_device\12.0\shell）
        这是正确的启动客户端，不会启动游戏中心
        """
        # 从注册表检测
        for hkey, subkey in self.MUMU_REGISTRY:
            try:
                with winreg.OpenKey(hkey, subkey) as key:
                    install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                    if install_path and os.path.exists(install_path):
                        # 优先查找 MuMuNxDevice.exe（正确的启动客户端）
                        device_shell_path = os.path.join(install_path, "nx_device", "12.0", "shell")
                        if os.path.exists(device_shell_path):
                            device_exe = os.path.join(device_shell_path, "MuMuNxDevice.exe")
                            if os.path.exists(device_exe):
                                return device_shell_path
                        
                        # 递归扫描查找 MuMuNxDevice.exe
                        found_path = self._scan_directory_for_mumu_device(install_path, max_depth=4)
                        if found_path:
                            return found_path
            except (FileNotFoundError, OSError):
                continue
        
        # 从常见路径检测 - 更新为查找 nx_device 路径
        device_paths = [
            r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell",
            r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell",
            r"D:\Program Files (x86)\Netease\MuMu\nx_device\12.0\shell",
            r"C:\Program Files (x86)\Netease\MuMu\nx_device\12.0\shell",
        ]
        
        for path in device_paths:
            if os.path.exists(path):
                device_exe = os.path.join(path, "MuMuNxDevice.exe")
                if os.path.exists(device_exe):
                    return path
        
        # 兼容旧的检测逻辑（如果上面都找不到）
        for path in self.MUMU_PATHS:
            if os.path.exists(path):
                for exe in ["MuMuNxDevice.exe", "MuMuNxMain.exe", "adb.exe"]:
                    if os.path.exists(os.path.join(path, exe)):
                        return path
        
        return None
    
    @classmethod
    def auto_detect_path(cls) -> Optional[str]:
        """自动检测模拟器路径（类方法）"""
        controller = cls()
        return controller.emulator_path
    
    @classmethod
    def detect_all_emulators(cls) -> List[Tuple[EmulatorType, str]]:
        """检测所有已安装的模拟器"""
        found = []
        controller = cls()
        
        # 检测MuMu
        path = controller._detect_mumu()
        if path:
            found.append((EmulatorType.MUMU, path))
        
        return found
    
    @classmethod
    def search_in_drive(cls, drive_letter: str) -> List[str]:
        """在指定盘符中搜索 MuMu 模拟器
        
        Args:
            drive_letter: 盘符，如 "C:" 或 "D:"
            
        Returns:
            找到的所有 MuMu 模拟器路径列表
        """
        found_paths = []
        
        # 确保盘符格式正确
        if not drive_letter.endswith(':'):
            drive_letter = drive_letter.rstrip('\\') + ':'
        
        # 常见的 MuMu 安装位置
        common_paths = [
            os.path.join(drive_letter, "\\", "Program Files", "Netease"),
            os.path.join(drive_letter, "\\", "Program Files (x86)", "Netease"),
            os.path.join(drive_letter, "\\", "Netease"),
            os.path.join(drive_letter, "\\", "MuMu"),
        ]
        
        # MuMu 可执行文件列表
        mumu_executables = ["MuMuNxMain.exe", "MuMuNxDevice.exe", "adb.exe"]
        
        # 搜索常见路径
        for base_path in common_paths:
            if not os.path.exists(base_path):
                continue
            
            try:
                # 递归搜索该路径下的 MuMu 安装
                for root, dirs, files in os.walk(base_path):
                    # 检查是否包含 MuMu 可执行文件
                    if any(exe in files for exe in mumu_executables):
                        # 验证这是一个有效的 MuMu 目录
                        if cls._is_valid_mumu_path(root):
                            if root not in found_paths:
                                found_paths.append(root)
                    
                    # 限制搜索深度，避免搜索太深
                    depth = root[len(base_path):].count(os.sep)
                    if depth >= 4:
                        dirs.clear()  # 不再深入搜索
            except (PermissionError, OSError):
                continue
        
        return found_paths
    
    @classmethod
    def _is_valid_mumu_path(cls, path: str) -> bool:
        """验证是否是有效的 MuMu 模拟器路径
        
        Args:
            path: 要验证的路径
            
        Returns:
            是否是有效的 MuMu 路径
        """
        # 检查是否包含关键文件
        key_files = ["MuMuNxDevice.exe", "MuMuNxMain.exe"]
        has_key_file = any(os.path.exists(os.path.join(path, f)) for f in key_files)
        
        # 检查路径是否包含 MuMu 相关关键词
        path_lower = path.lower()
        has_mumu_keyword = any(keyword in path_lower for keyword in ['mumu', 'netease'])
        
        return has_key_file and has_mumu_keyword
    
    async def launch_instance(self, instance_index: int = 0, timeout: int = 120) -> bool:
        """启动模拟器实例"""
        retry_count = 0
        
        while retry_count < self.max_retry:
            try:
                # MuMu模拟器：先检查是否已运行
                print(f"检查MuMu模拟器是否已运行...")
                if await self._is_running(instance_index):
                    print(f"✅ MuMu模拟器已在运行")
                    return True
                
                # 自动启动 MuMu 模拟器
                print(f"正在启动MuMu模拟器... (尝试 {retry_count + 1}/{self.max_retry})")
                if await self._launch_mumu_device(instance_index):
                    print(f"✅ MuMu模拟器启动成功")
                    return True
                else:
                    print(f"❌ MuMu模拟器启动失败")
                    retry_count += 1
                    if retry_count < self.max_retry:
                        print(f"等待10秒后重试...")
                        await asyncio.sleep(10)
                    continue
                
            except Exception as e:
                print(f"启动失败: {e}")
                retry_count += 1
                if retry_count < self.max_retry:
                    print(f"等待10秒后重试...")
                    await asyncio.sleep(10)
        
        print(f"❌ 启动失败，已重试 {self.max_retry} 次")
        return False

    async def quit_instance(self, instance_index: int = 0) -> bool:
        """关闭模拟器实例（MuMu模拟器暂不支持）"""
        # MuMu模拟器没有命令行关闭功能
        return False
    
    async def quit_all(self) -> bool:
        """关闭所有模拟器实例（MuMu模拟器暂不支持）"""
        # MuMu模拟器没有命令行关闭功能
        return False
    
    async def list_instances(self) -> List[EmulatorInstance]:
        """列出所有模拟器实例（MuMu模拟器暂不支持）"""
        # MuMu模拟器没有命令行列出实例的功能
        return []
    
    async def wait_for_boot(self, instance_index: int, timeout: int = 120) -> bool:
        """等待模拟器启动完成
        
        改进的检测逻辑：
        1. 检查ADB端口是否可用
        2. 检查模拟器窗口是否存在（通过窗口标题）
        """
        start_time = asyncio.get_event_loop().time()
        
        # 预期的窗口标题
        if instance_index == 0:
            expected_title = "MuMu安卓设备"
        else:
            expected_title = f"MuMu安卓设备-{instance_index}"
        
        print(f"等待模拟器窗口: {expected_title}")
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # 方法1: 检查ADB连接
            adb_running = await self._is_running(instance_index)
            
            # 方法2: 检查窗口是否存在
            window_exists = await self._check_window_exists(expected_title)
            
            if adb_running or window_exists:
                print(f"✅ MuMu模拟器检测成功: {expected_title}")
                if window_exists and not adb_running:
                    print(f"  → 窗口已存在，等待ADB连接...")
                    # 额外等待ADB连接建立
                    await asyncio.sleep(5)
                else:
                    # 额外等待系统完全启动
                    await asyncio.sleep(3)
                return True
            
            await asyncio.sleep(2)
        
        print(f"❌ 超时: 未检测到模拟器 {expected_title}")
        return False
    
    async def _check_window_exists(self, title: str) -> bool:
        """检查指定标题的窗口是否存在
        
        Args:
            title: 窗口标题
            
        Returns:
            bool: 窗口是否存在
        """
        if sys.platform != 'win32':
            return False
        
        try:
            import win32gui
            
            found = False
            
            def callback(hwnd, _):
                nonlocal found
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if title in window_title:
                        found = True
                return True
            
            await asyncio.to_thread(win32gui.EnumWindows, callback, None)
            return found
            
        except Exception as e:
            print(f"  ⚠️ 窗口检测失败: {e}")
            return False
    
    async def _launch_mumu_device(self, instance_index: int) -> bool:
        """启动 MuMu 模拟器设备
        
        启动策略：
        1. 优先使用MuMuNxDevice（更可靠）
        2. 备用MuMuManager（可能需要特殊权限）
        
        Args:
            instance_index: 实例索引
            
        Returns:
            是否启动成功
        """
        try:
            # 方法1: 使用MuMuNxDevice直接启动（推荐，更可靠）
            device_exe = os.path.join(self.emulator_path, "MuMuNxDevice.exe")
            
            if os.path.exists(device_exe):
                print(f"使用MuMuNxDevice启动实例 {instance_index}...")
                print(f"路径: {device_exe}")
                
                # 启动指定实例（使用 -v 参数指定实例索引）
                # 不使用CREATE_NO_WINDOW，让窗口正常显示
                if instance_index == 0:
                    # 实例0不需要参数
                    subprocess.Popen([device_exe])
                else:
                    # 其他实例需要 -v 参数
                    subprocess.Popen([device_exe, "-v", str(instance_index)])
                
                print(f"✅ 启动命令已执行")
                
                # 等待进程启动
                print(f"等待模拟器进程启动...")
                await asyncio.sleep(5)
                
            else:
                # 方法2: 尝试使用MuMuManager（备用方案）
                mumu_root = os.path.dirname(os.path.dirname(os.path.dirname(self.emulator_path)))
                manager_exe = os.path.join(mumu_root, "nx_main", "MuMuManager.exe")
                
                # 如果预期路径不存在，扫描磁盘查找
                if not os.path.exists(manager_exe):
                    print(f"⚠️ 预期路径不存在: {manager_exe}")
                    print(f"正在扫描磁盘查找MuMuManager...")
                    
                    # 扫描MuMu根目录
                    found_manager = self._scan_for_file(mumu_root, "MuMuManager.exe", max_depth=3)
                    if found_manager:
                        manager_exe = found_manager
                        print(f"✓ 找到MuMuManager: {manager_exe}")
                
                if os.path.exists(manager_exe):
                    print(f"使用MuMuManager启动实例 {instance_index}...")
                    print(f"路径: {manager_exe}")
                    
                    cmd = [manager_exe, "-v", str(instance_index)]
                    subprocess.Popen(cmd)
                    
                    print(f"✅ 启动命令已执行")
                    
                    # 等待进程启动
                    print(f"等待模拟器进程启动...")
                    await asyncio.sleep(8)
                else:
                    print(f"❌ 未找到MuMu启动程序")
                    return False
            
            # 重启ADB服务器以确保连接正常
            if self._adb_path:
                print(f"重启ADB服务器以建立连接...")
                try:
                    # 停止ADB服务器
                    subprocess.run(
                        [self._adb_path, "kill-server"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        encoding='utf-8',
                        errors='ignore',
                        startupinfo=STARTUPINFO,
                        creationflags=CREATE_NO_WINDOW
                    )
                    
                    await asyncio.sleep(2)
                    
                    # 启动ADB服务器
                    subprocess.run(
                        [self._adb_path, "start-server"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        encoding='utf-8',
                        errors='ignore',
                        startupinfo=STARTUPINFO,
                        creationflags=CREATE_NO_WINDOW
                    )
                    
                    await asyncio.sleep(2)
                    
                    # 连接设备
                    adb_port = self._get_adb_port(instance_index)
                    device_id = f"127.0.0.1:{adb_port}"
                    subprocess.run(
                        [self._adb_path, "connect", device_id],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        encoding='utf-8',
                        errors='ignore',
                        startupinfo=STARTUPINFO,
                        creationflags=CREATE_NO_WINDOW
                    )
                    
                    print(f"✅ ADB服务器已重启并连接")
                except Exception as e:
                    print(f"⚠️ ADB服务器重启失败: {e}")
            
            # 等待启动完成
            print(f"等待模拟器完全启动...")
            if await self.wait_for_boot(instance_index, timeout=20):
                # 启动成功后，最小化MuMuNxMain窗口
                print(f"最小化MuMuNxMain窗口...")
                await self._minimize_mumu_main_window()
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False
    
    def _scan_for_file(self, base_path: str, filename: str, max_depth: int = 3) -> Optional[str]:
        """扫描目录查找指定文件
        
        Args:
            base_path: 基础路径
            filename: 文件名
            max_depth: 最大扫描深度
            
        Returns:
            找到的文件完整路径，未找到返回None
        """
        def scan_recursive(current_path: str, depth: int) -> Optional[str]:
            if depth > max_depth:
                return None
            
            try:
                # 检查当前目录
                target_file = os.path.join(current_path, filename)
                if os.path.exists(target_file):
                    return target_file
                
                # 递归扫描子目录
                for item in os.listdir(current_path):
                    item_path = os.path.join(current_path, item)
                    if os.path.isdir(item_path):
                        # 跳过不相关的目录
                        skip_dirs = ['$', 'temp', 'cache', 'log', 'backup']
                        if any(skip in item.lower() for skip in skip_dirs):
                            continue
                        
                        result = scan_recursive(item_path, depth + 1)
                        if result:
                            return result
            except (PermissionError, OSError):
                pass
            
            return None
        
        return scan_recursive(base_path, 0)
    
    async def _minimize_mumu_main_window(self) -> bool:
        """最小化MuMuNxMain窗口（保留MuMu安卓设备窗口）"""
        if sys.platform != 'win32':
            return False
        
        try:
            # 等待窗口出现
            max_attempts = 10
            for attempt in range(max_attempts):
                windows = []
                
                def callback(hwnd, _):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if title:
                            windows.append((hwnd, title))
                    return True
                
                win32gui.EnumWindows(callback, None)
                
                # 查找MuMuNxMain窗口
                for hwnd, title in windows:
                    # MuMuNxMain的窗口通常包含"MuMu模拟器"或"游戏"
                    if ("MuMu模拟器" in title or "游戏" in title) and "安卓设备" not in title:
                        print(f"找到MuMuNxMain窗口: {title}")
                        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                        print(f"✅ MuMuNxMain窗口已最小化")
                        return True
                
                await asyncio.sleep(1)
            
            print(f"⚠️ 未找到MuMuNxMain窗口（可能已经最小化）")
            return False
            
        except Exception as e:
            print(f"⚠️ 最小化窗口失败: {e}")
            return False
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False
    
    async def _is_running(self, instance_index: int) -> bool:
        """检查实例是否运行中
        
        同时检查窗口和ADB连接，两者都确认才返回True
        """
        try:
            # 方法1: 检查窗口是否存在
            if instance_index == 0:
                expected_title = "MuMu安卓设备"
            else:
                expected_title = f"MuMu安卓设备-{instance_index}"
            
            window_exists = await self._check_window_exists(expected_title)
            
            # 方法2: 检查ADB连接
            if not self._adb_path:
                return window_exists  # 如果没有ADB，只能依赖窗口检测
            
            adb_port = self._get_adb_port(instance_index)
            device_id = f"127.0.0.1:{adb_port}"
            
            # 快速检查ADB设备列表
            try:
                devices_result = subprocess.run(
                    [self._adb_path, "devices"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    encoding='utf-8',
                    errors='ignore',
                    startupinfo=STARTUPINFO,
                    creationflags=CREATE_NO_WINDOW
                )
                
                # 检查设备是否在列表中且状态为device
                adb_connected = False
                for line in devices_result.stdout.split('\n'):
                    if device_id in line and "device" in line and "offline" not in line:
                        adb_connected = True
                        break
                
                # 两者都确认才返回True
                return window_exists and adb_connected
                
            except subprocess.TimeoutExpired:
                return False
                
        except Exception as e:
            return False
    
    async def get_running_instances(self) -> List[int]:
        """获取所有正在运行的MuMu模拟器实例ID
        
        优化版：使用并发连接
        
        Returns:
            正在运行的实例ID列表
        """
        running_instances = []
        
        try:
            # MuMu模拟器：快速连接所有可能的实例端口（0-9）
            if not self._adb_path:
                return []
            
            print(f"正在快速扫描MuMu实例...")
            
            # 使用线程池并发连接（避免复杂的异步进程）
            def try_connect_sync(instance_id: int):
                """同步尝试连接单个实例"""
                port = 16384 + instance_id * 32
                device_id = f"127.0.0.1:{port}"
                
                try:
                    # 使用更短的超时（1秒）
                    subprocess.run(
                        [self._adb_path, "connect", device_id],
                        capture_output=True,
                        text=True,
                        timeout=1,
                        encoding='utf-8',
                        errors='ignore',
                        startupinfo=STARTUPINFO,
                        creationflags=CREATE_NO_WINDOW
                    )
                    return True
                except (subprocess.TimeoutExpired, Exception):
                    return False
            
            # 快速连接所有可能的实例端口（0-9）
            print(f"正在快速扫描MuMu实例...")
            
            # 使用并发连接提高速度
            import concurrent.futures
            
            def try_connect_instance(instance_id: int) -> bool:
                """尝试连接单个实例"""
                port = 16384 + instance_id * 32
                device_id = f"127.0.0.1:{port}"
                
                try:
                    # 使用更短的超时（0.5秒）
                    result = subprocess.run(
                        [self._adb_path, "connect", device_id],
                        capture_output=True,
                        text=True,
                        timeout=0.5,
                        encoding='utf-8',
                        errors='ignore',
                        startupinfo=STARTUPINFO,
                        creationflags=CREATE_NO_WINDOW
                    )
                    return True
                except (subprocess.TimeoutExpired, Exception):
                    return False
            
            # 并发连接所有可能的实例（0-9）
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(try_connect_instance, i): i for i in range(10)}
                # 等待所有连接完成（最多0.5秒）
                concurrent.futures.wait(futures, timeout=0.6)
            
            # 短暂等待连接稳定
            await asyncio.sleep(0.2)
            
            # 获取ADB设备列表
            devices_result = subprocess.run(
                [self._adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=5,
                encoding='utf-8',
                errors='ignore',
                startupinfo=STARTUPINFO,
                creationflags=CREATE_NO_WINDOW
            )
            
            # 解析设备列表，提取端口号
            print(f"ADB devices输出:")
            print(devices_result.stdout)
            
            for line in devices_result.stdout.split('\n'):
                if '127.0.0.1:' in line and 'device' in line:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == 'device':
                        # 提取端口号
                        device_id = parts[0]
                        try:
                            port = int(device_id.split(':')[1])
                            # 根据端口号反推实例ID
                            # MuMu端口规则：16384 + instance_id * 32
                            if port >= 16384 and (port - 16384) % 32 == 0:
                                instance_id = (port - 16384) // 32
                                running_instances.append(instance_id)
                                print(f"检测到MuMu实例 {instance_id} (端口 {port})")
                            else:
                                print(f"跳过非MuMu端口: {port}")
                        except (ValueError, IndexError) as e:
                            print(f"解析端口失败: {device_id}, 错误: {e}")
                            continue
                    else:
                        print(f"跳过非device状态的行: {line}")
                elif '127.0.0.1:' in line:
                    print(f"跳过非device状态的设备: {line}")
            
            return sorted(running_instances)
            
        except Exception as e:
            print(f"❌ 获取运行中实例失败: {e}")
            return []
    
    def _get_adb_port(self, instance_index: int) -> int:
        """获取 ADB 端口
        
        优先从配置文件读取，如果读取失败则使用计算方式
        """
        try:
            # 尝试从配置文件读取端口
            mumu_root = os.path.dirname(os.path.dirname(os.path.dirname(self.emulator_path)))
            config_file = os.path.join(mumu_root, "vms", f"MuMuPlayer-12.0-{instance_index}", "configs", "vm_config.json")
            
            if os.path.exists(config_file):
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    port = config.get("vm", {}).get("nat", {}).get("port_forward", {}).get("adb", {}).get("host_port")
                    if port:
                        return int(port)
        except Exception as e:
            print(f"⚠️ 读取配置文件失败: {e}")
        
        # 备用方案：使用计算方式（MuMu模拟器端口规则：16384 + 索引 * 32）
        return 16384 + instance_index * 32
    
    async def get_adb_port(self, instance_index: int) -> int:
        """获取实例的 ADB 端口"""
        return self._get_adb_port(instance_index)
    
    async def set_resolution(self, instance_index: int, width: int = 540, height: int = 960, dpi: int = 160) -> bool:
        """设置模拟器分辨率（MuMu模拟器暂不支持命令行设置）
        
        Args:
            instance_index: 实例索引
            width: 宽度
            height: 高度
            dpi: DPI
            
        Returns:
            是否成功
        """
        # MuMu模拟器没有命令行设置分辨率的功能
        return False
    
    async def get_resolution(self, instance_index: int) -> Optional[Tuple[int, int]]:
        """获取模拟器分辨率（通过 ADB）
        
        Args:
            instance_index: 实例索引
            
        Returns:
            (宽度, 高度) 或 None
        """
        try:
            if not self._adb_path:
                return None
            
            adb_port = self._get_adb_port(instance_index)
            device_id = f"127.0.0.1:{adb_port}"
            
            # 先连接 ADB（隐藏窗口）
            connect_cmd = [self._adb_path, "connect", device_id]
            subprocess.run(
                connect_cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='ignore',
                startupinfo=STARTUPINFO,
                creationflags=CREATE_NO_WINDOW
            )
            
            # 等待连接稳定
            import time
            time.sleep(1)
            
            # 使用 ADB 获取分辨率（隐藏窗口）
            cmd = [self._adb_path, "-s", device_id, "shell", "wm", "size"]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='ignore',
                startupinfo=STARTUPINFO,
                creationflags=CREATE_NO_WINDOW
            )
            
            if result.stdout:
                # 解析输出，格式如 "Physical size: 540x960"
                import re
                match = re.search(r'(\d+)x(\d+)', result.stdout)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
            
            return None
        except Exception:
            return None
    
    async def check_and_set_resolution(self, instance_index: int, 
                                        required_width: int = 540, 
                                        required_height: int = 960) -> Tuple[bool, str]:
        """检查并设置分辨率
        
        Args:
            instance_index: 实例索引
            required_width: 要求的宽度
            required_height: 要求的高度
            
        Returns:
            (是否符合要求, 消息)
        """
        current = await self.get_resolution(instance_index)
        
        if current is None:
            return False, "无法获取当前分辨率"
        
        width, height = current
        
        if width == required_width and height == required_height:
            return True, f"分辨率正确: {width}x{height}"
        
        # 分辨率不符合，尝试设置
        msg = f"当前分辨率 {width}x{height} 不符合要求 {required_width}x{required_height}"
        return False, msg
    
    def get_adb_path(self) -> Optional[str]:
        """获取 ADB 路径"""
        return self._adb_path
    
    def is_available(self) -> bool:
        """检查模拟器是否可用"""
        return self._console_path is not None and os.path.exists(self._console_path)
    
    def get_emulator_info(self) -> str:
        """获取模拟器信息"""
        if self.emulator_type == EmulatorType.MUMU:
            return f"MuMu模拟器: {self.emulator_path}"
        return "未检测到模拟器"
