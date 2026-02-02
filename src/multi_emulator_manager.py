"""
多开模拟器管理器
Multi-Emulator Manager for handling multiple emulator instances
"""

import asyncio
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .emulator_controller import EmulatorController
from .adb_bridge import ADBBridge
from .auto_login import AutoLogin
from .ui_automation import UIAutomation
from .screen_capture import ScreenCapture


@dataclass
class AccountConfig:
    """账号配置"""
    phone: str
    password: str
    name: str


@dataclass
class EmulatorConfig:
    """模拟器配置"""
    index: int
    name: str
    adb_port: int
    device_id: str
    accounts: List[AccountConfig]


class MultiEmulatorManager:
    """多开模拟器管理器"""
    
    def __init__(self, config_path: str = "emulator_account_config.yaml"):
        """初始化管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.emulators: List[EmulatorConfig] = []
        self.settings: Dict = {}
        self.emulator_controller: Optional[EmulatorController] = None
        self.adb: Optional[ADBBridge] = None
        
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加载模拟器配置
        for emu_config in config.get('emulators', []):
            accounts = [
                AccountConfig(
                    phone=acc['phone'],
                    password=acc['password'],
                    name=acc.get('name', acc['phone'])
                )
                for acc in emu_config.get('accounts', [])
            ]
            
            self.emulators.append(EmulatorConfig(
                index=emu_config['index'],
                name=emu_config.get('name', f"模拟器{emu_config['index']}"),
                adb_port=emu_config.get('adb_port', 62001 + emu_config['index']),
                device_id=emu_config.get('device_id', f"emulator-{5554 + emu_config['index'] * 2}"),
                accounts=accounts
            ))
        
        # 加载全局设置
        self.settings = config.get('settings', {})
        
        # 初始化控制器
        self.emulator_controller = EmulatorController()
        self.adb = ADBBridge()
    
    async def launch_emulator(self, emulator: EmulatorConfig, log_callback=None) -> bool:
        """启动指定模拟器
        
        Args:
            emulator: 模拟器配置
            log_callback: 日志回调函数
            
        Returns:
            是否启动成功
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        log(f"正在启动 {emulator.name} (索引: {emulator.index})...")
        
        timeout = self.settings.get('launch_timeout', 120)
        success = await self.emulator_controller.launch_instance(emulator.index, timeout)
        
        if success:
            log(f"✅ {emulator.name} 启动成功")
            # 等待 ADB 连接
            await asyncio.sleep(5)
            return True
        else:
            log(f"❌ {emulator.name} 启动失败")
            return False
    
    async def process_emulator(self, emulator: EmulatorConfig, log_callback=None) -> Dict:
        """处理单个模拟器的所有账号
        
        Args:
            emulator: 模拟器配置
            log_callback: 日志回调函数
            
        Returns:
            处理结果统计
        """
        def log(msg):
            if log_callback:
                log_callback(f"[{emulator.name}] {msg}")
        
        result = {
            'emulator': emulator.name,
            'total': len(emulator.accounts),
            'success': 0,
            'failed': 0,
            'accounts': []
        }
        
        # 启动模拟器
        if self.settings.get('auto_launch_emulator', True):
            if not await self.launch_emulator(emulator, log_callback):
                log("模拟器启动失败，跳过该模拟器")
                result['failed'] = len(emulator.accounts)
                return result
        
        # 创建自动化组件
        ui_automation = UIAutomation(self.adb)
        screen_capture = ScreenCapture(self.adb)
        auto_login = AutoLogin(
            ui_automation, 
            screen_capture, 
            self.adb,
            enable_cache=self.settings.get('enable_cache_login', True)
        )
        
        # 处理每个账号
        for i, account in enumerate(emulator.accounts, 1):
            log(f"[{i}/{len(emulator.accounts)}] 开始处理账号: {account.name}")
            
            try:
                # 登录
                login_result = await auto_login.login(
                    emulator.device_id,
                    account.phone,
                    account.password,
                    log_callback=lambda msg: log(f"  {msg}"),
                    use_cache=True
                )
                
                if login_result.success:
                    log(f"✅ 账号 {account.name} 登录成功")
                    result['success'] += 1
                    result['accounts'].append({
                        'name': account.name,
                        'phone': account.phone,
                        'status': 'success'
                    })
                    
                    # 这里可以执行业务逻辑
                    # await self.do_business_tasks(emulator.device_id, account)
                    
                    # 退出登录
                    await self.adb.shell(emulator.device_id, "am force-stop com.ry.xmsc")
                    
                    # 等待一下再处理下一个账号
                    if i < len(emulator.accounts):
                        delay = self.settings.get('account_switch_delay', 3)
                        await asyncio.sleep(delay)
                else:
                    log(f"❌ 账号 {account.name} 登录失败: {login_result.error_message}")
                    result['failed'] += 1
                    result['accounts'].append({
                        'name': account.name,
                        'phone': account.phone,
                        'status': 'failed',
                        'error': login_result.error_message
                    })
            
            except Exception as e:
                log(f"❌ 账号 {account.name} 处理异常: {e}")
                result['failed'] += 1
                result['accounts'].append({
                    'name': account.name,
                    'phone': account.phone,
                    'status': 'error',
                    'error': str(e)
                })
        
        return result
    
    async def run_all(self, log_callback=None) -> List[Dict]:
        """运行所有模拟器和账号
        
        Args:
            log_callback: 日志回调函数
            
        Returns:
            所有模拟器的处理结果
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        log(f"开始处理 {len(self.emulators)} 个模拟器...")
        
        if self.settings.get('parallel_emulators', True):
            # 并发处理多个模拟器
            log("模式: 并发处理多个模拟器")
            tasks = [
                self.process_emulator(emu, log_callback)
                for emu in self.emulators
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    log(f"❌ {self.emulators[i].name} 处理异常: {result}")
                    final_results.append({
                        'emulator': self.emulators[i].name,
                        'error': str(result)
                    })
                else:
                    final_results.append(result)
            
            return final_results
        else:
            # 串行处理模拟器
            log("模式: 串行处理模拟器")
            results = []
            for emu in self.emulators:
                result = await self.process_emulator(emu, log_callback)
                results.append(result)
            return results
    
    async def run_specific_emulator(self, emulator_index: int, log_callback=None) -> Optional[Dict]:
        """运行指定索引的模拟器
        
        Args:
            emulator_index: 模拟器索引
            log_callback: 日志回调函数
            
        Returns:
            处理结果
        """
        emulator = next((e for e in self.emulators if e.index == emulator_index), None)
        if not emulator:
            if log_callback:
                log_callback(f"❌ 未找到索引为 {emulator_index} 的模拟器")
            return None
        
        return await self.process_emulator(emulator, log_callback)
    
    def print_summary(self, results: List[Dict]):
        """打印处理结果摘要
        
        Args:
            results: 处理结果列表
        """
        print("\n" + "="*70)
        print("处理结果摘要")
        print("="*70)
        
        total_accounts = 0
        total_success = 0
        total_failed = 0
        
        for result in results:
            if 'error' in result:
                print(f"\n{result['emulator']}: 处理异常")
                print(f"  错误: {result['error']}")
                continue
            
            print(f"\n{result['emulator']}:")
            print(f"  总账号数: {result['total']}")
            print(f"  成功: {result['success']}")
            print(f"  失败: {result['failed']}")
            
            total_accounts += result['total']
            total_success += result['success']
            total_failed += result['failed']
        
        print(f"\n{'='*70}")
        print("总计:")
        print(f"  模拟器数: {len(results)}")
        print(f"  总账号数: {total_accounts}")
        print(f"  成功: {total_success}")
        print(f"  失败: {total_failed}")
        print(f"  成功率: {total_success/total_accounts*100:.1f}%" if total_accounts > 0 else "  成功率: N/A")
        print("="*70)
