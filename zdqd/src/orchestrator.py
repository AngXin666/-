"""
任务编排器模块
Orchestrator Module
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import Account, AccountResult, Config
from .emulator_controller import EmulatorController
from .adb_bridge import ADBBridge
from .screen_capture import ScreenCapture
from .ui_automation import UIAutomation
from .auto_login import AutoLogin
from .ximeng_automation import XimengAutomation
from .account_manager import AccountManager
from .instance_manager import InstanceManager


class Orchestrator:
    """任务编排器 - 协调整个自动化流程"""
    
    def __init__(self, config: Config, account_manager: AccountManager):
        """初始化任务编排器
        
        Args:
            config: 配置对象
            account_manager: 账号管理器
            
        Raises:
            RuntimeError: 如果ModelManager未初始化
        """
        # 检查ModelManager是否已初始化
        from .model_manager import ModelManager
        model_manager = ModelManager.get_instance()
        
        if not model_manager.is_initialized():
            raise RuntimeError(
                "ModelManager未初始化。请在创建Orchestrator前调用 "
                "ModelManager.get_instance().initialize_all_models()"
            )
        
        self.config = config
        self.account_manager = account_manager
        
        # 初始化控制器
        self.emulator_controller = EmulatorController(config.nox_path)
        self.adb_bridge = ADBBridge(config.adb_path)
        
        # 初始化实例管理器
        self.instance_manager = InstanceManager(
            self.emulator_controller, 
            config.max_concurrent_instances
        )
        
        # 自动化组件将在运行时为每个实例创建
        self._automations: Dict[str, XimengAutomation] = {}
    
    def _create_automation(self, device_id: str) -> XimengAutomation:
        """为设备创建自动化组件
        
        注意：XimengAutomation会自动从ModelManager获取共享的模型实例，
        不会创建新的模型实例。
        
        Args:
            device_id: 设备 ID
            
        Returns:
            自动化器实例
        """
        if device_id not in self._automations:
            screen_capture = ScreenCapture(self.adb_bridge, self.config.screenshot_dir)
            ui_automation = UIAutomation(self.adb_bridge, screen_capture)
            auto_login = AutoLogin(ui_automation, screen_capture)
            
            # XimengAutomation会从ModelManager获取共享模型
            # 不会重复加载模型，确保内存效率
            self._automations[device_id] = XimengAutomation(
                ui_automation, screen_capture, auto_login
            )
        
        return self._automations[device_id]
    
    async def run_on_instance(self, device_id: str, account: Account, 
                             log_callback=None, stop_check=None) -> AccountResult:
        """在单个实例上执行账号任务
        
        Args:
            device_id: 设备 ID
            account: 账号信息
            log_callback: 日志回调函数
            stop_check: 停止检查函数
            
        Returns:
            账号处理结果
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        automation = self._create_automation(device_id)
        
        # 启动目标应用
        log(f"启动应用: {self.config.target_app_package}")
        # 使用完整的Activity路径启动APP,确保在MuMu模拟器上也能正常工作
        activity_name = self.config.target_app_activity if self.config.target_app_activity else None
        await self.adb_bridge.start_app(device_id, self.config.target_app_package, activity_name)
        await asyncio.sleep(3)
        
        # 使用GPU加速的启动流程处理（包含白屏卡死检测、弹窗处理、广告跳过）
        log("处理应用启动流程（GPU加速）...")
        startup_success = await automation.handle_startup_flow_integrated(
            device_id, 
            log_callback=log_callback,
            stop_check=stop_check,
            package_name=self.config.target_app_package
        )
        
        if not startup_success:
            log("启动流程失败")
            result = AccountResult(
                phone=account.phone,
                success=False,
                error_message="应用启动流程失败",
                timestamp=datetime.now()
            )
            self.account_manager.update_account_result(account.phone, result)
            return result
        
        log("启动流程完成，开始执行业务流程...")
        
        # 执行完整工作流
        result = await automation.run_full_workflow(device_id, account)
        
        # 更新账号结果
        self.account_manager.update_account_result(account.phone, result)
        
        return result

    async def run_batch(self, instance_count: int) -> Dict[int, AccountResult]:
        """批量执行任务，自动分配账号到实例
        
        Args:
            instance_count: 使用的实例数量
            
        Returns:
            各实例执行结果
        """
        results: Dict[int, AccountResult] = {}
        
        # 启动实例
        start_results = await self.instance_manager.start_instances(instance_count)
        
        # 获取成功启动的实例
        running_indices = [i for i, success in start_results.items() if success]
        
        if not running_indices:
            return results
        
        # 为每个实例分配账号并执行
        tasks = []
        task_mapping = []  # (index, account)
        
        for index in running_indices:
            account = self.account_manager.get_next_account()
            if account is None:
                break
            
            device_id = self.instance_manager.get_device_id(index)
            if device_id:
                tasks.append(self.run_on_instance(device_id, account))
                task_mapping.append((index, account))
        
        # 并发执行
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(task_results):
            index, account = task_mapping[i]
            if isinstance(result, Exception):
                results[index] = AccountResult(
                    phone=account.phone,
                    success=False,
                    error_message=str(result),
                    timestamp=datetime.now()
                )
            else:
                results[index] = result
        
        return results
    
    async def run_all_accounts(self) -> None:
        """处理所有账号"""
        # 加载账号
        accounts = self.account_manager.load_accounts()
        
        if not accounts:
            return
        
        instance_count = min(
            self.config.max_concurrent_instances,
            len(accounts)
        )
        
        # 循环处理直到所有账号完成
        while True:
            pending = self.account_manager.get_pending_accounts()
            if not pending:
                break
            
            # 批量处理
            await self.run_batch(instance_count)
            
            # 短暂等待
            await asyncio.sleep(2)
        
        # 停止所有实例
        await self.instance_manager.stop_all()
        
        # 数据已自动保存到数据库
    
    async def cleanup(self) -> None:
        """清理资源"""
        await self.instance_manager.stop_all()
        self._automations.clear()
