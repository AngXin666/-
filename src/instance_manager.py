"""
实例管理器模块
Instance Manager Module
"""

import asyncio
from typing import Dict, List, Any, Callable, Optional

from .models import EmulatorInstance, InstanceStatus
from .emulator_controller import EmulatorController


class InstanceManager:
    """实例管理器 - 管理多个模拟器实例的并发操作"""
    
    def __init__(self, emulator_controller: EmulatorController, max_concurrent: int = 5):
        """初始化实例管理器
        
        Args:
            emulator_controller: 模拟器控制器实例
            max_concurrent: 最大并发实例数
        """
        self.emulator_controller = emulator_controller
        self.max_concurrent = max_concurrent
        self.instances: Dict[int, EmulatorInstance] = {}
    
    async def start_instances(self, count: int, timeout: int = 120) -> Dict[int, bool]:
        """批量启动实例
        
        Args:
            count: 要启动的实例数量
            timeout: 每个实例的启动超时时间
            
        Returns:
            各实例启动结果 {index: success}
        """
        # 限制并发数
        actual_count = min(count, self.max_concurrent)
        
        # 并发启动所有实例
        tasks = []
        for i in range(actual_count):
            tasks.append(self._start_single_instance(i, timeout))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {i: (r is True) for i, r in enumerate(results)}
    
    async def _start_single_instance(self, index: int, timeout: int) -> bool:
        """启动单个实例
        
        Args:
            index: 实例索引
            timeout: 超时时间
            
        Returns:
            是否成功
        """
        try:
            success = await self.emulator_controller.launch_instance(index, timeout)
            
            if success:
                # 获取 ADB 端口
                adb_port = await self.emulator_controller.get_adb_port(index)
                
                self.instances[index] = EmulatorInstance(
                    name=f"Nox_{index}",
                    index=index,
                    status=InstanceStatus.RUNNING,
                    device_id=f"127.0.0.1:{adb_port}" if adb_port else None,
                    adb_port=adb_port
                )
            
            return success
        except Exception:
            return False
    
    async def stop_instances(self, indices: Optional[List[int]] = None) -> Dict[int, bool]:
        """批量停止实例
        
        Args:
            indices: 要停止的实例索引列表，为空则停止所有
            
        Returns:
            各实例停止结果
        """
        if indices is None:
            indices = list(self.instances.keys())
        
        tasks = []
        for index in indices:
            tasks.append(self._stop_single_instance(index))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {indices[i]: (r is True) for i, r in enumerate(results)}

    async def _stop_single_instance(self, index: int) -> bool:
        """停止单个实例
        
        Args:
            index: 实例索引
            
        Returns:
            是否成功
        """
        try:
            success = await self.emulator_controller.quit_instance(index)
            
            if success and index in self.instances:
                self.instances[index].status = InstanceStatus.STOPPED
            
            return success
        except Exception:
            return False
    
    async def stop_all(self) -> bool:
        """停止所有实例
        
        Returns:
            是否全部成功
        """
        success = await self.emulator_controller.quit_all()
        
        if success:
            for instance in self.instances.values():
                instance.status = InstanceStatus.STOPPED
        
        return success
    
    async def execute_on_all(self, task: Callable, 
                             indices: Optional[List[int]] = None) -> Dict[int, Any]:
        """在所有实例上执行任务
        
        Args:
            task: 异步任务函数，接收 device_id 参数
            indices: 要执行的实例索引列表，为空则在所有运行中的实例上执行
            
        Returns:
            各实例执行结果
        """
        if indices is None:
            indices = [
                i for i, inst in self.instances.items() 
                if inst.status == InstanceStatus.RUNNING
            ]
        
        tasks = []
        valid_indices = []
        
        for index in indices:
            instance = self.instances.get(index)
            if instance and instance.device_id:
                tasks.append(self._execute_task_isolated(task, instance.device_id))
                valid_indices.append(index)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {valid_indices[i]: r for i, r in enumerate(results)}
    
    async def _execute_task_isolated(self, task: Callable, device_id: str) -> Any:
        """隔离执行任务（确保单个实例失败不影响其他实例）
        
        Args:
            task: 任务函数
            device_id: 设备 ID
            
        Returns:
            任务结果或异常
        """
        try:
            return await task(device_id)
        except Exception as e:
            return e
    
    def get_instance_status(self, index: int) -> Optional[InstanceStatus]:
        """获取实例状态
        
        Args:
            index: 实例索引
            
        Returns:
            实例状态
        """
        instance = self.instances.get(index)
        return instance.status if instance else None
    
    def get_running_instances(self) -> List[EmulatorInstance]:
        """获取所有运行中的实例
        
        Returns:
            运行中的实例列表
        """
        return [
            inst for inst in self.instances.values() 
            if inst.status == InstanceStatus.RUNNING
        ]
    
    def get_device_id(self, index: int) -> Optional[str]:
        """获取实例的设备 ID
        
        Args:
            index: 实例索引
            
        Returns:
            设备 ID
        """
        instance = self.instances.get(index)
        return instance.device_id if instance else None
