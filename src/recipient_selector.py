"""
收款人选择器模块
Recipient Selector Module

负责从多个收款人中选择一个，支持轮询和随机两种策略
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Dict


class RecipientSelector:
    """收款人选择器
    
    负责从多个收款人中选择一个，支持轮询和随机两种策略。
    
    特性：
    - 轮询选择：按顺序循环选择收款人，确保负载均衡
    - 随机选择：随机选择收款人，增加不可预测性
    - 持久化状态：轮询索引保存到文件，重启后继续
    - 自我过滤：自动过滤掉发送人自己，避免循环转账
    
    使用示例：
        # 创建选择器（默认轮询策略）
        selector = RecipientSelector(strategy="rotation")
        
        # 选择收款人
        recipients = ["13800138000", "13900139000", "14000140000"]
        selected = selector.select_recipient(
            recipients, 
            sender_phone="18888888888",
            key="user001"
        )
        
        # 使用随机策略
        selector = RecipientSelector(strategy="random")
        selected = selector.select_recipient(recipients)
    """
    
    def __init__(self, strategy: str = "rotation"):
        """初始化选择器
        
        Args:
            strategy: 选择策略，"rotation"(轮询) 或 "random"(随机)，默认为轮询
        """
        self.strategy = strategy
        self.rotation_file = Path("runtime_data/transfer_rotation.json")
        self.rotation_state = self._load_rotation_state()
    
    def select_recipient(
        self, 
        recipients: List[str], 
        sender_phone: str = None,
        key: str = None
    ) -> Optional[str]:
        """选择一个收款人
        
        根据配置的策略（轮询或随机）从收款人列表中选择一个。
        自动过滤掉发送人自己，避免循环转账。
        
        Args:
            recipients: 收款人列表（手机号）
            sender_phone: 发送人手机号（用于过滤，避免自己转给自己）
            key: 轮询键（用于区分不同的轮询组，仅轮询策略需要）
                 建议使用用户ID或管理员ID作为key
            
        Returns:
            选中的收款人手机号，如果没有可用收款人返回None
            
        Raises:
            ValueError: 如果收款人列表为空
        """
        if not recipients:
            raise ValueError("收款人列表不能为空")
        
        # 过滤掉发送人自己
        filtered_recipients = [r for r in recipients if r != sender_phone]
        
        if not filtered_recipients:
            # 所有收款人都是发送人自己，无法选择
            return None
        
        # 根据策略选择
        if self.strategy == "rotation":
            if not key:
                # 如果没有提供key，使用默认key
                key = "default"
            return self._select_by_rotation(filtered_recipients, key)
        elif self.strategy == "random":
            return self._select_by_random(filtered_recipients)
        else:
            # 未知策略，默认使用轮询
            if not key:
                key = "default"
            return self._select_by_rotation(filtered_recipients, key)
    
    def _select_by_rotation(
        self, 
        recipients: List[str], 
        key: str
    ) -> str:
        """轮询选择
        
        按顺序循环选择收款人，确保每个收款人都能被均匀选中。
        轮询状态会持久化保存，重启后继续。
        
        Args:
            recipients: 收款人列表
            key: 轮询键（用于区分不同的轮询组）
            
        Returns:
            选中的收款人
        """
        # 获取当前索引
        current_index = self.rotation_state.get(key, 0)
        
        # 确保索引在有效范围内
        if current_index >= len(recipients):
            current_index = 0
        
        # 选择收款人
        selected = recipients[current_index]
        
        # 更新索引（循环）
        next_index = (current_index + 1) % len(recipients)
        self.rotation_state[key] = next_index
        
        # 保存状态
        self._save_rotation_state()
        
        return selected
    
    def _select_by_random(self, recipients: List[str]) -> str:
        """随机选择
        
        从收款人列表中随机选择一个。
        
        Args:
            recipients: 收款人列表
            
        Returns:
            选中的收款人
        """
        return random.choice(recipients)
    
    def _load_rotation_state(self) -> Dict[str, int]:
        """加载轮询状态
        
        从文件中加载轮询状态。如果文件不存在或损坏，返回空字典。
        
        Returns:
            轮询状态字典，键为轮询键，值为当前索引
        """
        try:
            if self.rotation_file.exists():
                with open(self.rotation_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    # 验证数据格式
                    if isinstance(state, dict):
                        return state
            return {}
        except Exception as e:
            print(f"加载轮询状态失败: {e}")
            return {}
    
    def _save_rotation_state(self):
        """保存轮询状态
        
        将轮询状态保存到文件。如果目录不存在，会自动创建。
        """
        try:
            # 确保目录存在
            self.rotation_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存状态
            with open(self.rotation_file, 'w', encoding='utf-8') as f:
                json.dump(self.rotation_state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存轮询状态失败: {e}")
    
    def reset_rotation(self, key: str = None):
        """重置轮询状态
        
        将指定key的轮询索引重置为0。如果不指定key，重置所有。
        
        Args:
            key: 轮询键，如果为None则重置所有
        """
        if key is None:
            self.rotation_state = {}
        else:
            if key in self.rotation_state:
                del self.rotation_state[key]
        
        self._save_rotation_state()
    
    def get_rotation_index(self, key: str) -> int:
        """获取当前轮询索引
        
        Args:
            key: 轮询键
            
        Returns:
            当前索引，如果key不存在返回0
        """
        return self.rotation_state.get(key, 0)
