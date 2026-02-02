"""
转账配置管理模块
Transfer Configuration Manager Module
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


class TransferConfig:
    """转账配置类"""
    
    def __init__(self):
        self.config_file = Path("transfer_config.json")
        self.min_balance = 0.0  # 最小保留余额
        self.min_transfer_amount = 30.0  # 起步金额（最小转账金额）
        self.recipient_ids = []  # 1级收款用户ID列表（兼容旧版本）
        self.level_recipients = {  # 多级收款账号配置
            1: [],  # 1级收款账号
            2: [],  # 2级收款账号
            3: []   # 3级收款账号
        }
        self.enabled = False  # 是否启用转账功能
        self.multi_level_enabled = False  # 是否启用多级转账
        self.max_transfer_level = 1  # 最大转账级数（1-3）
        self.use_user_manager_recipients = True  # 是否使用用户管理的收款人配置（用于回滚）
        self.load()
    
    def load(self):
        """加载配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.min_balance = data.get('min_balance', 0.0)
                    self.min_transfer_amount = data.get('min_transfer_amount', 30.0)
                    
                    # 加载多级收款账号配置
                    level_recipients = data.get('level_recipients', {})
                    if level_recipients:
                        # 如果有多级配置，使用多级配置
                        self.level_recipients = {
                            1: level_recipients.get('1', level_recipients.get(1, [])),
                            2: level_recipients.get('2', level_recipients.get(2, [])),
                            3: level_recipients.get('3', level_recipients.get(3, []))
                        }
                    else:
                        # 兼容旧版本：将recipient_ids作为1级收款账号
                        old_recipients = data.get('recipient_ids', [])
                        self.level_recipients = {
                            1: old_recipients,
                            2: [],
                            3: []
                        }
                    
                    # 保持recipient_ids与level_recipients[1]同步（兼容性）
                    self.recipient_ids = self.level_recipients[1]
                    
                    self.enabled = data.get('enabled', False)
                    self.multi_level_enabled = data.get('multi_level_enabled', False)
                    self.max_transfer_level = data.get('max_transfer_level', 1)
                    # 确保级数在1-3之间
                    if self.max_transfer_level < 1:
                        self.max_transfer_level = 1
                    elif self.max_transfer_level > 3:
                        self.max_transfer_level = 3
                    
                    # 加载用户管理收款人配置开关
                    self.use_user_manager_recipients = data.get('use_user_manager_recipients', True)
            except Exception as e:
                print(f"加载转账配置失败: {e}")
    
    def save(self):
        """保存配置"""
        try:
            # 保持recipient_ids与level_recipients[1]同步
            self.recipient_ids = self.level_recipients[1]
            
            data = {
                'min_balance': self.min_balance,
                'min_transfer_amount': self.min_transfer_amount,
                'recipient_ids': self.recipient_ids,  # 保留用于兼容
                'level_recipients': {
                    '1': self.level_recipients[1],
                    '2': self.level_recipients[2],
                    '3': self.level_recipients[3]
                },
                'enabled': self.enabled,
                'multi_level_enabled': self.multi_level_enabled,
                'max_transfer_level': self.max_transfer_level,
                'use_user_manager_recipients': self.use_user_manager_recipients
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存转账配置失败: {e}")
    
    def add_recipient(self, user_id: str, level: int = 1):
        """添加收款账号
        
        Args:
            user_id: 用户ID
            level: 收款级别（1-3）
        """
        if level < 1 or level > 3:
            level = 1
        
        if user_id and user_id not in self.level_recipients[level]:
            self.level_recipients[level].append(user_id)
            # 保持recipient_ids与level_recipients[1]同步
            if level == 1:
                self.recipient_ids = self.level_recipients[1]
            self.save()
    
    def remove_recipient(self, user_id: str, level: int = 1):
        """移除收款账号
        
        Args:
            user_id: 用户ID
            level: 收款级别（1-3）
        """
        if level < 1 or level > 3:
            level = 1
            
        if user_id in self.level_recipients[level]:
            self.level_recipients[level].remove(user_id)
            # 保持recipient_ids与level_recipients[1]同步
            if level == 1:
                self.recipient_ids = self.level_recipients[1]
            self.save()
    
    def get_recipients(self, level: int = 1) -> List[str]:
        """获取指定级别的收款账号列表
        
        Args:
            level: 收款级别（1-3）
            
        Returns:
            收款账号ID列表
        """
        if level < 1 or level > 3:
            level = 1
        return self.level_recipients.get(level, [])
    
    def set_min_balance(self, amount: float):
        """设置最小保留余额"""
        self.min_balance = max(0.0, amount)
        self.save()
    
    def set_enabled(self, enabled: bool):
        """设置是否启用"""
        self.enabled = enabled
        self.save()
    
    def should_transfer(self, user_id: str, balance: float, current_level: int = 0) -> bool:
        """判断是否应该转账
        
        Args:
            user_id: 用户ID
            balance: 当前余额
            current_level: 当前转账级别（0表示初始账号，1-3表示收款账号级别）
            
        Returns:
            bool: 是否应该转账
        """
        # 如果未启用，不转账
        if not self.enabled:
            return False
        
        # 如果是初始账号（level=0）
        if current_level == 0:
            # 如果是收款账号，根据多级转账设置决定
            for level in [1, 2, 3]:
                if user_id in self.get_recipients(level):
                    # 如果启用了多级转账，且当前级别小于最大级别，则继续转账
                    if self.multi_level_enabled and level < self.max_transfer_level:
                        break  # 继续检查余额
                    else:
                        return False  # 不转账
            
            # 如果没有收款账号，不转账
            if not self.recipient_ids:
                return False
        else:
            # 如果是收款账号（level>0）
            # 检查是否启用了多级转账
            if not self.multi_level_enabled:
                return False
            
            # 检查是否达到最大级别
            if current_level >= self.max_transfer_level:
                return False
            
            # 检查下一级是否有收款账号
            next_level = current_level + 1
            if not self.get_recipients(next_level):
                return False
        
        # 如果余额未达到（起步金额 + 保留余额），不转账
        # 例如：起步金额30元，保留余额5元，则余额需要>=35元才转账
        min_required_balance = self.min_transfer_amount + self.min_balance
        if balance < min_required_balance:
            return False
        
        return True
    
    def get_transfer_recipient(self, user_id: str, current_level: int = 0) -> Optional[str]:
        """获取转账收款人
        
        Args:
            user_id: 当前用户ID
            current_level: 当前转账级别（0表示初始账号，1-3表示收款账号级别）
            
        Returns:
            str: 收款用户ID，如果没有则返回None
        """
        if current_level == 0:
            # 初始账号：查找该账号属于哪个级别，然后转给下一级
            for level in [1, 2, 3]:
                if user_id in self.get_recipients(level):
                    # 找到了，转给下一级
                    next_level = level + 1
                    if next_level <= 3:
                        recipients = self.get_recipients(next_level)
                        if recipients:
                            return recipients[0]  # 返回第一个收款账号
                    return None
            
            # 不是收款账号，转给1级收款账号
            if self.recipient_ids:
                return self.recipient_ids[0]
        else:
            # 收款账号：转给下一级
            next_level = current_level + 1
            if next_level <= 3:
                recipients = self.get_recipients(next_level)
                if recipients:
                    return recipients[0]
        
        return None
    
    def get_account_level(self, user_id: str) -> int:
        """获取账号的级别
        
        Args:
            user_id: 用户ID
            
        Returns:
            int: 级别（0表示普通账号，1-3表示收款账号级别）
        """
        for level in [1, 2, 3]:
            if user_id in self.get_recipients(level):
                return level
        return 0
    
    def get_transfer_amount(self, balance: float) -> float:
        """计算转账金额
        
        Args:
            balance: 当前余额
            
        Returns:
            float: 转账金额
        """
        if balance <= self.min_balance:
            return 0.0
        return balance - self.min_balance
    
    def get_next_recipient(self, current_index: int = 0) -> Optional[str]:
        """获取下一个收款账号（轮询分配）
        
        Args:
            current_index: 当前索引
            
        Returns:
            str: 收款用户ID，如果没有则返回None
        """
        if not self.recipient_ids:
            return None
        
        index = current_index % len(self.recipient_ids)
        return self.recipient_ids[index]
    
    def get_transfer_recipient_from_user_manager(
        self, 
        phone: str, 
        selector: 'RecipientSelector'
    ) -> Optional[str]:
        """从用户管理获取收款人（新方法）
        
        优先从用户管理读取收款人配置，支持多收款人轮询/随机选择。
        
        Args:
            phone: 发送人手机号
            selector: 收款人选择器
            
        Returns:
            收款人手机号，如果没有返回None
        """
        try:
            from .user_manager import UserManager
            manager = UserManager()
            
            # 1. 获取账号的管理员
            user = manager.get_account_user(phone)
            if not user or not user.enabled:
                return None
            
            # 2. 获取收款人列表
            recipients = user.transfer_recipients
            if not recipients:
                return None
            
            # 3. 使用选择器选择一个收款人
            selected = selector.select_recipient(
                recipients, 
                sender_phone=phone,
                key=user.user_id  # 使用用户ID作为轮询键
            )
            
            return selected
            
        except Exception as e:
            print(f"从用户管理获取收款人失败: {e}")
            return None
    
    def get_transfer_recipient_enhanced(
        self, 
        phone: str,
        user_id: str, 
        current_level: int = 0,
        selector: 'RecipientSelector' = None
    ) -> Optional[str]:
        """获取转账收款人（增强版）
        
        优先从用户管理读取，失败则降级到转账配置。
        支持多收款人轮询/随机选择。
        
        Args:
            phone: 发送人手机号
            user_id: 发送人用户ID
            current_level: 当前转账级别（0表示初始账号，1-3表示收款账号级别）
            selector: 收款人选择器（可选）
            
        Returns:
            收款人手机号或用户ID
        """
        # 检查是否启用用户管理收款人配置
        if not self.use_user_manager_recipients:
            print(f"  [转账配置] 用户管理收款人配置已禁用，使用转账配置")
            return self.get_transfer_recipient(user_id, current_level)
        
        # 1. 尝试从用户管理获取（只对初始账号有效）
        if current_level == 0 and selector:
            recipient = self.get_transfer_recipient_from_user_manager(
                phone, 
                selector
            )
            if recipient:
                return recipient
            else:
                # 降级日志
                print(f"  [转账配置] 用户管理未配置收款人，降级到转账配置")
        
        # 2. 降级到转账配置（原有逻辑）
        return self.get_transfer_recipient(user_id, current_level)


# 全局配置实例
_transfer_config = None


def get_transfer_config() -> TransferConfig:
    """获取转账配置实例"""
    global _transfer_config
    if _transfer_config is None:
        _transfer_config = TransferConfig()
    return _transfer_config
