"""
用户管理模块
User Management Module

功能：
1. 管理多个用户信息
2. 为每个用户配置转账收款人
3. 将账号分配给指定用户
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class User:
    """用户信息"""
    user_id: str                      # 用户ID（唯一标识）
    user_name: str                    # 用户名称
    transfer_recipients: List[str]    # 转账收款人手机号列表（支持多个收款账号）
    description: str = ""             # 备注说明
    enabled: bool = True              # 是否启用
    
    # 兼容旧版本的单个收款人字段
    @property
    def transfer_recipient(self) -> str:
        """获取第一个收款人（兼容旧代码）"""
        return self.transfer_recipients[0] if self.transfer_recipients else ""
    
    @transfer_recipient.setter
    def transfer_recipient(self, value: str):
        """设置收款人（兼容旧代码）"""
        if value:
            self.transfer_recipients = [value]
        else:
            self.transfer_recipients = []


class UserManager:
    """用户管理器"""
    
    def __init__(self, config_dir: Path = None):
        """初始化用户管理器
        
        Args:
            config_dir: 配置文件目录，默认为 runtime_data/
        """
        if config_dir is None:
            config_dir = Path("runtime_data")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 用户配置文件
        self.users_file = self.config_dir / "users.json"
        
        # 账号-用户映射文件
        self.account_mapping_file = self.config_dir / "account_user_mapping.json"
        
        # 加载数据
        self.users: Dict[str, User] = self._load_users()
        self.account_mapping: Dict[str, str] = self._load_account_mapping()
    
    def _load_users(self) -> Dict[str, User]:
        """加载用户列表（兼容旧格式）"""
        if not self.users_file.exists():
            return {}
        
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            users = {}
            for user_id, user_dict in data.items():
                # 兼容旧格式：如果是单个收款人字符串，转换为列表
                if 'transfer_recipient' in user_dict and 'transfer_recipients' not in user_dict:
                    # 旧格式：单个收款人
                    recipient = user_dict.pop('transfer_recipient')
                    user_dict['transfer_recipients'] = [recipient] if recipient else []
                elif 'transfer_recipients' not in user_dict:
                    # 没有收款人字段，设置为空列表
                    user_dict['transfer_recipients'] = []
                
                users[user_id] = User(**user_dict)
            
            return users
        except Exception as e:
            print(f"加载用户列表失败: {e}")
            return {}
    
    def _save_users(self):
        """保存用户列表"""
        try:
            data = {}
            for user_id, user in self.users.items():
                data[user_id] = asdict(user)
            
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户列表失败: {e}")
    
    def _load_account_mapping(self) -> Dict[str, str]:
        """加载账号-用户映射（优先从数据库加载，然后合并JSON文件）"""
        mapping = {}
        
        # 1. 从JSON文件加载（兼容旧数据）
        if self.account_mapping_file.exists():
            try:
                with open(self.account_mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
            except Exception as e:
                print(f"加载账号映射失败: {e}")
        
        # 2. 从数据库加载（覆盖JSON中的数据）
        try:
            from .local_db import LocalDatabase
            db = LocalDatabase()
            
            # 获取所有有管理员的账号
            db_mapping = db.get_all_account_owners()
            
            # 将数据库中的管理员名称转换为用户ID
            for phone, owner_name in db_mapping.items():
                # 查找对应的用户ID
                user_id = self._find_user_id_by_name(owner_name)
                if user_id:
                    mapping[phone] = user_id
            
            if db_mapping:
                print(f"✓ 从数据库加载了 {len(db_mapping)} 个账号的管理员信息")
        except Exception as e:
            print(f"⚠️ 从数据库加载账号映射失败: {e}")
        
        return mapping
    
    def _find_user_id_by_name(self, user_name: str) -> Optional[str]:
        """根据用户名查找用户ID
        
        Args:
            user_name: 用户名称
            
        Returns:
            用户ID，如果未找到返回None
        """
        for user_id, user in self.users.items():
            if user.user_name == user_name:
                return user_id
        return None
    
    def _save_account_mapping(self):
        """保存账号-用户映射"""
        try:
            with open(self.account_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.account_mapping, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存账号映射失败: {e}")
    
    def add_user(self, user: User) -> bool:
        """添加用户
        
        Args:
            user: 用户对象
            
        Returns:
            是否成功
        """
        if user.user_id in self.users:
            print(f"用户ID已存在: {user.user_id}")
            return False
        
        self.users[user.user_id] = user
        self._save_users()
        return True
    
    def update_user(self, user: User) -> bool:
        """更新用户信息
        
        Args:
            user: 用户对象
            
        Returns:
            是否成功
        """
        if user.user_id not in self.users:
            print(f"用户不存在: {user.user_id}")
            return False
        
        self.users[user.user_id] = user
        self._save_users()
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """删除管理员（只删除管理员角色，不删除数据库中的账号数据）
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否成功
        """
        if user_id not in self.users:
            print(f"用户不存在: {user_id}")
            return False
        
        # 获取该管理员的所有账号
        accounts_to_remove = [phone for phone, uid in self.account_mapping.items() if uid == user_id]
        
        # 删除管理员角色
        del self.users[user_id]
        self._save_users()
        
        # 清除账号分配关系
        for phone in accounts_to_remove:
            del self.account_mapping[phone]
        
        if accounts_to_remove:
            self._save_account_mapping()
            
            # 同步清除数据库中这些账号的管理员字段
            try:
                from .local_db import LocalDatabase
                db = LocalDatabase()
                # 将这些账号的 owner 字段设置为 None（清空管理员）
                db.batch_update_account_owner(accounts_to_remove, None)
                print(f"✓ 已清除数据库中 {len(accounts_to_remove)} 个账号的管理员")
            except Exception as e:
                print(f"⚠️ 清除数据库管理员失败: {e}")
        
        return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户对象，如果不存在返回None
        """
        return self.users.get(user_id)
    
    def get_all_users(self) -> List[User]:
        """获取所有用户
        
        Returns:
            用户列表
        """
        return list(self.users.values())
    
    def assign_account(self, phone: str, user_id: str) -> bool:
        """将账号分配给用户
        
        Args:
            phone: 手机号
            user_id: 用户ID
            
        Returns:
            是否成功
        """
        if user_id not in self.users:
            print(f"用户不存在: {user_id}")
            return False
        
        self.account_mapping[phone] = user_id
        self._save_account_mapping()
        return True
    
    def batch_assign_accounts(self, phones: List[str], user_id: str) -> int:
        """批量分配账号给用户
        
        Args:
            phones: 手机号列表
            user_id: 用户ID
            
        Returns:
            成功分配的数量
        """
        if user_id not in self.users:
            print(f"用户不存在: {user_id}")
            return 0
        
        count = 0
        for phone in phones:
            self.account_mapping[phone] = user_id
            count += 1
        
        self._save_account_mapping()
        
        # 同步更新数据库中的管理员信息
        user = self.users[user_id]
        owner_name = user.user_name
        
        try:
            from .local_db import LocalDatabase
            db = LocalDatabase()
            db.batch_update_account_owner(phones, owner_name)
            print(f"✓ 已同步更新数据库中 {count} 个账号的管理员")
        except Exception as e:
            print(f"⚠️ 同步更新数据库管理员失败: {e}")
        
        return count
    
    def unassign_account(self, phone: str) -> bool:
        """取消账号分配
        
        Args:
            phone: 手机号
            
        Returns:
            是否成功
        """
        if phone not in self.account_mapping:
            return False
        
        del self.account_mapping[phone]
        self._save_account_mapping()
        return True
    
    def get_account_user(self, phone: str) -> Optional[User]:
        """获取账号所属的用户
        
        Args:
            phone: 手机号
            
        Returns:
            用户对象，如果未分配返回None
        """
        user_id = self.account_mapping.get(phone)
        if user_id:
            return self.users.get(user_id)
        return None
    
    def get_user_accounts(self, user_id: str) -> List[str]:
        """获取用户的所有账号
        
        Args:
            user_id: 用户ID
            
        Returns:
            手机号列表
        """
        return [phone for phone, uid in self.account_mapping.items() if uid == user_id]
    
    def get_transfer_recipient(self, phone: str) -> Optional[str]:
        """获取账号的转账收款人
        
        Args:
            phone: 手机号
            
        Returns:
            转账收款人手机号，如果未配置返回None
        """
        user = self.get_account_user(phone)
        if user and user.enabled:
            return user.transfer_recipient
        return None
    
    def get_unassigned_accounts(self, all_phones: List[str]) -> List[str]:
        """获取未分配的账号列表
        
        Args:
            all_phones: 所有手机号列表
            
        Returns:
            未分配的手机号列表
        """
        return [phone for phone in all_phones if phone not in self.account_mapping]
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        total_users = len(self.users)
        enabled_users = sum(1 for user in self.users.values() if user.enabled)
        total_accounts = len(self.account_mapping)
        
        # 按用户统计账号数
        user_account_counts = {}
        for user_id in self.users.keys():
            user_account_counts[user_id] = len(self.get_user_accounts(user_id))
        
        return {
            'total_users': total_users,
            'enabled_users': enabled_users,
            'total_accounts': total_accounts,
            'user_account_counts': user_account_counts
        }
    
    def remove_account_from_all_users(self, phone: str) -> bool:
        """从所有用户中移除账号分配
        
        Args:
            phone: 手机号
            
        Returns:
            是否成功
        """
        if phone in self.account_mapping:
            user_id = self.account_mapping[phone]
            
            # 从映射中删除
            del self.account_mapping[phone]
            self._save_account_mapping()
            
            # 从数据库中删除
            try:
                from .local_db import LocalDatabase
                db = LocalDatabase()
                conn = db._get_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE history_records SET owner = NULL WHERE phone = ?", (phone,))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"从数据库移除账号分配失败 ({phone}): {e}")
            
            return True
        
        return False
    
    def clear_all_account_assignments(self) -> bool:
        """清空所有账号分配
        
        Returns:
            是否成功
        """
        try:
            # 清空映射
            self.account_mapping = {}
            self._save_account_mapping()
            
            # 从数据库中清空
            try:
                from .local_db import LocalDatabase
                db = LocalDatabase()
                conn = db._get_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE history_records SET owner = NULL")
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"从数据库清空账号分配失败: {e}")
            
            return True
        except Exception as e:
            print(f"清空账号分配失败: {e}")
            return False


def test_user_manager():
    """测试用户管理器"""
    print("=" * 60)
    print("用户管理器测试")
    print("=" * 60)
    
    # 创建管理器
    manager = UserManager()
    
    # 添加用户
    user1 = User(
        user_id="user001",
        user_name="张三",
        transfer_recipient="13800138000",
        description="第一批账号"
    )
    
    user2 = User(
        user_id="user002",
        user_name="李四",
        transfer_recipient="13900139000",
        description="第二批账号"
    )
    
    print("\n添加用户...")
    manager.add_user(user1)
    manager.add_user(user2)
    print(f"✓ 已添加 {len(manager.get_all_users())} 个用户")
    
    # 分配账号
    print("\n分配账号...")
    manager.assign_account("18888888888", "user001")
    manager.assign_account("17777777777", "user001")
    manager.assign_account("16666666666", "user002")
    print("✓ 账号分配完成")
    
    # 查询
    print("\n查询用户账号...")
    for user in manager.get_all_users():
        accounts = manager.get_user_accounts(user.user_id)
        print(f"  {user.user_name}: {len(accounts)} 个账号")
        for phone in accounts:
            print(f"    - {phone}")
    
    # 获取转账收款人
    print("\n获取转账收款人...")
    phone = "18888888888"
    recipient = manager.get_transfer_recipient(phone)
    print(f"  {phone} → {recipient}")
    
    # 统计信息
    print("\n统计信息...")
    stats = manager.get_statistics()
    print(f"  总用户数: {stats['total_users']}")
    print(f"  启用用户数: {stats['enabled_users']}")
    print(f"  总账号数: {stats['total_accounts']}")
    
    print("\n✅ 测试完成")


if __name__ == '__main__':
    test_user_manager()
