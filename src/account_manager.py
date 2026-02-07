"""
账号管理模块
Account Manager Module
"""

import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from .models import Account, AccountResult, AccountStatus
from .local_db import LocalDatabase


class AccountManager:
    """账号管理器"""
    
    # 中国手机号正则（11位，以1开头）
    PHONE_PATTERN = re.compile(r'^1[3-9]\d{9}$')
    
    # 账号密码分隔符
    SEPARATOR = "----"
    
    def __init__(self, accounts_file: str):
        """初始化账号管理器
        
        Args:
            accounts_file: 账号文件路径（文本格式：账号----密码）
        """
        self.accounts_file = accounts_file
        self.accounts: List[Account] = []
        self.results: Dict[str, AccountResult] = {}
        self._current_index = 0
        self.db = LocalDatabase()  # 初始化数据库
    
    def validate_phone(self, phone: str) -> bool:
        """验证手机号格式
        
        Args:
            phone: 手机号
            
        Returns:
            是否有效
        """
        if not phone:
            return False
        # 移除可能的空格和横线
        phone = phone.replace(" ", "").replace("-", "")
        return bool(self.PHONE_PATTERN.match(phone))
    
    def load_accounts(self) -> List[Account]:
        """从文本文件加载账号列表（支持加密文件）
        
        文件格式：每行一个账号，格式为 账号----密码
        支持加密文件（.enc后缀）和明文文件
        
        Returns:
            账号列表
        """
        self.accounts = []
        
        # 从数据库加载历史记录中的 nickname 和 user_id
        history_data = {}
        try:
            records = self.db.get_all_history_records()
            for record in records:
                phone = record.get('phone')
                nickname = record.get('nickname')
                user_id = record.get('user_id')
                if phone and nickname and user_id:
                    history_data[phone] = {
                        'nickname': nickname,
                        'user_id': user_id
                    }
        except Exception as e:
            print(f"[数据库] 加载历史记录失败: {e}")
            pass  # 如果读取失败，继续使用空的history_data
        
        # 使用加密账号文件管理器读取账号
        try:
            from .encrypted_accounts_file import EncryptedAccountsFile
        except ImportError:
            try:
                from encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                from src.encrypted_accounts_file import EncryptedAccountsFile
        
        encrypted_file = EncryptedAccountsFile(self.accounts_file)
        
        # 读取账号列表（自动处理加密/明文文件）
        try:
            accounts_list = encrypted_file.read_accounts()
            
            for phone, password in accounts_list:
                # 验证手机号格式
                if not self.validate_phone(phone):
                    continue
                
                # 从历史记录中获取nickname和user_id（如果有）
                hist = history_data.get(phone, {})
                
                account = Account(
                    phone=phone,
                    password=password,
                    status=AccountStatus.PENDING,
                    nickname=hist.get('nickname'),  # 从历史记录中获取
                    user_id=hist.get('user_id')  # 从历史记录中获取
                )
                self.accounts.append(account)
        except Exception as e:
            print(f"[账号管理器] 加载账号失败: {e}")
            pass
        
        self._current_index = 0
        return self.accounts
    
    def get_next_account(self) -> Optional[Account]:
        """获取下一个待处理账号
        
        Returns:
            下一个待处理账号，没有则返回 None
        """
        while self._current_index < len(self.accounts):
            account = self.accounts[self._current_index]
            if account.status == AccountStatus.PENDING:
                account.status = AccountStatus.PROCESSING
                self._current_index += 1
                return account
            self._current_index += 1
        return None
    
    def get_pending_accounts(self) -> List[Account]:
        """获取所有待处理账号
        
        Returns:
            待处理账号列表
        """
        return [a for a in self.accounts if a.status == AccountStatus.PENDING]

    def update_account_result(self, phone: str, result: AccountResult) -> None:
        """更新账号处理结果
        
        Args:
            phone: 手机号
            result: 处理结果
        """
        self.results[phone] = result
        
        # 更新账号状态
        for account in self.accounts:
            if account.phone == phone:
                account.status = AccountStatus.COMPLETED if result.success else AccountStatus.FAILED
                break
        
        # 自动保存到数据库
        self._save_result_to_db(result)
    
    def get_account_result(self, phone: str) -> Optional[AccountResult]:
        """获取账号处理结果
        
        Args:
            phone: 手机号
            
        Returns:
            处理结果
        """
        return self.results.get(phone)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息
        
        Returns:
            统计信息字典
        """
        total = len(self.accounts)
        completed = sum(1 for a in self.accounts if a.status == AccountStatus.COMPLETED)
        failed = sum(1 for a in self.accounts if a.status == AccountStatus.FAILED)
        pending = sum(1 for a in self.accounts if a.status == AccountStatus.PENDING)
        processing = sum(1 for a in self.accounts if a.status == AccountStatus.PROCESSING)
        
        # 计算成功率
        success_rate = (completed / total * 100) if total > 0 else 0
        
        # 计算总抽奖金额
        total_draw_amount = sum(
            r.draw_result.total_amount 
            for r in self.results.values() 
            if r.draw_result and r.draw_result.success
        )
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "processing": processing,
            "success_rate": success_rate,
            "total_draw_amount": total_draw_amount
        }
    
    def generate_report(self, output_path: str) -> bool:
        """生成汇总报告
        
        Args:
            output_path: 报告输出路径
            
        Returns:
            是否成功生成
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # 写入标题
                f.write("=" * 80 + "\n")
                f.write("自动签到助手报告\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                # 写入统计
                stats = self.get_statistics()
                f.write("【汇总统计】\n")
                f.write(f"总计账号: {stats['total']} 个\n")
                f.write(f"成功: {stats['completed']} 个\n")
                f.write(f"失败: {stats['failed']} 个\n")
                f.write(f"成功率: {stats['success_rate']:.1f}%\n")
                f.write(f"总抽奖金额: {stats['total_draw_amount']:.2f} 元\n")
                f.write("\n" + "-" * 80 + "\n\n")
                
                # 写入详细结果
                f.write("【详细结果】\n\n")
                for phone, result in self.results.items():
                    f.write(f"手机号: {phone}\n")
                    f.write(f"  处理状态: {'✓ 成功' if result.success else '✗ 失败'}\n")
                    f.write(f"  处理时间: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("\n")
                    
                    # 余额信息
                    f.write("  【余额信息】\n")
                    if result.balance_before is not None:
                        f.write(f"    登录前余额: {result.balance_before:.2f} 元\n")
                    else:
                        f.write(f"    登录前余额: 未获取\n")
                    
                    if result.balance_after is not None:
                        f.write(f"    登录后余额: {result.balance_after:.2f} 元\n")
                        if result.balance_before is not None:
                            diff = result.balance_after - result.balance_before
                            if diff > 0:
                                f.write(f"    余额变化: +{diff:.2f} 元 ↑\n")
                            elif diff < 0:
                                f.write(f"    余额变化: {diff:.2f} 元 ↓\n")
                            else:
                                f.write(f"    余额变化: 无变化\n")
                    else:
                        f.write(f"    登录后余额: 未获取\n")
                    
                    # 积分和抵扣券
                    if result.points is not None:
                        f.write(f"    积分: {result.points}\n")
                    if result.vouchers is not None:
                        f.write(f"    抵扣券: {result.vouchers} 张\n")
                    if result.total_draw_times is not None:
                        f.write(f"    总抽奖次数: {result.total_draw_times} 次\n")
                    f.write("\n")
                    
                    # 签到信息
                    f.write("  【签到信息】\n")
                    if result.sign_in_result:
                        sign_in = result.sign_in_result
                        if sign_in.already_signed:
                            f.write(f"    状态: 今日已签到\n")
                        elif sign_in.success:
                            f.write(f"    状态: ✓ 签到成功\n")
                            if sign_in.reward_amount > 0:
                                f.write(f"    奖励金额: {sign_in.reward_amount:.2f} 元\n")
                            if sign_in.screenshot_path:
                                f.write(f"    截图路径: {sign_in.screenshot_path}\n")
                        else:
                            f.write(f"    状态: ✗ 签到失败\n")
                            if sign_in.error_message:
                                f.write(f"    错误信息: {sign_in.error_message}\n")
                    else:
                        f.write(f"    状态: 未执行\n")
                    f.write("\n")
                    
                    # 抽奖信息
                    f.write("  【抽奖信息】\n")
                    if result.draw_result:
                        draw = result.draw_result
                        if draw.success:
                            f.write(f"    状态: ✓ 抽奖完成\n")
                            f.write(f"    抽奖次数: {draw.draw_count} 次\n")
                            if draw.draw_count > 0:
                                f.write(f"    抽中总金额: {draw.total_amount:.2f} 元\n")
                                if draw.amounts:
                                    f.write(f"    每次金额: {', '.join([f'{a:.2f}' for a in draw.amounts])} 元\n")
                        else:
                            f.write(f"    状态: ✗ 抽奖失败\n")
                            if draw.error_message:
                                f.write(f"    错误信息: {draw.error_message}\n")
                    else:
                        f.write(f"    状态: 未执行\n")
                    f.write("\n")
                    
                    # 错误信息
                    if result.error_message:
                        f.write(f"  【错误信息】\n")
                        f.write(f"    {result.error_message}\n")
                        f.write("\n")
                    
                    f.write("-" * 80 + "\n\n")
            
            return True
        except Exception as e:
            print(f"生成报告失败: {e}")
            return False
    
    def _get_sign_in_status(self, result: AccountResult) -> str:
        """获取签到状态描述"""
        if not result.sign_in_result:
            return '未执行'
        if result.sign_in_result.already_signed:
            return '已签到'
        if result.sign_in_result.success:
            return '签到成功'
        return '签到失败'
    

    def _save_result_to_db(self, result: AccountResult) -> None:
        """保存处理结果到数据库（只保存已完成的操作）
        
        只保存状态为"成功"或"失败"的记录，跳过"待处理"状态。
        使用 UPSERT 逻辑，每个账号每天只有一条记录。
        
        数据完整性检查：
        - 如果昵称或用户ID为空/None/'-'/'N/A'，尝试从历史记录或缓存中获取
        - 自动获取账号的管理员信息
        
        Args:
            result: 账号处理结果
        """
        try:
            # 状态验证：只保存成功或失败的记录
            status = '成功' if result.success else '失败'
            
            # ===== 数据完整性检查和自动修复 =====
            nickname = result.nickname
            user_id = result.user_id
            
            # 检查昵称和用户ID是否有效
            invalid_values = [None, '', '-', 'N/A', '待处理']
            nickname_invalid = nickname in invalid_values
            user_id_invalid = user_id in invalid_values
            
            if nickname_invalid or user_id_invalid:
                print(f"[数据库] ⚠️ 检测到无效数据 - 昵称: '{nickname}', 用户ID: '{user_id}'")
                print(f"[数据库] 尝试从历史记录中获取正确值...")
                
                # 策略1: 从历史记录中获取最近的正常值
                latest_record = self.db.get_latest_record_by_phone(result.phone)
                if latest_record:
                    if nickname_invalid and latest_record.get('nickname') not in invalid_values:
                        nickname = latest_record.get('nickname')
                        print(f"[数据库] ✓ 从历史记录获取昵称: {nickname}")
                    
                    if user_id_invalid and latest_record.get('user_id') not in invalid_values:
                        user_id = latest_record.get('user_id')
                        print(f"[数据库] ✓ 从历史记录获取用户ID: {user_id}")
                
                # 策略2: 从 phone_userid_mapping.txt 中获取用户ID
                if user_id_invalid:
                    try:
                        from pathlib import Path
                        mapping_file = Path("login_cache") / "phone_userid_mapping.txt"
                        if mapping_file.exists():
                            with open(mapping_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    line = line.strip()
                                    if '=' in line:
                                        phone, mapped_user_id = line.split('=', 1)
                                        if phone == result.phone:
                                            user_id = mapped_user_id
                                            print(f"[数据库] ✓ 从映射文件获取用户ID: {user_id}")
                                            break
                    except Exception as e:
                        print(f"[数据库] ⚠️ 读取映射文件失败: {e}")
                
                # 如果仍然无效，设置为"待更新"
                if nickname_invalid and nickname in invalid_values:
                    nickname = "待更新"
                    print(f"[数据库] ⚠️ 无法获取昵称，设置为: {nickname}")
                
                if user_id_invalid and user_id in invalid_values:
                    user_id = "待更新"
                    print(f"[数据库] ⚠️ 无法获取用户ID，设置为: {user_id}")
            
            # ===== 获取管理员信息 =====
            owner = None
            try:
                from .user_manager import UserManager
                manager = UserManager()
                user = manager.get_account_user(result.phone)
                if user:
                    owner = user.user_name
                    print(f"[数据库] ✓ 管理员: {owner}")
            except Exception as e:
                print(f"[数据库] ⚠️ 获取管理员失败: {e}")
            
            # 构建数据库记录
            record = {
                'phone': result.phone,
                'nickname': nickname,  # 使用修复后的值
                'user_id': user_id,    # 使用修复后的值
                'balance_before': result.balance_before,
                'points': result.points,
                'vouchers': result.vouchers,
                'coupons': result.coupons,
                'checkin_reward': result.checkin_reward,
                'checkin_total_times': result.checkin_total_times,
                'checkin_balance_after': result.checkin_balance_after,
                'balance_after': result.balance_after,
                'duration': result.duration,
                'status': status,
                'login_method': result.login_method,
                'run_date': datetime.now().strftime('%Y-%m-%d'),
                'transfer_amount': result.transfer_amount,  # 转账金额
                'transfer_recipient': result.transfer_recipient,  # 收款人
                'owner': owner  # 管理员
            }
            
            # 调用 UPSERT 方法
            if self.db.upsert_history_record(record):
                print(f"[数据库] ✅ 已保存 {result.phone} 的记录")
            else:
                print(f"[数据库] ⚠️ 保存 {result.phone} 的记录失败")
                
        except Exception as e:
            print(f"[数据库] ❌ 保存记录时出错: {e}")
    
    def reset(self) -> None:
        """重置所有账号状态"""
        for account in self.accounts:
            account.status = AccountStatus.PENDING
        self.results.clear()
        self._current_index = 0
