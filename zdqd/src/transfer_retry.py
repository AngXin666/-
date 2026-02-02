"""
转账重试机制
Transfer Retry Mechanism
"""

import asyncio
from typing import Dict, Callable, Optional


class TransferRetry:
    """转账重试管理器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 5.0):
        """初始化重试管理器
        
        Args:
            max_retries: 最大重试次数（默认3次）
            retry_delay: 重试延迟（秒，默认5秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def transfer_with_retry(
        self,
        transfer_func: Callable,
        device_id: str,
        recipient_id: str,
        log_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict:
        """带重试的转账执行
        
        Args:
            transfer_func: 转账函数
            device_id: 设备ID
            recipient_id: 收款人ID
            log_callback: 日志回调函数
            **kwargs: 其他参数
            
        Returns:
            转账结果字典
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        last_result = None
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    log(f"  [重试] 第 {attempt + 1}/{self.max_retries} 次尝试...")
                    await asyncio.sleep(self.retry_delay)
                
                # 执行转账
                result = await transfer_func(
                    device_id,
                    recipient_id,
                    log_callback=log_callback,
                    **kwargs
                )
                
                # 如果成功，直接返回
                if result.get('success'):
                    if attempt > 0:
                        log(f"  [重试] ✓ 重试成功")
                    return result
                
                # 记录失败结果
                last_result = result
                
                # 如果是最后一次尝试，不再重试
                if attempt == self.max_retries - 1:
                    log(f"  [重试] ❌ 已达到最大重试次数 ({self.max_retries})")
                    break
                
                # 分析失败原因，决定是否继续重试
                error_type = result.get('error_type')
                if error_type and not self._should_retry(error_type):
                    log(f"  [重试] ⚠️ 错误类型不适合重试: {error_type}")
                    break
                
                log(f"  [重试] ⚠️ 转账失败: {result.get('message', '未知错误')}")
                
            except Exception as e:
                # 检查异常中是否包含错误类型
                error_type = None
                if hasattr(e, 'error_type'):
                    error_type = e.error_type
                elif isinstance(e, str) and 'ACCOUNT_FROZEN' in str(e):
                    # 从异常消息中提取错误类型
                    try:
                        from .models.error_types import ErrorType
                        error_type = ErrorType.ACCOUNT_FROZEN
                    except:
                        pass
                
                log(f"  [重试] ❌ 转账异常: {e}")
                last_result = {
                    'success': False,
                    'message': f"转账异常: {e}",
                    'error_type': error_type,
                    'amount': 0.0,
                    'chain': []
                }
                
                # 如果是最后一次尝试，不再重试
                if attempt == self.max_retries - 1:
                    break
                
                # 检查是否应该重试
                if error_type and not self._should_retry(error_type):
                    log(f"  [重试] ⚠️ 错误类型不适合重试: {error_type}")
                    break
        
        # 所有重试都失败，返回最后一次的结果
        return last_result or {
            'success': False,
            'message': "转账失败",
            'amount': 0.0,
            'chain': []
        }
    
    def _should_retry(self, error_type) -> bool:
        """判断错误类型是否应该重试
        
        Args:
            error_type: 错误类型
            
        Returns:
            是否应该重试
        """
        # 导入错误类型
        try:
            from .models.error_types import ErrorType
        except ImportError:
            from models.error_types import ErrorType
        
        # 以下错误类型不应该重试
        no_retry_errors = [
            ErrorType.ACCOUNT_FROZEN,  # 账号冻结
            ErrorType.INSUFFICIENT_BALANCE,  # 余额不足
            ErrorType.INVALID_RECIPIENT,  # 无效收款人
        ]
        
        return error_type not in no_retry_errors


# 全局实例
_transfer_retry = None


def get_transfer_retry(max_retries: int = 3, retry_delay: float = 5.0) -> TransferRetry:
    """获取转账重试管理器实例
    
    Args:
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        
    Returns:
        TransferRetry实例
    """
    global _transfer_retry
    if _transfer_retry is None:
        _transfer_retry = TransferRetry(max_retries, retry_delay)
    return _transfer_retry
