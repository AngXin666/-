"""错误类型枚举模块

定义所有可能的错误类型常量，用于在GUI中显示用户友好的错误信息。
"""

from enum import Enum


class ErrorType(Enum):
    """错误类型枚举
    
    定义系统中所有可能的错误类型，每个错误类型对应一个用户友好的显示文本。
    """
    
    # 登录相关错误
    LOGIN_PHONE_NOT_EXIST = "login_phone_not_exist"
    LOGIN_PASSWORD_ERROR = "login_password_error"
    
    # 导航相关错误
    CANNOT_REACH_PROFILE = "cannot_reach_profile"
    CANNOT_READ_PROFILE = "cannot_read_profile"
    CANNOT_REACH_CHECKIN = "cannot_reach_checkin"
    
    # 签到相关错误
    CHECKIN_FAILED = "checkin_failed"
    CHECKIN_EXCEPTION = "checkin_exception"
    
    # 数据获取错误
    CANNOT_GET_FINAL_DATA = "cannot_get_final_data"
    
    # 转账相关错误
    TRANSFER_FAILED = "transfer_failed"
    
    @staticmethod
    def to_display_text(error_type: 'ErrorType') -> str:
        """将错误类型转换为用户友好的显示文本
        
        Args:
            error_type: 错误类型枚举值
            
        Returns:
            格式化的错误文本，格式为"阶段:具体原因"或"✅ 成功"
            
        Examples:
            >>> ErrorType.to_display_text(ErrorType.LOGIN_PHONE_NOT_EXIST)
            '登录失败:手机号不存在'
            
            >>> ErrorType.to_display_text(ErrorType.CHECKIN_FAILED)
            '失败:签到失败'
        """
        error_map = {
            # 登录相关错误
            ErrorType.LOGIN_PHONE_NOT_EXIST: "登录失败:手机号不存在",
            ErrorType.LOGIN_PASSWORD_ERROR: "登录失败:密码错误",
            
            # 导航相关错误
            ErrorType.CANNOT_REACH_PROFILE: "失败:无法到达个人页",
            ErrorType.CANNOT_READ_PROFILE: "失败:无法读取个人资料",
            ErrorType.CANNOT_REACH_CHECKIN: "失败:无法到达签到页",
            
            # 签到相关错误
            ErrorType.CHECKIN_FAILED: "失败:签到失败",
            ErrorType.CHECKIN_EXCEPTION: "失败:签到异常",
            
            # 数据获取错误
            ErrorType.CANNOT_GET_FINAL_DATA: "失败:获取最终资料失败",
            
            # 转账相关错误
            ErrorType.TRANSFER_FAILED: "失败:转账失败",
        }
        
        return error_map.get(error_type, "失败:未知错误")
