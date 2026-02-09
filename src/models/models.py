"""
数据模型定义
Data Models for Nox Emulator Automation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from .error_types import ErrorType


class InstanceStatus(Enum):
    """模拟器实例状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class AccountStatus(Enum):
    """账号处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(Enum):
    """UI 操作类型"""
    TAP = "tap"
    SWIPE = "swipe"
    INPUT = "input"
    WAIT = "wait"
    SCREENSHOT = "screenshot"


@dataclass
class EmulatorInstance:
    """模拟器实例信息"""
    name: str
    index: int
    status: InstanceStatus
    device_id: Optional[str] = None
    adb_port: Optional[int] = None


@dataclass
class Account:
    """账号信息"""
    phone: str
    password: str
    status: AccountStatus = AccountStatus.PENDING
    # 可选字段：如果GUI已经知道这些信息，可以直接传递，避免OCR识别
    nickname: Optional[str] = None  # 昵称
    user_id: Optional[str] = None  # 用户ID


@dataclass
class SignInResult:
    """签到结果"""
    success: bool
    already_signed: bool = False
    reward: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional['ErrorType'] = None  # 错误类型（ErrorType枚举）
    reward_amount: float = 0.0  # 奖励金额
    total_times: Optional[int] = None  # 签到总次数
    screenshot_path: Optional[str] = None  # 截图路径
    ocr_texts: List[str] = field(default_factory=list)  # OCR识别的文本


@dataclass
class DrawResult:
    """抽奖结果"""
    success: bool
    draw_count: int = 0
    total_amount: float = 0.0
    amounts: List[float] = field(default_factory=list)
    error_message: Optional[str] = None

    def __post_init__(self):
        """确保 total_amount 等于 amounts 列表的总和"""
        if self.amounts and self.total_amount == 0.0:
            self.total_amount = sum(self.amounts)


@dataclass
class AccountResult:
    """账号处理结果 - 增强版数据收集"""
    # 必填字段
    phone: str
    success: bool
    
    # 用户身份信息
    nickname: Optional[str] = None  # 昵称
    user_id: Optional[str] = None  # 用户ID
    
    # 财务数据（操作前）
    balance_before: Optional[float] = None  # 余额前
    points: Optional[int] = None  # 积分
    vouchers: Optional[float] = None  # 抵扣券数量/金额（保留小数精度）
    coupons: Optional[int] = None  # 优惠券数量
    
    # 签到数据
    checkin_reward: float = 0.0  # 签到总奖励（所有签到的累加）
    checkin_total_times: Optional[int] = None  # 签到总次数
    checkin_rewards_detail: List[float] = field(default_factory=list)  # 签到奖励明细列表
    checkin_balance_after: Optional[float] = None  # 签到后余额（签到完成后立即获取的余额）
    
    # 财务数据（操作后）
    balance_after: Optional[float] = None  # 余额（最终余额）
    
    # 遗留字段（保持兼容性）
    total_draw_times: Optional[int] = None  # 总抽奖次数
    sign_in_result: Optional[SignInResult] = None  # 签到结果（旧版）
    draw_result: Optional[DrawResult] = None  # 抽奖结果
    
    # 错误信息字段（新增）
    error_type: Optional[ErrorType] = None  # 错误类型（枚举）
    error_message: Optional[str] = None  # 详细错误信息
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 性能数据（新增）
    duration: Optional[float] = None  # 处理耗时（秒）
    login_method: Optional[str] = None  # 登录方式：'缓存' 或 '正常登录'
    
    # 转账数据（新增）
    transfer_amount: Optional[float] = None  # 转账金额
    transfer_recipient: Optional[str] = None  # 收款人ID
    
    @property
    def balance_change(self) -> Optional[float]:
        """余额变化 = 余额 - 余额前"""
        if self.balance_before is not None and self.balance_after is not None:
            return self.balance_after - self.balance_before
        return None


@dataclass
class Action:
    """UI 操作定义"""
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    delay_after: float = 0.5
    retry_count: int = 3
    timeout: int = 10


@dataclass
class AdSkipResult:
    """广告跳过结果"""
    skipped: bool
    method: str
    duration: float


@dataclass
class LoginResult:
    """登录结果"""
    success: bool
    error_message: Optional[str] = None


@dataclass
class Config:
    """配置信息"""
    nox_path: str = ""
    adb_path: str = ""
    emulator_type_selection: str = "自动检测"  # 新增：记住用户的模拟器类型选择
    target_app_package: str = ""
    target_app_keyword: str = "溪盟"
    target_app_activity: Optional[str] = None
    accounts_file: str = "./accounts.txt"
    max_concurrent_instances: int = 5
    emulator_start_timeout: int = 120
    app_start_timeout: int = 15
    element_wait_timeout: int = 10
    max_retry_count: int = 3
    retry_interval: int = 10
    screenshot_dir: str = "./screenshots"
    report_dir: str = "./reports"
    log_dir: str = "./logs"
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # 流程控制配置
    workflow_mode: str = "complete"  # 流程模式：complete, quick_checkin, login_only, transfer_only, custom
    workflow_enable_login: bool = True  # 是否启用登录流程
    workflow_enable_profile: bool = True  # 是否启用获取资料流程
    workflow_enable_checkin: bool = True  # 是否启用签到流程
    workflow_enable_transfer: bool = True  # 是否启用转账流程
    
    # 定时运行配置
    scheduled_run_enabled: bool = False  # 是否启用定时运行
    scheduled_run_time: str = "08:00"  # 定时运行时间

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "nox_path": self.nox_path,
            "adb_path": self.adb_path,
            "emulator_type_selection": self.emulator_type_selection,  # 新增
            "target_app_package": self.target_app_package,
            "target_app_keyword": self.target_app_keyword,
            "target_app_activity": self.target_app_activity,
            "accounts_file": self.accounts_file,
            "max_concurrent_instances": self.max_concurrent_instances,
            "emulator_start_timeout": self.emulator_start_timeout,
            "app_start_timeout": self.app_start_timeout,
            "element_wait_timeout": self.element_wait_timeout,
            "max_retry_count": self.max_retry_count,
            "retry_interval": self.retry_interval,
            "screenshot_dir": self.screenshot_dir,
            "report_dir": self.report_dir,
            "log_dir": self.log_dir,
            "log_level": self.log_level,
            "debug_mode": self.debug_mode,
            # 流程控制配置
            "workflow_mode": self.workflow_mode,
            "workflow_enable_login": self.workflow_enable_login,
            "workflow_enable_profile": self.workflow_enable_profile,
            "workflow_enable_checkin": self.workflow_enable_checkin,
            "workflow_enable_transfer": self.workflow_enable_transfer,
            # 定时运行配置
            "scheduled_run_enabled": self.scheduled_run_enabled,
            "scheduled_run_time": self.scheduled_run_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """从字典创建配置对象"""
        return cls(
            nox_path=data.get("nox_path", ""),
            adb_path=data.get("adb_path", ""),
            emulator_type_selection=data.get("emulator_type_selection", "自动检测"),  # 新增
            target_app_package=data.get("target_app_package", ""),
            target_app_keyword=data.get("target_app_keyword", "溪盟"),
            target_app_activity=data.get("target_app_activity"),
            accounts_file=data.get("accounts_file", "./accounts.csv"),
            max_concurrent_instances=data.get("max_concurrent_instances", 5),
            emulator_start_timeout=data.get("emulator_start_timeout", 120),
            app_start_timeout=data.get("app_start_timeout", 15),
            element_wait_timeout=data.get("element_wait_timeout", 10),
            max_retry_count=data.get("max_retry_count", 3),
            retry_interval=data.get("retry_interval", 10),
            screenshot_dir=data.get("screenshot_dir", "./screenshots"),
            report_dir=data.get("report_dir", "./reports"),
            log_dir=data.get("log_dir", "./logs"),
            log_level=data.get("log_level", "INFO"),
            debug_mode=data.get("debug_mode", False),
            # 流程控制配置
            workflow_mode=data.get("workflow_mode", "complete"),
            workflow_enable_login=data.get("workflow_enable_login", True),
            workflow_enable_profile=data.get("workflow_enable_profile", True),
            workflow_enable_checkin=data.get("workflow_enable_checkin", True),
            workflow_enable_transfer=data.get("workflow_enable_transfer", True),
            # 定时运行配置
            scheduled_run_enabled=data.get("scheduled_run_enabled", False),
            scheduled_run_time=data.get("scheduled_run_time", "08:00"),
        )


# CSV报告列定义（按指定顺序）
CSV_COLUMNS = [
    'nickname',              # 昵称
    'user_id',               # ID
    'phone',                 # 手机号码
    'balance_before',        # 余额前
    'points',                # 积分
    'vouchers',              # 抵扣券
    'coupons',               # 优惠券
    'checkin_reward',        # 签到奖励
    'checkin_total_times',   # 签到总次数
    'checkin_balance_after', # 签到后余额（内部字段）
    'balance_after',         # 余额（最终余额）
    'duration',              # 耗时（秒）
    'status',                # 状态（success字段）
    'login_method',          # 登录方式
    'error_message',         # 错误信息
    'timestamp'              # 时间戳
]
