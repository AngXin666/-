"""
动态页面状态类
Dynamic Page State Class

从配置文件动态加载页面状态，无需手动修改代码
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set


class PageStateType:
    """页面状态类型（类似枚举的单个值）"""
    
    def __init__(self, name: str, value: str, chinese_name: str, description: str = ""):
        self.name = name
        self.value = value
        self.chinese_name = chinese_name
        self.description = description
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"PageState.{self.name}"
    
    def __eq__(self, other):
        if isinstance(other, PageStateType):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False
    
    def __hash__(self):
        return hash(self.value)


class PageStateMeta(type):
    """PageState 元类，实现动态属性访问"""
    
    def __getattr__(cls, name):
        """动态获取页面状态"""
        if name.startswith('_'):
            raise AttributeError(f"'{cls.__name__}' object has no attribute '{name}'")
        
        # 从已加载的状态中查找
        if name in cls._states:
            return cls._states[name]
        
        # 如果找不到，返回 UNKNOWN
        return cls._states.get('UNKNOWN', PageStateType('UNKNOWN', 'unknown', '未知页面'))
    
    def __contains__(cls, item):
        """支持 in 操作符"""
        if isinstance(item, PageStateType):
            return item.value in [s.value for s in cls._states.values()]
        elif isinstance(item, str):
            return item in [s.value for s in cls._states.values()]
        return False


class PageState(metaclass=PageStateMeta):
    """动态页面状态类
    
    从配置文件加载所有页面状态，支持类似枚举的访问方式：
    - PageState.LOGIN
    - PageState.HOME
    - 等等
    
    特性：
    - 完全动态，无需修改代码
    - 类型安全（通过 PageStateType 包装）
    - 支持比较操作
    - 支持字符串转换
    """
    
    _states: Dict[str, PageStateType] = {}
    _loaded = False
    _config_path: Optional[Path] = None
    
    @classmethod
    def load_from_config(cls, config_path: Optional[Path] = None):
        """从配置文件加载页面状态
        
        Args:
            config_path: 配置文件路径，默认为 config/page_state_mapping.json
        """
        if cls._loaded and config_path == cls._config_path:
            return  # 已加载且路径相同，跳过
        
        if config_path is None:
            # 自动检测配置文件路径
            import sys
            if getattr(sys, 'frozen', False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "page_state_mapping.json"
        
        cls._config_path = config_path
        
        # 加载配置
        if not config_path.exists():
            # 配置文件不存在，使用默认状态
            cls._load_default_states()
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 清空现有状态
            cls._states.clear()
            
            # 加载映射
            mappings = config.get('mappings', {})
            for class_name, state_config in mappings.items():
                state_name = state_config.get('state', 'UNKNOWN')
                state_value = state_config.get('state_value', 'unknown')
                chinese_name = state_config.get('chinese_name', class_name)
                description = state_config.get('description', '')
                
                # 创建状态对象
                state_obj = PageStateType(state_name, state_value, chinese_name, description)
                cls._states[state_name] = state_obj
            
            # 确保 UNKNOWN 状态存在
            if 'UNKNOWN' not in cls._states:
                cls._states['UNKNOWN'] = PageStateType('UNKNOWN', 'unknown', '未知页面', '未知或未识别的页面')
            
            cls._loaded = True
            
        except Exception as e:
            print(f"[PageState] 加载配置失败: {e}，使用默认状态")
            cls._load_default_states()
    
    @classmethod
    def _load_default_states(cls):
        """加载默认状态（向后兼容）"""
        default_states = {
            'UNKNOWN': ('unknown', '未知页面'),
            'LAUNCHER': ('launcher', 'Android桌面'),
            'AD': ('ad', '广告页'),
            'HOME': ('home', '首页'),
            'PROFILE': ('profile', '个人页（未登录）'),
            'PROFILE_LOGGED': ('profile_logged', '个人页（已登录）'),
            'LOGIN': ('login', '登录页'),
            'LOGIN_ERROR': ('login_error', '登录错误'),
            'LOADING': ('loading', '加载中'),
            'POPUP': ('popup', '弹窗'),
            'CHECKIN': ('checkin', '签到页'),
            'CHECKIN_POPUP': ('checkin_popup', '签到弹窗'),
            'WARMTIP': ('warmtip', '温馨提示'),
            'STARTUP_POPUP': ('startup_popup', '启动页服务弹窗'),
            'HOME_NOTICE': ('home_notice', '首页公告'),
            'HOME_ERROR_POPUP': ('home_error_popup', '首页异常代码弹窗'),
            'POINTS_PAGE': ('points_page', '积分页'),
            'SPLASH': ('splash', '启动页'),
            'TRANSFER': ('transfer', '转账页'),
            'TRANSFER_CONFIRM': ('transfer_confirm', '转账确认弹窗'),
            'WALLET': ('wallet', '钱包页'),
            'TRANSACTION_HISTORY': ('transaction_history', '交易流水'),
            'CATEGORY': ('category', '分类页'),
            'SEARCH': ('search', '搜索页'),
            'ARTICLE': ('article', '文章页'),
            'SETTINGS': ('settings', '设置页'),
            'COUPON': ('coupon', '优惠劵页'),
            'PROFILE_AD': ('profile_ad', '个人页广告'),
        }
        
        cls._states.clear()
        for state_name, (state_value, chinese_name) in default_states.items():
            state_obj = PageStateType(state_name, state_value, chinese_name)
            cls._states[state_name] = state_obj
        
        cls._loaded = True
    
    @classmethod
    def get_by_value(cls, value: str) -> Optional[PageStateType]:
        """通过 value 获取状态
        
        Args:
            value: 状态值（如 "login", "home"）
            
        Returns:
            PageStateType 对象，如果找不到返回 UNKNOWN
        """
        for state in cls._states.values():
            if state.value == value:
                return state
        return cls._states.get('UNKNOWN')
    
    @classmethod
    def get_by_name(cls, name: str) -> Optional[PageStateType]:
        """通过名称获取状态
        
        Args:
            name: 状态名称（如 "LOGIN", "HOME"）
            
        Returns:
            PageStateType 对象，如果找不到返回 UNKNOWN
        """
        return cls._states.get(name, cls._states.get('UNKNOWN'))
    
    @classmethod
    def all_states(cls) -> Dict[str, PageStateType]:
        """获取所有状态"""
        return cls._states.copy()
    
    @classmethod
    def all_values(cls) -> Set[str]:
        """获取所有状态值"""
        return {state.value for state in cls._states.values()}
    
    @classmethod
    def reload(cls):
        """重新加载配置"""
        cls._loaded = False
        cls.load_from_config(cls._config_path)


# 自动加载配置
PageState.load_from_config()


# 为了兼容性，提供一个函数来获取中文名称
def get_chinese_name(state: PageStateType) -> str:
    """获取页面状态的中文名称
    
    Args:
        state: PageStateType 对象
        
    Returns:
        中文名称
    """
    return state.chinese_name if isinstance(state, PageStateType) else str(state)
