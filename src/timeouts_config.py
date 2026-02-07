"""
超时配置模块 - 统一管理所有超时时间
Timeouts Configuration Module - Centralized timeout management
"""

import json
from pathlib import Path
from typing import Optional, Dict


class TimeoutsConfig:
    """超时配置管理器
    
    提供统一的超时时间配置，支持从文件加载和运行时修改
    """
    
    # ===== 导航相关超时 =====
    NAVIGATION_TIMEOUT = 15.0  # 导航超时（秒）
    PAGE_LOAD_TIMEOUT = 15.0   # 页面加载超时（秒）- 统一为15秒
    PAGE_TRANSITION_TIMEOUT = 5.0  # 页面切换超时（秒）
    
    # ===== 签到相关超时 =====
    CHECKIN_TIMEOUT = 15.0  # 签到超时（秒）
    CHECKIN_PAGE_LOAD = 3.0  # 签到页面加载等待（秒）
    CHECKIN_POPUP_WAIT = 2.0  # 签到弹窗等待（秒）
    CHECKIN_BUTTON_WAIT = 0.5  # 签到按钮点击后等待（秒）
    
    # ===== 转账相关超时 =====
    TRANSFER_TIMEOUT = 20.0  # 转账超时（秒）
    TRANSFER_PAGE_LOAD = 2.0  # 转账页面加载等待（秒）
    TRANSFER_INPUT_WAIT = 1.0  # 转账输入后等待（秒）
    TRANSFER_CONFIRM_WAIT = 2.0  # 转账确认等待（秒）
    
    # ===== OCR识别超时 =====
    OCR_TIMEOUT = 5.0  # OCR识别超时（秒）
    OCR_TIMEOUT_SHORT = 2.0  # OCR短超时（秒）
    OCR_TIMEOUT_LONG = 15.0  # OCR长超时（秒）- 统一为15秒
    
    # ===== 页面检测超时 =====
    PAGE_DETECT_TIMEOUT = 5.0  # 页面检测超时（秒）
    ELEMENT_DETECT_TIMEOUT = 3.0  # 元素检测超时（秒）
    
    # ===== 网络请求超时 =====
    HTTP_REQUEST_TIMEOUT = 15.0  # HTTP请求超时（秒）- 统一为15秒
    HTTP_REQUEST_SHORT = 5.0  # HTTP短超时（秒）
    
    # ===== 等待时间 =====
    WAIT_SHORT = 0.5  # 短等待（秒）
    WAIT_MEDIUM = 1.0  # 中等待（秒）
    WAIT_LONG = 2.0  # 长等待（秒）
    WAIT_EXTRA_LONG = 3.0  # 超长等待（秒）
    
    # ===== 智能等待器超时 =====
    SMART_WAIT_TIMEOUT = 15.0  # 智能等待器默认超时（秒）- 统一为15秒
    SMART_WAIT_INTERVAL = 0.5  # 智能等待器检测间隔（秒）
    
    # ===== 缓存相关 =====
    CACHE_TTL_SHORT = 0.5  # 短缓存时间（秒）
    CACHE_TTL_MEDIUM = 1.0  # 中等缓存时间（秒）
    CACHE_TTL_LONG = 3.0  # 长缓存时间（秒）
    
    # 配置文件路径
    _config_file: Optional[Path] = None
    _custom_config: Dict[str, float] = {}
    
    @classmethod
    def load_from_file(cls, config_path: str) -> bool:
        """从配置文件加载超时配置
        
        配置文件格式（JSON）：
        {
            "NAVIGATION_TIMEOUT": 15.0,
            "PAGE_LOAD_TIMEOUT": 10.0,
            ...
        }
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                print(f"[超时配置] 配置文件不存在: {config_path}")
                return False
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            cls._custom_config = config_data
            cls._config_file = config_file
            
            # 应用配置到类属性
            for key, value in config_data.items():
                if hasattr(cls, key) and isinstance(value, (int, float)):
                    setattr(cls, key, float(value))
            
            print(f"[超时配置] 成功加载配置文件: {config_path}")
            print(f"[超时配置] 加载了 {len(config_data)} 个配置项")
            return True
            
        except Exception as e:
            print(f"[超时配置] 加载配置文件失败: {e}")
            return False
    
    @classmethod
    def save_to_file(cls, config_path: Optional[str] = None) -> bool:
        """保存当前配置到文件
        
        Args:
            config_path: 配置文件路径（可选，默认使用加载时的路径）
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 确定保存路径
            if config_path:
                save_path = Path(config_path)
            elif cls._config_file:
                save_path = cls._config_file
            else:
                save_path = Path("config/timeouts.json")
            
            # 确保目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 收集所有超时配置
            config_data = {}
            for attr_name in dir(cls):
                if attr_name.isupper() and not attr_name.startswith('_'):
                    attr_value = getattr(cls, attr_name)
                    if isinstance(attr_value, (int, float)):
                        config_data[attr_name] = float(attr_value)
            
            # 保存到文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"[超时配置] 成功保存配置到: {save_path}")
            print(f"[超时配置] 保存了 {len(config_data)} 个配置项")
            return True
            
        except Exception as e:
            print(f"[超时配置] 保存配置文件失败: {e}")
            return False
    
    @classmethod
    def set_timeout(cls, name: str, value: float) -> bool:
        """设置单个超时配置
        
        Args:
            name: 配置名称（如"NAVIGATION_TIMEOUT"）
            value: 超时时间（秒）
            
        Returns:
            bool: 是否设置成功
        """
        try:
            if not hasattr(cls, name):
                print(f"[超时配置] 配置项不存在: {name}")
                return False
            
            if not isinstance(value, (int, float)) or value <= 0:
                print(f"[超时配置] 无效的超时值: {value}")
                return False
            
            setattr(cls, name, float(value))
            cls._custom_config[name] = float(value)
            print(f"[超时配置] 已更新 {name} = {value}")
            return True
            
        except Exception as e:
            print(f"[超时配置] 设置配置失败: {e}")
            return False
    
    @classmethod
    def get_timeout(cls, name: str, default: Optional[float] = None) -> float:
        """获取超时配置
        
        Args:
            name: 配置名称
            default: 默认值（如果配置不存在）
            
        Returns:
            float: 超时时间（秒）
        """
        if hasattr(cls, name):
            return getattr(cls, name)
        elif default is not None:
            return default
        else:
            raise ValueError(f"配置项不存在且未提供默认值: {name}")
    
    @classmethod
    def reset_to_defaults(cls) -> None:
        """重置所有配置到默认值"""
        cls._custom_config.clear()
        cls._config_file = None
        
        # 重新加载默认值（通过重新导入模块）
        # 这里简单地重新赋值
        cls.NAVIGATION_TIMEOUT = 15.0
        cls.PAGE_LOAD_TIMEOUT = 15.0  # 统一为15秒
        cls.PAGE_TRANSITION_TIMEOUT = 5.0
        cls.CHECKIN_TIMEOUT = 15.0
        cls.CHECKIN_PAGE_LOAD = 3.0
        cls.CHECKIN_POPUP_WAIT = 2.0
        cls.CHECKIN_BUTTON_WAIT = 0.5
        cls.TRANSFER_TIMEOUT = 20.0
        cls.TRANSFER_PAGE_LOAD = 2.0
        cls.TRANSFER_INPUT_WAIT = 1.0
        cls.TRANSFER_CONFIRM_WAIT = 2.0
        cls.OCR_TIMEOUT = 5.0
        cls.OCR_TIMEOUT_SHORT = 2.0
        cls.OCR_TIMEOUT_LONG = 15.0  # 统一为15秒
        cls.PAGE_DETECT_TIMEOUT = 5.0
        cls.ELEMENT_DETECT_TIMEOUT = 3.0
        cls.HTTP_REQUEST_TIMEOUT = 15.0  # 统一为15秒
        cls.HTTP_REQUEST_SHORT = 5.0
        cls.WAIT_SHORT = 0.5
        cls.WAIT_MEDIUM = 1.0
        cls.WAIT_LONG = 2.0
        cls.WAIT_EXTRA_LONG = 3.0
        cls.SMART_WAIT_TIMEOUT = 15.0  # 统一为15秒
        cls.SMART_WAIT_INTERVAL = 0.5
        cls.CACHE_TTL_SHORT = 0.5
        cls.CACHE_TTL_MEDIUM = 1.0
        cls.CACHE_TTL_LONG = 3.0
        
        print("[超时配置] 已重置所有配置到默认值")
    
    @classmethod
    def print_config(cls) -> None:
        """打印当前所有配置"""
        print("\n" + "=" * 60)
        print("超时配置 - 当前设置")
        print("=" * 60)
        
        # 按类别分组打印
        categories = {
            "导航相关": ["NAVIGATION_TIMEOUT", "PAGE_LOAD_TIMEOUT", "PAGE_TRANSITION_TIMEOUT"],
            "签到相关": ["CHECKIN_TIMEOUT", "CHECKIN_PAGE_LOAD", "CHECKIN_POPUP_WAIT", "CHECKIN_BUTTON_WAIT"],
            "转账相关": ["TRANSFER_TIMEOUT", "TRANSFER_PAGE_LOAD", "TRANSFER_INPUT_WAIT", "TRANSFER_CONFIRM_WAIT"],
            "OCR识别": ["OCR_TIMEOUT", "OCR_TIMEOUT_SHORT", "OCR_TIMEOUT_LONG"],
            "页面检测": ["PAGE_DETECT_TIMEOUT", "ELEMENT_DETECT_TIMEOUT"],
            "网络请求": ["HTTP_REQUEST_TIMEOUT", "HTTP_REQUEST_SHORT"],
            "等待时间": ["WAIT_SHORT", "WAIT_MEDIUM", "WAIT_LONG", "WAIT_EXTRA_LONG"],
            "智能等待": ["SMART_WAIT_TIMEOUT", "SMART_WAIT_INTERVAL"],
            "缓存时间": ["CACHE_TTL_SHORT", "CACHE_TTL_MEDIUM", "CACHE_TTL_LONG"],
        }
        
        for category, config_names in categories.items():
            print(f"\n{category}:")
            for name in config_names:
                if hasattr(cls, name):
                    value = getattr(cls, name)
                    is_custom = name in cls._custom_config
                    marker = " *" if is_custom else ""
                    print(f"  {name:30s} = {value:6.2f} 秒{marker}")
        
        if cls._custom_config:
            print("\n注: 标记 * 的配置项已被自定义修改")
        
        if cls._config_file:
            print(f"\n配置文件: {cls._config_file}")
        
        print("=" * 60 + "\n")


# 创建全局便捷函数
def get_timeout(name: str, default: Optional[float] = None) -> float:
    """获取超时配置的便捷函数
    
    Args:
        name: 配置名称
        default: 默认值
        
    Returns:
        float: 超时时间（秒）
    """
    return TimeoutsConfig.get_timeout(name, default)


def load_timeouts_config(config_path: str) -> bool:
    """加载超时配置的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        bool: 是否加载成功
    """
    return TimeoutsConfig.load_from_file(config_path)


# 模块初始化时尝试加载配置文件
def _init_config():
    """初始化配置（尝试加载默认配置文件）"""
    default_config_paths = [
        "config/timeouts.json",
        "config/timeouts_config.json",
        ".kiro/timeouts.json",
    ]
    
    for config_path in default_config_paths:
        if Path(config_path).exists():
            TimeoutsConfig.load_from_file(config_path)
            break


# 自动初始化
_init_config()
