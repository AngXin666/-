"""
深度学习页面守护配置
定义每个页面的处理规则
"""

from enum import Enum
from typing import Dict, Callable, Optional


class PageAction(Enum):
    """页面处理动作"""
    NONE = "none"  # 不需要处理(流程页面)
    BACK = "back"  # 按返回键
    LOGIN = "login"  # 执行登录
    CLOSE = "close"  # 关闭弹窗
    WAIT = "wait"  # 等待
    RESTART = "restart"  # 重启应用
    ERROR = "error"  # 错误页面(关闭并记录)


class PageGuardConfig:
    """页面守护配置"""
    
    # 页面处理规则
    PAGE_RULES: Dict[str, dict] = {
        # ===== 流程页面 (不需要处理) =====
        "首页": {
            "action": PageAction.NONE,
            "description": "主页面",
            "button": "我的按钮",  # 到个人页
        },
        "个人页_已登录": {
            "action": PageAction.NONE,
            "description": "个人中心(已登录)",
            "button": "首页按钮",  # 返回首页
        },
        "签到页": {
            "action": PageAction.NONE,
            "description": "签到页面",
            "button": "签到按钮",
        },
        "钱包页": {
            "action": PageAction.NONE,
            "description": "钱包页面",
            "button": "转增按钮",  # 到转账页
        },
        "转账页": {
            "action": PageAction.NONE,
            "description": "转账页面",
            "button": "转账按钮",
        },
        "设置": {
            "action": PageAction.NONE,
            "description": "设置页面",
        },
        
        # ===== 需要返回的页面 =====
        "交易流水": {
            "action": PageAction.BACK,
            "description": "交易记录页面,按返回键",
            "button": "返回按钮",
        },
        "积分页": {
            "action": PageAction.BACK,
            "description": "积分页面,按返回键",
            "button": "返回按钮",
        },
        "我的优惠劵": {
            "action": PageAction.BACK,
            "description": "优惠券列表,按返回键",
            "button": "返回按钮",
        },
        "商品列表": {
            "action": PageAction.BACK,
            "description": "商品列表页面,按返回键",
            "button": "返回按钮",
        },
        "分类页": {
            "action": PageAction.BACK,
            "description": "分类页面,点击首页按钮",
            "button": "首页按钮",
        },
        "搜索页": {
            "action": PageAction.BACK,
            "description": "搜索页面,按返回键",
            "button": "返回按钮",
        },
        "文章页": {
            "action": PageAction.BACK,
            "description": "文章详情页面,按返回键",
            "button": "返回按钮",
        },
        
        # ===== 需要登录的页面 =====
        "个人页_未登录": {
            "action": PageAction.LOGIN,
            "description": "个人中心(未登录),需要登录",
            "button": "请登陆按钮",
        },
        "登录页": {
            "action": PageAction.LOGIN,
            "description": "登录页面,需要登录",
        },
        
        # ===== 需要关闭的弹窗 =====
        "启动页服务弹窗": {
            "action": PageAction.CLOSE,
            "description": "启动页服务弹窗,点击关闭",
            "close_position": (270, 800),  # 关闭按钮位置
        },
        "温馨提示": {
            "action": PageAction.CLOSE,
            "description": "温馨提示弹窗,点击确认",
            "close_position": (270, 700),  # 确认按钮位置
        },
        "签到弹窗": {
            "action": PageAction.CLOSE,
            "description": "签到弹窗,点击关闭",
            "close_position": (270, 800),  # 关闭按钮位置
        },
        "首页公告": {
            "action": PageAction.CLOSE,
            "description": "首页公告弹窗,点击关闭",
            "close_position": (270, 800),  # 关闭按钮位置
        },
        
        # ===== 需要等待的页面 =====
        "广告页": {
            "action": PageAction.WAIT,
            "description": "广告页面,等待跳过按钮",
            "wait_time": 5,  # 最多等待5秒
            "skip_position": (480, 50),  # 跳过按钮位置(右上角)
        },
        "加载页": {
            "action": PageAction.WAIT,
            "description": "加载页面,等待加载完成",
            "wait_time": 10,  # 最多等待10秒
        },
        
        # ===== 错误页面 (需要关闭并记录) =====
        "手机号码不存在": {
            "action": PageAction.ERROR,
            "description": "手机号码不存在错误",
            "error_type": "手机号码不存在",
            "close_position": (270, 700),  # 确认按钮位置
        },
        "用户名或密码错误弹窗": {
            "action": PageAction.ERROR,
            "description": "用户名或密码错误",
            "error_type": "用户名或密码错误",
            "close_position": (270, 700),  # 确认按钮位置
        },
        
        # ===== 特殊页面 =====
        "模拟器桌面": {
            "action": PageAction.RESTART,
            "description": "应用已退出到桌面,重新启动",
        },
    }
    
    @classmethod
    def get_page_action(cls, page_name: str) -> PageAction:
        """获取页面处理动作
        
        Args:
            page_name: 页面名称
            
        Returns:
            PageAction: 处理动作
        """
        rule = cls.PAGE_RULES.get(page_name)
        if rule:
            return rule["action"]
        return PageAction.NONE  # 未知页面默认不处理
    
    @classmethod
    def get_page_rule(cls, page_name: str) -> Optional[dict]:
        """获取页面处理规则
        
        Args:
            page_name: 页面名称
            
        Returns:
            dict: 处理规则,如果不存在返回None
        """
        return cls.PAGE_RULES.get(page_name)
    
    @classmethod
    def is_normal_page(cls, page_name: str) -> bool:
        """判断是否是正常流程页面
        
        Args:
            page_name: 页面名称
            
        Returns:
            bool: 是否是正常页面
        """
        action = cls.get_page_action(page_name)
        return action == PageAction.NONE
    
    @classmethod
    def needs_handling(cls, page_name: str) -> bool:
        """判断页面是否需要处理
        
        Args:
            page_name: 页面名称
            
        Returns:
            bool: 是否需要处理
        """
        action = cls.get_page_action(page_name)
        return action != PageAction.NONE
    
    @classmethod
    def get_all_normal_pages(cls) -> list:
        """获取所有正常流程页面
        
        Returns:
            list: 正常页面列表
        """
        return [
            page_name for page_name, rule in cls.PAGE_RULES.items()
            if rule["action"] == PageAction.NONE
        ]
    
    @classmethod
    def get_pages_by_action(cls, action: PageAction) -> list:
        """获取指定动作的所有页面
        
        Args:
            action: 处理动作
            
        Returns:
            list: 页面列表
        """
        return [
            page_name for page_name, rule in cls.PAGE_RULES.items()
            if rule["action"] == action
        ]
