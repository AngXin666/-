"""
自动签到助手
Emulator Automation Script

自动签到助手 - 支持批量多开、自动登录、签到、抽奖
"""

__version__ = "2.1.0"
__author__ = "Automation Script"

from .models import (
    Account,
    AccountResult,
    AccountStatus,
    Config,
    DrawResult,
    EmulatorInstance,
    InstanceStatus,
    SignInResult,
)
from .config import ConfigLoader
from .emulator_controller import EmulatorController, EmulatorType
from .adb_bridge import ADBBridge
from .screen_capture import ScreenCapture
from .ui_automation import UIAutomation, Action, ActionType
from .auto_login import AutoLogin, LoginResult
from .account_manager import AccountManager
from .ximeng_automation import XimengAutomation
from .instance_manager import InstanceManager
from .orchestrator import Orchestrator
from .logger import Logger, get_logger
from .gui import AutomationGUI

__all__ = [
    # Models
    "Account",
    "AccountResult",
    "AccountStatus",
    "Config",
    "DrawResult",
    "EmulatorInstance",
    "InstanceStatus",
    "SignInResult",
    # Config
    "ConfigLoader",
    # Controllers
    "EmulatorController",
    "EmulatorType",
    "ADBBridge",
    # Screen & UI
    "ScreenCapture",
    "UIAutomation",
    "Action",
    "ActionType",
    # Automation
    "AutoLogin",
    "LoginResult",
    "AccountManager",
    "XimengAutomation",
    # Management
    "InstanceManager",
    "Orchestrator",
    # Logging
    "Logger",
    "get_logger",
    # GUI
    "AutomationGUI",
]
