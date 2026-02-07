"""
勾选状态管理器

负责保存和加载GUI表格中账号的勾选状态
"""

import json
from pathlib import Path
from typing import Dict
from datetime import datetime
from .logging_config import setup_logger

# 使用统一的日志配置
logger = setup_logger(__name__)


class SelectionManager:
    """表格勾选状态管理器"""
    
    # 配置文件路径常量
    CONFIG_DIR = Path(".kiro/settings")
    CONFIG_FILE = CONFIG_DIR / "account_selection.json"
    CONFIG_VERSION = "1.0"
    
    def __init__(self):
        """初始化管理器"""
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        try:
            self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            logger.debug(f"配置目录已确保存在: {self.CONFIG_DIR}")
        except Exception as e:
            logger.error(f"创建配置目录失败: {e}")
            raise
    
    def save_selections(self, selections: Dict[str, bool]) -> bool:
        """保存勾选状态到配置文件
        
        Args:
            selections: 手机号到勾选状态的映射 {"13800138000": True, ...}
            
        Returns:
            True if 保存成功, False otherwise
        """
        try:
            # 构建包含version、last_updated和selections的JSON结构
            config_data = {
                "version": self.CONFIG_VERSION,
                "last_updated": datetime.now().isoformat(),
                "selections": selections
            }
            
            # 使用UTF-8编码和indent=2格式化保存
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"勾选状态已保存: {len(selections)} 个账号")
            return True
            
        except Exception as e:
            logger.error(f"保存勾选状态失败: {e}")
            return False
    
    def load_selections(self) -> Dict[str, bool]:
        """从配置文件加载勾选状态
        
        Returns:
            手机号到勾选状态的映射，如果文件不存在或损坏则返回空字典
        """
        # 处理文件不存在的情况
        if not self.CONFIG_FILE.exists():
            logger.debug("配置文件不存在，返回空状态")
            return {}
        
        try:
            # 读取配置文件
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 验证版本
            version = config_data.get("version", "")
            if version != self.CONFIG_VERSION:
                logger.warning(f"配置文件版本不匹配: {version} != {self.CONFIG_VERSION}")
                # 尝试迁移或使用默认值
                return self._migrate_config(config_data)
            
            # 提取selections字段
            selections = config_data.get("selections", {})
            logger.debug(f"勾选状态已加载: {len(selections)} 个账号")
            return selections
            
        except json.JSONDecodeError as e:
            # 处理JSON解析错误
            logger.error(f"配置文件格式错误: {e}")
            return {}
        except Exception as e:
            logger.error(f"加载勾选状态失败: {e}")
            return {}
    
    def _migrate_config(self, old_config: dict) -> Dict[str, bool]:
        """迁移旧版本配置文件
        
        Args:
            old_config: 旧版本的配置数据
            
        Returns:
            迁移后的勾选状态映射
        """
        # 目前只有一个版本，直接返回selections字段
        # 未来如果有版本升级，在这里实现迁移逻辑
        selections = old_config.get("selections", {})
        logger.info(f"配置文件已迁移，加载 {len(selections)} 个账号状态")
        return selections
