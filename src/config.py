"""
配置管理模块
Configuration Management Module
"""

import os
import yaml
from typing import Optional
from pathlib import Path

from .models import Config


class ConfigLoader:
    """配置加载器"""
    
    DEFAULT_CONFIG_PATHS = [
        "./config.yaml",
        "./config.yml",
        "./config.json",
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置加载器
        
        Args:
            config_path: 配置文件路径，为空则自动搜索
        """
        self.config_path = config_path
        self._config: Optional[Config] = None
    
    def _find_config_file(self) -> Optional[str]:
        """查找配置文件"""
        if self.config_path and os.path.exists(self.config_path):
            return self.config_path
        
        for path in self.DEFAULT_CONFIG_PATHS:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_yaml(self, filepath: str) -> dict:
        """加载 YAML 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _save_yaml(self, filepath: str, data: dict) -> None:
        """保存 YAML 文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    def load(self) -> Config:
        """加载配置
        
        Returns:
            Config 对象
            
        Raises:
            FileNotFoundError: 配置文件不存在时（如果指定了路径）
        """
        config_file = self._find_config_file()
        
        if config_file is None:
            # 使用默认配置
            self._config = Config()
            return self._config
        
        try:
            data = self._load_yaml(config_file)
            self._config = Config.from_dict(data)
            self.config_path = config_file
            return self._config
        except Exception as e:
            # 配置文件无效，使用默认值
            print(f"Warning: Failed to load config from {config_file}: {e}")
            self._config = Config()
            return self._config
    
    def reload(self) -> Config:
        """重新加载配置
        
        Returns:
            更新后的 Config 对象
        """
        return self.load()
    
    def save(self, config: Optional[Config] = None, filepath: Optional[str] = None) -> None:
        """保存配置到文件
        
        Args:
            config: 要保存的配置，为空则使用当前配置
            filepath: 保存路径，为空则使用当前配置文件路径
        """
        config_to_save = config or self._config
        if config_to_save is None:
            config_to_save = Config()
        
        save_path = filepath or self.config_path or "./config.yaml"
        
        # 确保目录存在
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._save_yaml(save_path, config_to_save.to_dict())
    
    @property
    def config(self) -> Config:
        """获取当前配置"""
        if self._config is None:
            self.load()
        return self._config
    
    def get_default_config(self) -> Config:
        """获取默认配置"""
        return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """便捷函数：加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config 对象
    """
    loader = ConfigLoader(config_path)
    return loader.load()
