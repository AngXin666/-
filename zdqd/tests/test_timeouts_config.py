"""
超时配置模块测试
"""

import pytest
import json
from pathlib import Path
from src.timeouts_config import TimeoutsConfig, get_timeout, load_timeouts_config


class TestTimeoutsConfig:
    """超时配置测试类"""
    
    def test_default_values(self):
        """测试默认值"""
        # 导航相关
        assert TimeoutsConfig.NAVIGATION_TIMEOUT == 15.0
        assert TimeoutsConfig.PAGE_LOAD_TIMEOUT == 10.0
        
        # 签到相关
        assert TimeoutsConfig.CHECKIN_TIMEOUT == 15.0
        assert TimeoutsConfig.CHECKIN_PAGE_LOAD == 3.0
        
        # 转账相关
        assert TimeoutsConfig.TRANSFER_TIMEOUT == 20.0
        assert TimeoutsConfig.TRANSFER_PAGE_LOAD == 2.0
        
        # OCR相关
        assert TimeoutsConfig.OCR_TIMEOUT == 5.0
        assert TimeoutsConfig.OCR_TIMEOUT_SHORT == 2.0
        
        # 等待时间
        assert TimeoutsConfig.WAIT_SHORT == 0.5
        assert TimeoutsConfig.WAIT_MEDIUM == 1.0
        assert TimeoutsConfig.WAIT_LONG == 2.0
    
    def test_get_timeout(self):
        """测试获取超时配置"""
        # 获取存在的配置
        assert get_timeout("NAVIGATION_TIMEOUT") == 15.0
        assert get_timeout("CHECKIN_TIMEOUT") == 15.0
        
        # 获取不存在的配置（使用默认值）
        assert get_timeout("NON_EXISTENT", 99.0) == 99.0
        
        # 获取不存在的配置（无默认值，应该抛出异常）
        with pytest.raises(ValueError):
            get_timeout("NON_EXISTENT")
    
    def test_set_timeout(self):
        """测试设置超时配置"""
        # 保存原始值
        original_value = TimeoutsConfig.NAVIGATION_TIMEOUT
        
        # 设置新值
        assert TimeoutsConfig.set_timeout("NAVIGATION_TIMEOUT", 60.0)
        assert TimeoutsConfig.NAVIGATION_TIMEOUT == 60.0
        
        # 恢复原始值
        TimeoutsConfig.set_timeout("NAVIGATION_TIMEOUT", original_value)
        
        # 设置无效值
        assert not TimeoutsConfig.set_timeout("NAVIGATION_TIMEOUT", -1.0)
        assert not TimeoutsConfig.set_timeout("NAVIGATION_TIMEOUT", 0)
        
        # 设置不存在的配置
        assert not TimeoutsConfig.set_timeout("NON_EXISTENT", 10.0)
    
    def test_save_and_load_config(self, tmp_path):
        """测试保存和加载配置"""
        # 创建临时配置文件
        config_file = tmp_path / "test_timeouts.json"
        
        # 修改一些配置
        TimeoutsConfig.set_timeout("NAVIGATION_TIMEOUT", 45.0)
        TimeoutsConfig.set_timeout("CHECKIN_TIMEOUT", 20.0)
        
        # 保存配置
        assert TimeoutsConfig.save_to_file(str(config_file))
        assert config_file.exists()
        
        # 验证文件内容
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        assert config_data["NAVIGATION_TIMEOUT"] == 45.0
        assert config_data["CHECKIN_TIMEOUT"] == 20.0
        
        # 重置配置
        TimeoutsConfig.reset_to_defaults()
        assert TimeoutsConfig.NAVIGATION_TIMEOUT == 15.0
        assert TimeoutsConfig.CHECKIN_TIMEOUT == 15.0
        
        # 加载配置
        assert TimeoutsConfig.load_from_file(str(config_file))
        assert TimeoutsConfig.NAVIGATION_TIMEOUT == 45.0
        assert TimeoutsConfig.CHECKIN_TIMEOUT == 20.0
        
        # 清理：重置配置
        TimeoutsConfig.reset_to_defaults()
    
    def test_reset_to_defaults(self):
        """测试重置到默认值"""
        # 修改一些配置
        TimeoutsConfig.set_timeout("NAVIGATION_TIMEOUT", 99.0)
        TimeoutsConfig.set_timeout("CHECKIN_TIMEOUT", 99.0)
        
        # 重置
        TimeoutsConfig.reset_to_defaults()
        
        # 验证已重置
        assert TimeoutsConfig.NAVIGATION_TIMEOUT == 15.0
        assert TimeoutsConfig.CHECKIN_TIMEOUT == 15.0
        assert TimeoutsConfig._custom_config == {}
    
    def test_print_config(self, capsys):
        """测试打印配置"""
        TimeoutsConfig.print_config()
        
        # 捕获输出
        captured = capsys.readouterr()
        
        # 验证输出包含关键信息
        assert "超时配置" in captured.out
        assert "NAVIGATION_TIMEOUT" in captured.out
        assert "CHECKIN_TIMEOUT" in captured.out
        assert "TRANSFER_TIMEOUT" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
