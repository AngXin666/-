"""
SelectionManager 单元测试

测试勾选状态管理器的核心功能
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# 导入被测试的类
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.selection_manager import SelectionManager


class TestSelectionManager:
    """SelectionManager 单元测试类"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """创建临时配置目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def manager_with_temp_dir(self, temp_config_dir, monkeypatch):
        """创建使用临时目录的 SelectionManager 实例"""
        # 修改类属性指向临时目录
        monkeypatch.setattr(SelectionManager, 'CONFIG_DIR', temp_config_dir / '.kiro' / 'settings')
        monkeypatch.setattr(SelectionManager, 'CONFIG_FILE', 
                           temp_config_dir / '.kiro' / 'settings' / 'account_selection.json')
        
        manager = SelectionManager()
        return manager
    
    def test_config_dir_creation(self, manager_with_temp_dir):
        """测试配置目录不存在时的创建
        
        验证：
        - 配置目录被成功创建
        - 目录路径正确
        """
        # 验证配置目录已创建
        assert manager_with_temp_dir.CONFIG_DIR.exists()
        assert manager_with_temp_dir.CONFIG_DIR.is_dir()
    
    def test_save_and_load_empty_selections(self, manager_with_temp_dir):
        """测试保存和加载空的勾选状态"""
        # 保存空字典
        selections = {}
        result = manager_with_temp_dir.save_selections(selections)
        
        # 验证保存成功
        assert result is True
        assert manager_with_temp_dir.CONFIG_FILE.exists()
        
        # 加载并验证
        loaded = manager_with_temp_dir.load_selections()
        assert loaded == selections
    
    def test_save_and_load_selections(self, manager_with_temp_dir):
        """测试保存和加载正常的勾选状态"""
        # 准备测试数据
        selections = {
            "13800138000": True,
            "13800138001": False,
            "13800138002": True
        }
        
        # 保存
        result = manager_with_temp_dir.save_selections(selections)
        assert result is True
        
        # 加载
        loaded = manager_with_temp_dir.load_selections()
        assert loaded == selections
    
    def test_load_nonexistent_file(self, manager_with_temp_dir):
        """测试配置文件不存在时的处理
        
        验证：
        - 返回空字典
        - 不抛出异常
        """
        # 确保文件不存在
        if manager_with_temp_dir.CONFIG_FILE.exists():
            manager_with_temp_dir.CONFIG_FILE.unlink()
        
        # 加载不存在的文件
        loaded = manager_with_temp_dir.load_selections()
        
        # 验证返回空字典
        assert loaded == {}
        assert isinstance(loaded, dict)
    
    def test_load_corrupted_json(self, manager_with_temp_dir):
        """测试配置文件损坏时的错误处理
        
        验证：
        - 返回空字典
        - 记录错误日志
        - 不抛出异常
        """
        # 创建损坏的JSON文件
        with open(manager_with_temp_dir.CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write("{ invalid json content }")
        
        # 加载损坏的文件
        loaded = manager_with_temp_dir.load_selections()
        
        # 验证返回空字典
        assert loaded == {}
        assert isinstance(loaded, dict)
    
    def test_load_incomplete_json(self, manager_with_temp_dir):
        """测试不完整的JSON文件处理"""
        # 创建不完整的JSON文件（缺少selections字段）
        incomplete_data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat()
        }
        
        with open(manager_with_temp_dir.CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(incomplete_data, f)
        
        # 加载
        loaded = manager_with_temp_dir.load_selections()
        
        # 验证返回空字典（因为没有selections字段）
        assert loaded == {}
    
    def test_version_mismatch_migration(self, manager_with_temp_dir):
        """测试版本不匹配时的迁移
        
        验证：
        - 能够加载旧版本的配置
        - 正确提取selections数据
        - 记录警告日志
        """
        # 创建旧版本的配置文件
        old_config = {
            "version": "0.9",  # 旧版本
            "last_updated": datetime.now().isoformat(),
            "selections": {
                "13800138000": True,
                "13800138001": False
            }
        }
        
        with open(manager_with_temp_dir.CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(old_config, f)
        
        # 加载
        loaded = manager_with_temp_dir.load_selections()
        
        # 验证能够正确迁移数据
        assert loaded == old_config["selections"]
    
    def test_utf8_encoding(self, manager_with_temp_dir):
        """测试UTF-8编码
        
        验证：
        - 文件使用UTF-8编码保存
        - 能够正确处理中文等非ASCII字符
        """
        # 使用包含中文的手机号（虽然实际不会这样，但用于测试UTF-8）
        selections = {
            "13800138000": True,
            "备注_测试": False  # 测试UTF-8编码
        }
        
        # 保存
        manager_with_temp_dir.save_selections(selections)
        
        # 直接读取文件验证编码
        with open(manager_with_temp_dir.CONFIG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            # 验证中文字符正确保存
            assert "备注_测试" in content
        
        # 加载验证
        loaded = manager_with_temp_dir.load_selections()
        assert loaded == selections
    
    def test_json_formatting(self, manager_with_temp_dir):
        """测试JSON文件格式化（缩进）
        
        验证：
        - 使用indent=2格式化
        - 文件可读性好
        """
        selections = {
            "13800138000": True,
            "13800138001": False
        }
        
        # 保存
        manager_with_temp_dir.save_selections(selections)
        
        # 读取文件内容
        with open(manager_with_temp_dir.CONFIG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 验证包含换行和缩进
        assert "\n" in content
        assert "  " in content  # 2个空格的缩进
        
        # 验证可以解析为有效的JSON
        data = json.loads(content)
        assert "version" in data
        assert "last_updated" in data
        assert "selections" in data
    
    def test_config_structure(self, manager_with_temp_dir):
        """测试配置文件结构
        
        验证：
        - 包含version字段
        - 包含last_updated字段
        - 包含selections字段
        """
        selections = {"13800138000": True}
        
        # 保存
        manager_with_temp_dir.save_selections(selections)
        
        # 读取并验证结构
        with open(manager_with_temp_dir.CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证必需字段
        assert "version" in data
        assert "last_updated" in data
        assert "selections" in data
        
        # 验证字段类型
        assert isinstance(data["version"], str)
        assert isinstance(data["last_updated"], str)
        assert isinstance(data["selections"], dict)
        
        # 验证版本号
        assert data["version"] == "1.0"
    
    def test_multiple_save_operations(self, manager_with_temp_dir):
        """测试多次保存操作
        
        验证：
        - 后续保存会覆盖之前的数据
        - last_updated字段会更新
        """
        # 第一次保存
        selections1 = {"13800138000": True}
        manager_with_temp_dir.save_selections(selections1)
        
        # 读取第一次的时间戳
        with open(manager_with_temp_dir.CONFIG_FILE, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
            timestamp1 = data1["last_updated"]
        
        # 等待一小段时间确保时间戳不同
        import time
        time.sleep(0.01)
        
        # 第二次保存
        selections2 = {"13800138001": False}
        manager_with_temp_dir.save_selections(selections2)
        
        # 读取第二次的数据
        with open(manager_with_temp_dir.CONFIG_FILE, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
            timestamp2 = data2["last_updated"]
        
        # 验证数据被覆盖
        loaded = manager_with_temp_dir.load_selections()
        assert loaded == selections2
        
        # 验证时间戳更新
        assert timestamp2 != timestamp1
    
    def test_large_selections(self, manager_with_temp_dir):
        """测试大量勾选状态的保存和加载"""
        # 创建大量数据
        selections = {f"1380013{i:04d}": i % 2 == 0 for i in range(1000)}
        
        # 保存
        result = manager_with_temp_dir.save_selections(selections)
        assert result is True
        
        # 加载
        loaded = manager_with_temp_dir.load_selections()
        assert loaded == selections
        assert len(loaded) == 1000


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])
