"""
收款人选择器单元测试
Recipient Selector Unit Tests
"""

import pytest
import json
from pathlib import Path
from src.recipient_selector import RecipientSelector


class TestRecipientSelector:
    """收款人选择器测试类"""
    
    @pytest.fixture
    def selector(self):
        """创建测试用的选择器实例"""
        selector = RecipientSelector(strategy="rotation")
        # 清理测试数据
        selector.reset_rotation()
        return selector
    
    @pytest.fixture
    def random_selector(self):
        """创建随机策略的选择器实例"""
        return RecipientSelector(strategy="random")
    
    @pytest.fixture
    def recipients(self):
        """测试用的收款人列表"""
        return ["13800138000", "13900139000", "14000140000"]
    
    def test_rotation_selection(self, selector, recipients):
        """测试轮询选择逻辑"""
        # 第一次选择
        r1 = selector.select_recipient(recipients, key="test_user")
        assert r1 == "13800138000"
        
        # 第二次选择
        r2 = selector.select_recipient(recipients, key="test_user")
        assert r2 == "13900139000"
        
        # 第三次选择
        r3 = selector.select_recipient(recipients, key="test_user")
        assert r3 == "14000140000"
        
        # 第四次选择（循环）
        r4 = selector.select_recipient(recipients, key="test_user")
        assert r4 == "13800138000"
    
    def test_random_selection(self, random_selector, recipients):
        """测试随机选择逻辑"""
        # 多次选择，验证都在列表中
        for _ in range(10):
            r = random_selector.select_recipient(recipients)
            assert r in recipients
    
    def test_filter_self(self, selector, recipients):
        """测试自我过滤逻辑"""
        # 发送人是第一个收款人
        r = selector.select_recipient(
            recipients, 
            sender_phone="13800138000",
            key="test_user2"
        )
        # 应该跳过自己，选择下一个
        assert r == "13900139000"
        
        # 再次选择
        r2 = selector.select_recipient(
            recipients,
            sender_phone="13800138000",
            key="test_user2"
        )
        # 应该选择第三个（因为第一个被过滤了）
        assert r2 == "14000140000"
    
    def test_persistence(self, recipients):
        """测试持久化存储"""
        # 第一个选择器
        selector1 = RecipientSelector(strategy="rotation")
        selector1.reset_rotation("test_persist")
        
        r1 = selector1.select_recipient(recipients, key="test_persist")
        assert r1 == "13800138000"
        
        # 创建新的选择器（模拟重启）
        selector2 = RecipientSelector(strategy="rotation")
        r2 = selector2.select_recipient(recipients, key="test_persist")
        # 应该继续上次的索引
        assert r2 == "13900139000"
        
        # 清理
        selector2.reset_rotation("test_persist")
    
    def test_empty_list(self, selector):
        """测试空列表处理"""
        with pytest.raises(ValueError):
            selector.select_recipient([], key="test_empty")
    
    def test_all_filtered(self, selector):
        """测试所有收款人都被过滤的情况"""
        recipients = ["13800138000"]
        r = selector.select_recipient(
            recipients,
            sender_phone="13800138000",
            key="test_all_filtered"
        )
        # 所有收款人都是自己，应该返回None
        assert r is None
    
    def test_multiple_keys(self, selector, recipients):
        """测试多个轮询键独立工作"""
        # 用户1的轮询
        r1_1 = selector.select_recipient(recipients, key="user1")
        assert r1_1 == "13800138000"
        
        # 用户2的轮询
        r2_1 = selector.select_recipient(recipients, key="user2")
        assert r2_1 == "13800138000"
        
        # 用户1继续轮询
        r1_2 = selector.select_recipient(recipients, key="user1")
        assert r1_2 == "13900139000"
        
        # 用户2继续轮询
        r2_2 = selector.select_recipient(recipients, key="user2")
        assert r2_2 == "13900139000"
    
    def test_reset_rotation(self, selector, recipients):
        """测试重置轮询状态"""
        # 选择几次
        selector.select_recipient(recipients, key="test_reset")
        selector.select_recipient(recipients, key="test_reset")
        
        # 重置
        selector.reset_rotation("test_reset")
        
        # 应该从头开始
        r = selector.select_recipient(recipients, key="test_reset")
        assert r == "13800138000"
    
    def test_get_rotation_index(self, selector, recipients):
        """测试获取轮询索引"""
        # 初始索引应该是0
        assert selector.get_rotation_index("test_index") == 0
        
        # 选择一次
        selector.select_recipient(recipients, key="test_index")
        
        # 索引应该是1
        assert selector.get_rotation_index("test_index") == 1
    
    def test_exception_handling(self, selector):
        """测试异常处理"""
        # 测试损坏的状态文件
        rotation_file = Path("runtime_data/transfer_rotation.json")
        rotation_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入无效的JSON
        with open(rotation_file, 'w') as f:
            f.write("invalid json")
        
        # 应该能够处理并返回空字典
        selector2 = RecipientSelector(strategy="rotation")
        assert selector2.rotation_state == {}
        
        # 清理
        selector2.reset_rotation()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
