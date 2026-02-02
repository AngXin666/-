"""
错误类型枚举的单元测试

测试内容：
1. 测试每个错误类型的映射是否正确
2. 测试未知错误类型的处理
3. 测试显示文本格式是否符合规范

Requirements: 1.1-1.12
"""

import sys
import unittest
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.error_types import ErrorType


class TestErrorTypes(unittest.TestCase):
    """错误类型枚举测试类"""
    
    def test_all_error_types_have_mapping(self):
        """测试所有错误类型都有对应的显示文本映射
        
        验证每个ErrorType枚举值都能通过to_display_text()获得非空的显示文本
        """
        # 获取所有错误类型
        all_error_types = list(ErrorType)
        
        # 验证每个错误类型都有映射
        for error_type in all_error_types:
            with self.subTest(error_type=error_type):
                display_text = ErrorType.to_display_text(error_type)
                
                # 验证返回值不为空
                self.assertIsNotNone(display_text, 
                    f"错误类型 {error_type} 返回了None")
                self.assertTrue(len(display_text) > 0, 
                    f"错误类型 {error_type} 返回了空字符串")
    
    def test_login_phone_not_exist_mapping(self):
        """测试登录失败-手机号不存在的映射 (Requirement 1.1)"""
        result = ErrorType.to_display_text(ErrorType.LOGIN_PHONE_NOT_EXIST)
        self.assertEqual(result, "登录失败:手机号不存在")
    
    def test_login_password_error_mapping(self):
        """测试登录失败-密码错误的映射 (Requirement 1.2)"""
        result = ErrorType.to_display_text(ErrorType.LOGIN_PASSWORD_ERROR)
        self.assertEqual(result, "登录失败:密码错误")
    
    def test_cannot_reach_profile_mapping(self):
        """测试无法到达个人页的映射 (Requirement 1.3)"""
        result = ErrorType.to_display_text(ErrorType.CANNOT_REACH_PROFILE)
        self.assertEqual(result, "失败:无法到达个人页")
    
    def test_cannot_read_profile_mapping(self):
        """测试无法读取个人资料的映射 (Requirement 1.4)"""
        result = ErrorType.to_display_text(ErrorType.CANNOT_READ_PROFILE)
        self.assertEqual(result, "失败:无法读取个人资料")
    
    def test_cannot_reach_checkin_mapping(self):
        """测试无法到达签到页的映射 (Requirement 1.5)"""
        result = ErrorType.to_display_text(ErrorType.CANNOT_REACH_CHECKIN)
        self.assertEqual(result, "失败:无法到达签到页")
    
    def test_checkin_failed_mapping(self):
        """测试签到失败的映射 (Requirement 1.6)"""
        result = ErrorType.to_display_text(ErrorType.CHECKIN_FAILED)
        self.assertEqual(result, "失败:签到失败")
    
    def test_checkin_exception_mapping(self):
        """测试签到异常的映射 (Requirement 1.7)"""
        result = ErrorType.to_display_text(ErrorType.CHECKIN_EXCEPTION)
        self.assertEqual(result, "失败:签到异常")
    
    def test_cannot_get_final_data_mapping(self):
        """测试获取最终资料失败的映射 (Requirement 1.8)"""
        result = ErrorType.to_display_text(ErrorType.CANNOT_GET_FINAL_DATA)
        self.assertEqual(result, "失败:获取最终资料失败")
    
    def test_transfer_failed_mapping(self):
        """测试转账失败的映射 (Requirement 1.9)"""
        result = ErrorType.to_display_text(ErrorType.TRANSFER_FAILED)
        self.assertEqual(result, "失败:转账失败")
    
    def test_display_text_format(self):
        """测试显示文本格式符合"阶段:具体原因"规范 (Requirement 1.12)"""
        all_error_types = list(ErrorType)
        
        for error_type in all_error_types:
            with self.subTest(error_type=error_type):
                display_text = ErrorType.to_display_text(error_type)
                
                # 验证格式：应该包含冒号，表示"阶段:具体原因"
                self.assertIn(":", display_text, 
                    f"错误文本 '{display_text}' 不符合'阶段:具体原因'格式")
                
                # 验证冒号前后都有内容
                parts = display_text.split(":")
                self.assertEqual(len(parts), 2, 
                    f"错误文本 '{display_text}' 应该只包含一个冒号")
                self.assertTrue(len(parts[0]) > 0, 
                    f"错误文本 '{display_text}' 的阶段部分为空")
                self.assertTrue(len(parts[1]) > 0, 
                    f"错误文本 '{display_text}' 的原因部分为空")
    
    def test_display_text_length_limit(self):
        """测试显示文本长度不超过30个字符 (Requirement 1.13)
        
        注意：这个测试验证to_display_text返回的原始文本。
        GUI层的_format_status方法会进一步处理超长文本。
        """
        all_error_types = list(ErrorType)
        
        for error_type in all_error_types:
            with self.subTest(error_type=error_type):
                display_text = ErrorType.to_display_text(error_type)
                
                # 验证长度不超过30个字符
                self.assertLessEqual(len(display_text), 30, 
                    f"错误文本 '{display_text}' 长度为 {len(display_text)}，超过30个字符")
    
    def test_unknown_error_type_handling(self):
        """测试未知错误类型的处理
        
        当传入一个不在映射表中的错误类型时，应该返回默认的"失败:未知错误"
        
        注意：由于ErrorType是枚举，正常情况下不会有"未知"的枚举值。
        这个测试主要验证to_display_text方法的容错性。
        """
        # 创建一个模拟的未知错误类型（通过修改枚举值）
        # 由于Python枚举的限制，我们通过传入None来测试容错性
        
        # 测试None的情况
        result = ErrorType.to_display_text(None)
        self.assertEqual(result, "失败:未知错误", 
            "未知错误类型应该返回'失败:未知错误'")
    
    def test_error_type_enum_values(self):
        """测试错误类型枚举值的唯一性和正确性"""
        all_error_types = list(ErrorType)
        
        # 验证枚举值的唯一性
        enum_values = [et.value for et in all_error_types]
        self.assertEqual(len(enum_values), len(set(enum_values)), 
            "错误类型枚举值存在重复")
        
        # 验证枚举数量（应该有9个错误类型）
        self.assertEqual(len(all_error_types), 9, 
            f"错误类型数量不正确，期望9个，实际{len(all_error_types)}个")
    
    def test_error_categories(self):
        """测试错误类型的分类是否完整
        
        验证是否包含所有必需的错误类别：
        - 登录相关错误（2个）
        - 导航相关错误（3个）
        - 签到相关错误（2个）
        - 数据获取错误（1个）
        - 转账相关错误（1个）
        """
        # 登录相关错误
        login_errors = [
            ErrorType.LOGIN_PHONE_NOT_EXIST,
            ErrorType.LOGIN_PASSWORD_ERROR,
        ]
        for error in login_errors:
            self.assertIn(error, ErrorType, 
                f"缺少登录相关错误: {error}")
        
        # 导航相关错误
        navigation_errors = [
            ErrorType.CANNOT_REACH_PROFILE,
            ErrorType.CANNOT_READ_PROFILE,
            ErrorType.CANNOT_REACH_CHECKIN,
        ]
        for error in navigation_errors:
            self.assertIn(error, ErrorType, 
                f"缺少导航相关错误: {error}")
        
        # 签到相关错误
        checkin_errors = [
            ErrorType.CHECKIN_FAILED,
            ErrorType.CHECKIN_EXCEPTION,
        ]
        for error in checkin_errors:
            self.assertIn(error, ErrorType, 
                f"缺少签到相关错误: {error}")
        
        # 数据获取错误
        self.assertIn(ErrorType.CANNOT_GET_FINAL_DATA, ErrorType, 
            "缺少数据获取错误")
        
        # 转账相关错误
        self.assertIn(ErrorType.TRANSFER_FAILED, ErrorType, 
            "缺少转账相关错误")


class TestErrorTypeEdgeCases(unittest.TestCase):
    """错误类型边界情况测试"""
    
    def test_display_text_no_trailing_spaces(self):
        """测试显示文本没有前后空格"""
        all_error_types = list(ErrorType)
        
        for error_type in all_error_types:
            with self.subTest(error_type=error_type):
                display_text = ErrorType.to_display_text(error_type)
                
                # 验证没有前后空格
                self.assertEqual(display_text, display_text.strip(), 
                    f"错误文本 '{display_text}' 包含前后空格")
    
    def test_display_text_consistency(self):
        """测试多次调用to_display_text返回相同结果"""
        all_error_types = list(ErrorType)
        
        for error_type in all_error_types:
            with self.subTest(error_type=error_type):
                # 多次调用
                result1 = ErrorType.to_display_text(error_type)
                result2 = ErrorType.to_display_text(error_type)
                result3 = ErrorType.to_display_text(error_type)
                
                # 验证结果一致
                self.assertEqual(result1, result2, 
                    f"错误类型 {error_type} 的显示文本不一致")
                self.assertEqual(result2, result3, 
                    f"错误类型 {error_type} 的显示文本不一致")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestErrorTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorTypeEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
