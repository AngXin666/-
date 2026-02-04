"""
简单测试ProfileReader的辅助方法
Simple test for ProfileReader helper methods
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.profile_reader import ProfileReader

def test_helper_methods():
    """测试文本特征检测辅助方法"""
    print("=" * 70)
    print("测试ProfileReader辅助方法")
    print("=" * 70)
    
    # 创建ProfileReader实例（不需要真实的ADB）
    # 我们需要绕过ModelManager的初始化，直接访问辅助方法
    # 创建一个最小化的ProfileReader实例
    from unittest.mock import Mock
    
    # 创建mock对象
    mock_adb = Mock()
    
    # 临时修改ProfileReader的__init__以避免ModelManager初始化
    original_init = ProfileReader.__init__
    
    def mock_init(self, adb, yolo_detector=None):
        self.adb = adb
        self._ocr_pool = None
        self._cache = None
        self._integrated_detector = None
        self._yolo_detector = None
        self._silent_log = None
    
    ProfileReader.__init__ = mock_init
    reader = ProfileReader(adb=mock_adb, yolo_detector=None)
    ProfileReader.__init__ = original_init
    
    # 测试中文文本检测
    print("\n[1] 测试中文文本检测...")
    test_cases = [
        ("张三", True),
        ("李四", True),
        ("王五123", True),
        ("123", False),
        ("abc", False),
        ("张三abc", True),
    ]
    
    passed = 0
    failed = 0
    for text, expected in test_cases:
        result = reader._is_chinese_text(text)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status} '{text}': {result} (期望: {expected})")
    
    # 测试纯数字检测
    print("\n[2] 测试纯数字检测...")
    test_cases = [
        ("123", True),
        ("456", True),
        ("12.3", False),
        ("1a3", False),
        ("张三", False),
    ]
    
    for text, expected in test_cases:
        result = reader._is_pure_number(text)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status} '{text}': {result} (期望: {expected})")
    
    # 测试纯符号检测
    print("\n[3] 测试纯符号检测...")
    test_cases = [
        ("!!!", True),
        ("@#$", True),
        ("123", False),
        ("abc", False),
        ("张三", False),
        ("a!", False),
    ]
    
    for text, expected in test_cases:
        result = reader._is_pure_symbol(text)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status} '{text}': {result} (期望: {expected})")
    
    # 测试置信度评分
    print("\n[4] 测试置信度评分...")
    test_cases = [
        ("张三", "中文昵称"),
        ("李四VIP", "中文+VIP"),
        ("123", "纯数字"),
        ("ID123", "包含排除关键字"),
        ("余额100", "包含排除关键字"),
        ("王五", "正常中文"),
    ]
    
    for text, description in test_cases:
        score = reader._calculate_nickname_confidence(text)
        print(f"  '{text}' ({description}): {score:.2f}")
        if 0.0 <= score <= 1.0:
            passed += 1
        else:
            failed += 1
            print(f"    ✗ 分数超出范围 [0.0, 1.0]")
    
    # 总结
    print("\n" + "=" * 70)
    print(f"测试总结:")
    print(f"  通过: {passed}")
    print(f"  失败: {failed}")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {failed} 个测试失败")
        return 1

if __name__ == '__main__':
    exit(test_helper_methods())
