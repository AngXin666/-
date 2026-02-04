"""
测试昵称提取逻辑
Test nickname extraction logic
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.profile_reader import ProfileReader
from unittest.mock import Mock

def test_nickname_extraction():
    """测试昵称提取逻辑"""
    print("=" * 70)
    print("测试昵称提取逻辑")
    print("=" * 70)
    
    # 创建最小化的ProfileReader实例
    mock_adb = Mock()
    
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
    
    # 测试用例
    test_cases = [
        {
            'name': '单个中文昵称',
            'texts': ['李四'],
            'expected': '李四',
        },
        {
            'name': '昵称+会员标识',
            'texts': ['王五VIP会员'],
            'expected': '王五',
        },
        {
            'name': '多个候选，选择最优',
            'texts': ['123', '李四', 'ID456'],
            'expected': '李四',
        },
        {
            'name': '包含排除关键字',
            'texts': ['ID123', '余额100'],
            'expected': None,
        },
        {
            'name': '空文本列表',
            'texts': [],
            'expected': None,
        },
        {
            'name': '钻石会员标识',
            'texts': ['赵六钻石会员'],
            'expected': '赵六',
        },
        {
            'name': '混合文本',
            'texts': ['1肉', '李四', '123'],
            'expected': '李四',
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[测试 {i}] {test_case['name']}")
        print(f"  输入: {test_case['texts']}")
        
        result = reader._extract_nickname_from_texts(test_case['texts'])
        
        print(f"  期望: {test_case['expected']}")
        print(f"  实际: {result}")
        
        if result == test_case['expected']:
            print(f"  ✓ 通过")
            passed += 1
        else:
            print(f"  ✗ 失败")
            failed += 1
    
    # 总结
    print("\n" + "=" * 70)
    print(f"测试总结:")
    print(f"  通过: {passed}/{len(test_cases)}")
    print(f"  失败: {failed}/{len(test_cases)}")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {failed} 个测试失败")
        return 1

if __name__ == '__main__':
    exit(test_nickname_extraction())
