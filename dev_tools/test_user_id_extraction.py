"""
测试用户ID提取逻辑的修复
"""

import re


def extract_user_id_old(texts):
    """旧版本：要求文本中必须包含'ID'"""
    for text in texts:
        text = text.strip()
        if 'ID' in text or 'id' in text:
            match = re.search(r'(\d{6,})', text)
            if match:
                return match.group(1)
    return None


def extract_user_id_new(texts):
    """新版本：直接提取6位以上的数字"""
    for text in texts:
        text = text.strip()
        # 修复：不要求文本中必须包含"ID"，直接提取6位以上的数字
        match = re.search(r'(\d{6,})', text)
        if match:
            return match.group(1)
    return None


def test_user_id_extraction():
    """测试用户ID提取"""
    
    print("="*80)
    print("测试用户ID提取逻辑")
    print("="*80)
    
    # 测试用例
    test_cases = [
        {
            'name': '完整格式（包含ID前缀）',
            'texts': ['ID:1643524'],
            'expected': '1643524'
        },
        {
            'name': '只有数字（OCR只识别到数字部分）',
            'texts': ['1643524'],
            'expected': '1643524'
        },
        {
            'name': '多个文本，数字在第二个',
            'texts': ['用户信息', '1643524'],
            'expected': '1643524'
        },
        {
            'name': '带空格的格式',
            'texts': ['ID: 1643524'],
            'expected': '1643524'
        },
        {
            'name': '中文冒号',
            'texts': ['ID：1643524'],
            'expected': '1643524'
        },
        {
            'name': '小写id',
            'texts': ['id:1643524'],
            'expected': '1643524'
        },
        {
            'name': '7位数字',
            'texts': ['1234567'],
            'expected': '1234567'
        },
        {
            'name': '8位数字',
            'texts': ['12345678'],
            'expected': '12345678'
        },
        {
            'name': '5位数字（不应匹配）',
            'texts': ['12345'],
            'expected': None
        },
        {
            'name': '混合文本',
            'texts': ['用户ID1643524其他文本'],
            'expected': '1643524'
        },
    ]
    
    print("\n" + "="*80)
    print("旧版本测试（要求包含'ID'）")
    print("="*80)
    
    old_passed = 0
    old_failed = 0
    
    for i, case in enumerate(test_cases, 1):
        result = extract_user_id_old(case['texts'])
        expected = case['expected']
        
        if result == expected:
            print(f"  ✓ 测试{i}: {case['name']}")
            print(f"      输入: {case['texts']}")
            print(f"      结果: {result}")
            old_passed += 1
        else:
            print(f"  ✗ 测试{i}: {case['name']}")
            print(f"      输入: {case['texts']}")
            print(f"      期望: {expected}")
            print(f"      实际: {result}")
            old_failed += 1
    
    print(f"\n旧版本: {old_passed}通过 / {old_failed}失败")
    
    print("\n" + "="*80)
    print("新版本测试（直接提取数字）")
    print("="*80)
    
    new_passed = 0
    new_failed = 0
    
    for i, case in enumerate(test_cases, 1):
        result = extract_user_id_new(case['texts'])
        expected = case['expected']
        
        if result == expected:
            print(f"  ✓ 测试{i}: {case['name']}")
            print(f"      输入: {case['texts']}")
            print(f"      结果: {result}")
            new_passed += 1
        else:
            print(f"  ✗ 测试{i}: {case['name']}")
            print(f"      输入: {case['texts']}")
            print(f"      期望: {expected}")
            print(f"      实际: {result}")
            new_failed += 1
    
    print(f"\n新版本: {new_passed}通过 / {new_failed}失败")
    
    print("\n" + "="*80)
    print("对比总结")
    print("="*80)
    print(f"旧版本: {old_passed}/{len(test_cases)} 通过 ({old_passed/len(test_cases)*100:.1f}%)")
    print(f"新版本: {new_passed}/{len(test_cases)} 通过 ({new_passed/len(test_cases)*100:.1f}%)")
    
    if new_passed > old_passed:
        print(f"\n✓ 新版本修复了 {new_passed - old_passed} 个问题")
    
    print("="*80)
    
    # 重点测试：日志中出现的实际情况
    print("\n" + "="*80)
    print("实际场景测试（日志中的情况）")
    print("="*80)
    
    real_case = {
        'name': '实际日志场景：OCR只识别到数字',
        'texts': ['1643524'],  # 这是日志中实际出现的情况
        'expected': '1643524'
    }
    
    old_result = extract_user_id_old(real_case['texts'])
    new_result = extract_user_id_new(real_case['texts'])
    
    print(f"\n场景: {real_case['name']}")
    print(f"输入: {real_case['texts']}")
    print(f"期望: {real_case['expected']}")
    print(f"\n旧版本结果: {old_result} {'✗ 失败' if old_result != real_case['expected'] else '✓ 成功'}")
    print(f"新版本结果: {new_result} {'✗ 失败' if new_result != real_case['expected'] else '✓ 成功'}")
    
    if old_result is None and new_result == real_case['expected']:
        print("\n✓ 修复成功！新版本能够正确提取用户ID")
    
    print("="*80)


if __name__ == "__main__":
    test_user_id_extraction()
