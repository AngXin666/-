"""
测试置信度评分系统的实现
Test the confidence scoring system implementation
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 直接导入需要测试的类（避免相对导入问题）
import importlib.util
spec = importlib.util.spec_from_file_location("profile_reader", os.path.join(os.path.dirname(__file__), 'src', 'profile_reader.py'))
profile_reader_module = importlib.util.module_from_spec(spec)

# 模拟必要的依赖
class MockADBBridge:
    pass

class MockYOLODetector:
    pass

# 注入模拟依赖到模块
sys.modules['src.adb_bridge'] = type('module', (), {'ADBBridge': MockADBBridge})()
sys.modules['src.account_cache'] = type('module', (), {'get_account_cache': lambda: None})()
sys.modules['src.ocr_image_processor'] = type('module', (), {'enhance_for_ocr': lambda x: x})()
sys.modules['src.ocr_thread_pool'] = type('module', (), {'get_ocr_pool': lambda: None})()
sys.modules['src.logger'] = type('module', (), {'get_silent_logger': lambda: None})()
sys.modules['src.model_manager'] = type('module', (), {'ModelManager': type('ModelManager', (), {'get_instance': lambda: type('instance', (), {'get_ocr_thread_pool': lambda: None})()})})()

# 现在加载模块
spec.loader.exec_module(profile_reader_module)
ProfileReader = profile_reader_module.ProfileReader


def test_confidence_scoring():
    """测试置信度评分功能"""
    
    # 创建ProfileReader实例（不需要ADB和YOLO检测器）
    reader = ProfileReader(adb=None, yolo_detector=None)
    
    print("=" * 60)
    print("测试置信度评分系统")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        # (文本, 期望分数范围, 描述)
        ("张三", (0.6, 1.0), "纯中文昵称"),
        ("李四VIP", (0.5, 0.9), "中文+字母昵称"),
        ("123", (0.0, 0.3), "短纯数字"),
        ("1234567", (0.3, 0.6), "长纯数字"),
        ("ID123456", (0.0, 0.0), "包含排除关键字ID"),
        ("余额100", (0.0, 0.0), "包含排除关键字余额"),
        ("积分", (0.0, 0.0), "包含排除关键字积分"),
        ("王五@#", (0.3, 0.7), "包含特殊符号"),
        ("赵六", (0.6, 1.0), "2字中文昵称"),
        ("孙七八九十一二三四五", (0.4, 0.7), "长中文昵称"),
        ("abc", (0.3, 0.6), "纯英文"),
        ("", (0.0, 0.5), "空字符串"),
    ]
    
    passed = 0
    failed = 0
    
    for text, (min_score, max_score), description in test_cases:
        score = reader._calculate_nickname_confidence(text)
        
        # 检查分数是否在0.0-1.0范围内
        if not (0.0 <= score <= 1.0):
            print(f"✗ {description}: '{text}' -> {score:.2f} (超出范围!)")
            failed += 1
            continue
        
        # 检查分数是否在期望范围内
        if min_score <= score <= max_score:
            print(f"✓ {description}: '{text}' -> {score:.2f} (期望: {min_score:.2f}-{max_score:.2f})")
            passed += 1
        else:
            print(f"✗ {description}: '{text}' -> {score:.2f} (期望: {min_score:.2f}-{max_score:.2f})")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    # 测试位置加分
    print("\n测试位置加分功能:")
    print("-" * 60)
    
    # 测试靠近中心的文本
    position_info_near = {
        'center_x': 200,
        'center_y': 100,
        'region_center_x': 210,
        'region_center_y': 105
    }
    
    # 测试远离中心的文本
    position_info_far = {
        'center_x': 200,
        'center_y': 100,
        'region_center_x': 300,
        'region_center_y': 200
    }
    
    score_near = reader._calculate_nickname_confidence("张三", position_info_near)
    score_far = reader._calculate_nickname_confidence("张三", position_info_far)
    score_no_pos = reader._calculate_nickname_confidence("张三", None)
    
    print(f"靠近中心: {score_near:.2f}")
    print(f"远离中心: {score_far:.2f}")
    print(f"无位置信息: {score_no_pos:.2f}")
    
    if score_near > score_far:
        print("✓ 位置加分功能正常")
    else:
        print("✗ 位置加分功能异常")
    
    # 测试辅助方法
    print("\n测试辅助方法:")
    print("-" * 60)
    
    # 测试中文检测
    assert reader._is_chinese_text("张三") == True, "中文检测失败"
    assert reader._is_chinese_text("123") == False, "中文检测失败"
    assert reader._is_chinese_text("张三123") == True, "中文检测失败"
    print("✓ _is_chinese_text 正常")
    
    # 测试纯数字检测
    assert reader._is_pure_number("123") == True, "纯数字检测失败"
    assert reader._is_pure_number("12.3") == False, "纯数字检测失败"
    assert reader._is_pure_number("1a3") == False, "纯数字检测失败"
    print("✓ _is_pure_number 正常")
    
    # 测试纯符号检测
    assert reader._is_pure_symbol("@#$") == True, "纯符号检测失败"
    assert reader._is_pure_symbol("a@#") == False, "纯符号检测失败"
    assert reader._is_pure_symbol("123") == False, "纯符号检测失败"
    print("✓ _is_pure_symbol 正常")
    
    # 测试中文字符检测
    assert reader._is_chinese_char("张") == True, "中文字符检测失败"
    assert reader._is_chinese_char("a") == False, "中文字符检测失败"
    assert reader._is_chinese_char("1") == False, "中文字符检测失败"
    print("✓ _is_chinese_char 正常")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = test_confidence_scoring()
    
    # 如果有失败的测试，退出码为1
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
