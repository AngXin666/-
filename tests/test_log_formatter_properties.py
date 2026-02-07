"""
LogFormatter 属性测试

使用 hypothesis 进行基于属性的测试，验证 LogFormatter 类的格式化方法
在所有输入下都保持一致的格式。
"""

import re
from hypothesis import given, strategies as st

from zdqd.src.concise_logger import LogFormatter


# ============================================================================
# Property 4: 步骤日志格式一致性
# **Validates: Requirements 2.1**
# ============================================================================

@given(
    step_number=st.integers(min_value=1, max_value=1000),
    title=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=100
    )
)
def test_step_format_consistency(step_number, title):
    """
    Property 4: 步骤日志格式一致性
    
    验证：对于任意步骤编号和标题，调用 format_step 方法后的输出
    应该严格匹配 "步骤{编号}: {标题}" 的格式
    
    **Validates: Requirements 2.1**
    """
    result = LogFormatter.format_step(step_number, title)
    
    # 验证格式：步骤{编号}: {标题}
    expected = f"步骤{step_number}: {title}"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    
    # 验证格式结构：必须以"步骤"开头，包含冒号和空格
    assert result.startswith("步骤"), "步骤日志必须以'步骤'开头"
    assert ": " in result, "步骤日志必须包含': '分隔符"
    
    # 验证步骤编号在正确位置
    pattern = r"^步骤(\d+): (.*)$"
    match = re.match(pattern, result, re.DOTALL)
    assert match is not None, f"步骤日志格式不匹配: {result}"
    assert int(match.group(1)) == step_number, "步骤编号不正确"
    assert match.group(2) == title, "步骤标题不正确"


# ============================================================================
# Property 5: 操作日志格式一致性
# **Validates: Requirements 2.2**
# ============================================================================

@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=100
    )
)
def test_action_format_consistency(description):
    """
    Property 5: 操作日志格式一致性
    
    验证：对于任意操作描述，调用 format_action 方法后的输出
    应该严格匹配 "  → {描述}" 的格式（两个空格+箭头+空格+描述）
    
    **Validates: Requirements 2.2**
    """
    result = LogFormatter.format_action(description)
    
    # 验证格式：  → {描述}
    expected = f"  → {description}"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    
    # 验证缩进：必须以两个空格开头
    assert result.startswith("  "), "操作日志必须以两个空格开头"
    
    # 验证箭头符号：第3个字符必须是箭头
    assert result[2] == "→", "操作日志第3个字符必须是箭头符号 '→'"
    
    # 验证空格：箭头后必须有一个空格
    assert result[3] == " ", "箭头符号后必须有一个空格"
    
    # 验证描述内容
    assert result[4:] == description, "操作描述内容不正确"


# ============================================================================
# Property 6: 成功日志格式一致性
# **Validates: Requirements 2.3**
# ============================================================================

@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=100
    )
)
def test_success_format_consistency_without_data(description):
    """
    Property 6: 成功日志格式一致性（无数据）
    
    验证：对于任意完成描述（无数据），调用 format_success 方法后的输出
    应该严格匹配 "  ✓ {描述}" 的格式（两个空格+勾号+空格+描述）
    
    **Validates: Requirements 2.3**
    """
    result = LogFormatter.format_success(description)
    
    # 验证格式：  ✓ {描述}
    expected = f"  ✓ {description}"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    
    # 验证缩进：必须以两个空格开头
    assert result.startswith("  "), "成功日志必须以两个空格开头"
    
    # 验证勾号符号：第3个字符必须是勾号
    assert result[2] == "✓", "成功日志第3个字符必须是勾号符号 '✓'"
    
    # 验证空格：勾号后必须有一个空格
    assert result[3] == " ", "勾号符号后必须有一个空格"
    
    # 验证描述内容
    assert result[4:] == description, "成功描述内容不正确"


@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=100
    ),
    data=st.dictionaries(
        keys=st.text(
            alphabet=st.characters(blacklist_characters="\n\r\t:,()"),
            min_size=1,
            max_size=20
        ),
        values=st.text(
            alphabet=st.characters(blacklist_characters="\n\r\t,()"),
            min_size=1,
            max_size=50
        ),
        min_size=1,
        max_size=5
    )
)
def test_success_format_consistency_with_data(description, data):
    """
    Property 6: 成功日志格式一致性（带数据）
    
    验证：对于任意完成描述和数据字典，调用 format_success 方法后的输出
    应该严格匹配 "  ✓ {描述} ({key1}: {value1}, {key2}: {value2})" 的格式
    
    **Validates: Requirements 2.3**
    """
    result = LogFormatter.format_success(description, data)
    
    # 验证基本格式：  ✓ {描述} (...)
    assert result.startswith("  ✓ "), "成功日志必须以'  ✓ '开头"
    assert result.startswith(f"  ✓ {description}"), "成功日志必须包含描述"
    
    # 验证数据部分：必须包含括号
    assert " (" in result, "带数据的成功日志必须包含左括号"
    assert result.endswith(")"), "带数据的成功日志必须以右括号结尾"
    
    # 提取数据部分
    data_start = result.index(" (") + 2
    data_end = result.rindex(")")
    data_str = result[data_start:data_end]
    
    # 验证所有键值对都在数据字符串中
    for key, value in data.items():
        expected_pair = f"{key}: {value}"
        assert expected_pair in data_str, f"数据字符串中应包含 '{expected_pair}'"
    
    # 验证数据项之间用逗号和空格分隔
    if len(data) > 1:
        assert ", " in data_str, "多个数据项之间应该用', '分隔"


# ============================================================================
# Property 7: 错误日志格式一致性
# **Validates: Requirements 2.4, 9.1**
# ============================================================================

@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=100
    )
)
def test_error_format_consistency(description):
    """
    Property 7: 错误日志格式一致性
    
    验证：对于任意错误描述，调用 format_error 方法后的输出
    应该严格匹配 "  ✗ 错误: {描述}" 的格式
    
    **Validates: Requirements 2.4, 9.1**
    """
    result = LogFormatter.format_error(description)
    
    # 验证格式：  ✗ 错误: {描述}
    expected = f"  ✗ 错误: {description}"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    
    # 验证缩进：必须以两个空格开头
    assert result.startswith("  "), "错误日志必须以两个空格开头"
    
    # 验证叉号符号：第3个字符必须是叉号
    assert result[2] == "✗", "错误日志第3个字符必须是叉号符号 '✗'"
    
    # 验证固定文本："错误: "
    assert result[3:8] == " 错误: ", "错误日志必须包含' 错误: '文本"
    
    # 验证描述内容
    assert result[8:] == description, "错误描述内容不正确"


# ============================================================================
# Property 8: 缩进和符号一致性
# **Validates: Requirements 2.5**
# ============================================================================

@given(
    action_desc=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    ),
    success_desc=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    ),
    error_desc=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_indentation_and_symbol_consistency(action_desc, success_desc, error_desc):
    """
    Property 8: 缩进和符号一致性
    
    验证：对于任意操作、成功或错误日志，它们的缩进应该都是两个空格，
    并且使用一致的符号（→、✓、✗）
    
    **Validates: Requirements 2.5**
    """
    action_result = LogFormatter.format_action(action_desc)
    success_result = LogFormatter.format_success(success_desc)
    error_result = LogFormatter.format_error(error_desc)
    
    # 验证所有日志都以两个空格开头（一致的缩进）
    assert action_result.startswith("  "), "操作日志缩进不一致"
    assert success_result.startswith("  "), "成功日志缩进不一致"
    assert error_result.startswith("  "), "错误日志缩进不一致"
    
    # 验证缩进长度一致（都是2个空格）
    action_indent = len(action_result) - len(action_result.lstrip(" "))
    success_indent = len(success_result) - len(success_result.lstrip(" "))
    error_indent = len(error_result) - len(error_result.lstrip(" "))
    
    assert action_indent == 2, f"操作日志缩进应为2个空格，实际为{action_indent}"
    assert success_indent == 2, f"成功日志缩进应为2个空格，实际为{success_indent}"
    assert error_indent == 2, f"错误日志缩进应为2个空格，实际为{error_indent}"
    
    # 验证使用一致的符号
    assert action_result[2] == "→", "操作日志应使用箭头符号 '→'"
    assert success_result[2] == "✓", "成功日志应使用勾号符号 '✓'"
    assert error_result[2] == "✗", "错误日志应使用叉号符号 '✗'"
    
    # 验证符号后都有一个空格
    assert action_result[3] == " ", "操作日志符号后应有空格"
    assert success_result[3] == " ", "成功日志符号后应有空格"
    assert error_result[3] == " ", "错误日志符号后应有空格"


# ============================================================================
# 额外的边界测试
# ============================================================================

@given(step_number=st.integers(min_value=1, max_value=1))
def test_step_format_single_digit(step_number):
    """测试单位数步骤编号的格式"""
    result = LogFormatter.format_step(step_number, "测试")
    assert result == f"步骤{step_number}: 测试"


@given(step_number=st.integers(min_value=100, max_value=999))
def test_step_format_three_digits(step_number):
    """测试三位数步骤编号的格式"""
    result = LogFormatter.format_step(step_number, "测试")
    assert result == f"步骤{step_number}: 测试"


@given(description=st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll")), min_size=1, max_size=50))
def test_action_format_english_only(description):
    """测试纯英文描述的操作日志格式"""
    result = LogFormatter.format_action(description)
    assert result == f"  → {description}"


@given(description=st.text(alphabet="一二三四五六七八九十", min_size=1, max_size=20))
def test_success_format_chinese_only(description):
    """测试纯中文描述的成功日志格式"""
    result = LogFormatter.format_success(description)
    assert result == f"  ✓ {description}"


@given(description=st.text(alphabet=st.characters(whitelist_categories=("Nd",)), min_size=1, max_size=20))
def test_error_format_numbers_only(description):
    """测试纯数字描述的错误日志格式"""
    result = LogFormatter.format_error(description)
    assert result == f"  ✗ 错误: {description}"
