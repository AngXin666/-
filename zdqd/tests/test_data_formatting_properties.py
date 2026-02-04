"""
数据格式化属性测试

使用 hypothesis 进行基于属性的测试，验证 ConciseLogger 类的数据格式化功能。
测试余额、积分、数据字典显示和错误堆栈信息记录。
"""

import re
import traceback
from unittest.mock import Mock
from hypothesis import given, strategies as st, assume

from zdqd.src.concise_logger import ConciseLogger, LogFormatter


# ============================================================================
# Property 9: 余额数据格式
# **Validates: Requirements 3.1**
# ============================================================================

@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    ),
    balance=st.floats(min_value=0.0, max_value=999999.99, allow_nan=False, allow_infinity=False)
)
def test_balance_data_format(description, balance):
    """
    Property 9: 余额数据格式
    
    验证：对于任意余额数值，当通过 success 方法的 data 参数传入"余额"键时，
    GUI 输出应该包含"余额: {数值:.2f}元"的格式
    
    **Validates: Requirements 3.1**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    # 格式化余额为两位小数
    balance_str = f"{balance:.2f}元"
    data = {"余额": balance_str}
    
    logger.success(description, data)
    
    # 验证 GUI 日志器被调用
    gui_logger.info.assert_called_once()
    gui_call_args = gui_logger.info.call_args[0][0]
    
    # 验证包含余额数据，格式为 "余额: XX.XX元"
    assert "余额: " in gui_call_args, "GUI 日志应包含'余额: '标识"
    assert balance_str in gui_call_args, f"GUI 日志应包含余额值 '{balance_str}'"
    
    # 验证完整格式
    expected_data_part = f"余额: {balance_str}"
    assert expected_data_part in gui_call_args, f"GUI 日志应包含完整的余额格式 '{expected_data_part}'"
    
    # 验证余额值是两位小数格式
    balance_pattern = r"余额: \d+\.\d{2}元"
    assert re.search(balance_pattern, gui_call_args), "余额应该是两位小数格式（XX.XX元）"


@given(
    balance1=st.floats(min_value=0.0, max_value=9999.99, allow_nan=False, allow_infinity=False),
    balance2=st.floats(min_value=0.0, max_value=9999.99, allow_nan=False, allow_infinity=False)
)
def test_balance_format_consistency(balance1, balance2):
    """
    Property 9: 余额数据格式一致性
    
    验证：不同的余额值应该使用相同的格式模式
    
    **Validates: Requirements 3.1**
    """
    gui_logger1 = Mock()
    gui_logger2 = Mock()
    logger1 = ConciseLogger("test", gui_logger1, None)
    logger2 = ConciseLogger("test", gui_logger2, None)
    
    balance1_str = f"{balance1:.2f}元"
    balance2_str = f"{balance2:.2f}元"
    
    logger1.success("测试1", {"余额": balance1_str})
    logger2.success("测试2", {"余额": balance2_str})
    
    result1 = gui_logger1.info.call_args[0][0]
    result2 = gui_logger2.info.call_args[0][0]
    
    # 提取余额部分的格式
    pattern = r"余额: \d+\.\d{2}元"
    match1 = re.search(pattern, result1)
    match2 = re.search(pattern, result2)
    
    assert match1 is not None, "第一个余额应该匹配格式"
    assert match2 is not None, "第二个余额应该匹配格式"
    
    # 验证格式结构一致（都是"余额: "开头，"元"结尾）
    assert "余额: " in result1 and "余额: " in result2, "格式应该一致"


@given(
    balance=st.floats(min_value=0.0, max_value=999999.99, allow_nan=False, allow_infinity=False)
)
def test_balance_format_in_formatter(balance):
    """
    Property 9: LogFormatter 中的余额格式
    
    验证：LogFormatter.format_success 方法正确格式化余额数据
    
    **Validates: Requirements 3.1**
    """
    balance_str = f"{balance:.2f}元"
    data = {"余额": balance_str}
    
    result = LogFormatter.format_success("获取资料完成", data)
    
    # 验证包含余额格式
    assert f"余额: {balance_str}" in result, "应该包含正确的余额格式"
    
    # 验证整体格式
    assert result.startswith("  ✓ "), "应该以成功标记开头"
    assert "(" in result and ")" in result, "数据应该在括号中"


# ============================================================================
# Property 10: 积分数据格式
# **Validates: Requirements 3.2**
# ============================================================================

@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    ),
    points=st.integers(min_value=0, max_value=999999)
)
def test_points_data_format(description, points):
    """
    Property 10: 积分数据格式
    
    验证：对于任意积分数值，当通过 success 方法的 data 参数传入"积分"键时，
    GUI 输出应该包含"积分: {数值}"的格式
    
    **Validates: Requirements 3.2**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    points_str = str(points)
    data = {"积分": points_str}
    
    logger.success(description, data)
    
    # 验证 GUI 日志器被调用
    gui_logger.info.assert_called_once()
    gui_call_args = gui_logger.info.call_args[0][0]
    
    # 验证包含积分数据，格式为 "积分: XXX"
    assert "积分: " in gui_call_args, "GUI 日志应包含'积分: '标识"
    assert points_str in gui_call_args, f"GUI 日志应包含积分值 '{points_str}'"
    
    # 验证完整格式
    expected_data_part = f"积分: {points_str}"
    assert expected_data_part in gui_call_args, f"GUI 日志应包含完整的积分格式 '{expected_data_part}'"
    
    # 验证积分值是整数格式（不包含小数点）
    points_pattern = r"积分: \d+"
    assert re.search(points_pattern, gui_call_args), "积分应该是整数格式"


@given(
    points1=st.integers(min_value=0, max_value=99999),
    points2=st.integers(min_value=0, max_value=99999)
)
def test_points_format_consistency(points1, points2):
    """
    Property 10: 积分数据格式一致性
    
    验证：不同的积分值应该使用相同的格式模式
    
    **Validates: Requirements 3.2**
    """
    gui_logger1 = Mock()
    gui_logger2 = Mock()
    logger1 = ConciseLogger("test", gui_logger1, None)
    logger2 = ConciseLogger("test", gui_logger2, None)
    
    logger1.success("测试1", {"积分": str(points1)})
    logger2.success("测试2", {"积分": str(points2)})
    
    result1 = gui_logger1.info.call_args[0][0]
    result2 = gui_logger2.info.call_args[0][0]
    
    # 提取积分部分的格式
    pattern = r"积分: \d+"
    match1 = re.search(pattern, result1)
    match2 = re.search(pattern, result2)
    
    assert match1 is not None, "第一个积分应该匹配格式"
    assert match2 is not None, "第二个积分应该匹配格式"
    
    # 验证格式结构一致（都是"积分: "开头）
    assert "积分: " in result1 and "积分: " in result2, "格式应该一致"


@given(
    points=st.integers(min_value=0, max_value=999999)
)
def test_points_format_in_formatter(points):
    """
    Property 10: LogFormatter 中的积分格式
    
    验证：LogFormatter.format_success 方法正确格式化积分数据
    
    **Validates: Requirements 3.2**
    """
    points_str = str(points)
    data = {"积分": points_str}
    
    result = LogFormatter.format_success("获取资料完成", data)
    
    # 验证包含积分格式
    assert f"积分: {points_str}" in result, "应该包含正确的积分格式"
    
    # 验证整体格式
    assert result.startswith("  ✓ "), "应该以成功标记开头"
    assert "(" in result and ")" in result, "数据应该在括号中"


@given(
    balance=st.floats(min_value=0.0, max_value=9999.99, allow_nan=False, allow_infinity=False),
    points=st.integers(min_value=0, max_value=99999)
)
def test_balance_and_points_together(balance, points):
    """
    Property 9 & 10: 余额和积分同时显示
    
    验证：当余额和积分同时存在时，两者都应该正确格式化并显示
    
    **Validates: Requirements 3.1, 3.2**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    balance_str = f"{balance:.2f}元"
    points_str = str(points)
    data = {"余额": balance_str, "积分": points_str}
    
    logger.success("获取资料完成", data)
    
    gui_call_args = gui_logger.info.call_args[0][0]
    
    # 验证同时包含余额和积分
    assert f"余额: {balance_str}" in gui_call_args, "应该包含余额"
    assert f"积分: {points_str}" in gui_call_args, "应该包含积分"
    
    # 验证用逗号分隔
    assert ", " in gui_call_args, "多个数据项应该用逗号分隔"


# ============================================================================
# Property 13: 数据字典完整显示
# **Validates: Requirements 3.5**
# ============================================================================

@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
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
        max_size=10
    )
)
def test_data_dictionary_complete_display(description, data):
    """
    Property 13: 数据字典完整显示
    
    验证：对于任意数据字典，当通过 success 方法的 data 参数传入时，
    GUI 输出应该包含字典中的所有键值对，格式为"key: value"并用逗号分隔
    
    **Validates: Requirements 3.5**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    logger.success(description, data)
    
    # 验证 GUI 日志器被调用
    gui_logger.info.assert_called_once()
    gui_call_args = gui_logger.info.call_args[0][0]
    
    # 验证所有键值对都在输出中
    for key, value in data.items():
        expected_pair = f"{key}: {value}"
        assert expected_pair in gui_call_args, f"GUI 日志应包含键值对 '{expected_pair}'"
    
    # 验证数据在括号中
    assert "(" in gui_call_args, "数据应该在左括号中"
    assert ")" in gui_call_args, "数据应该在右括号中"
    
    # 验证多个数据项用逗号和空格分隔
    if len(data) > 1:
        assert ", " in gui_call_args, "多个数据项应该用', '分隔"
    
    # 验证数据项数量正确
    data_count = len(data)
    # 计算逗号数量（应该是数据项数量 - 1）
    comma_count = gui_call_args.count(", ")
    # 注意：如果描述中也包含逗号，需要只计算括号内的逗号
    if "(" in gui_call_args:
        data_part_start = gui_call_args.index("(")
        data_part_end = gui_call_args.rindex(")")
        data_part = gui_call_args[data_part_start:data_part_end]
        comma_count_in_data = data_part.count(", ")
        if data_count > 1:
            assert comma_count_in_data == data_count - 1, f"数据项之间应该有 {data_count - 1} 个逗号分隔符"


@given(
    data=st.dictionaries(
        keys=st.text(
            alphabet=st.characters(blacklist_characters="\n\r\t:,()"),
            min_size=1,
            max_size=15
        ),
        values=st.text(
            alphabet=st.characters(blacklist_characters="\n\r\t,()"),
            min_size=1,
            max_size=30
        ),
        min_size=2,
        max_size=5
    )
)
def test_data_dictionary_order_preservation(data):
    """
    Property 13: 数据字典顺序保持
    
    验证：数据字典中的所有键值对都应该出现在输出中（顺序可能不同，但都应该存在）
    
    **Validates: Requirements 3.5**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    logger.success("测试", data)
    
    gui_call_args = gui_logger.info.call_args[0][0]
    
    # 提取括号内的数据部分
    data_start = gui_call_args.index("(") + 1
    data_end = gui_call_args.rindex(")")
    data_str = gui_call_args[data_start:data_end]
    
    # 验证所有键都在数据字符串中
    for key in data.keys():
        assert key in data_str, f"数据字符串应包含键 '{key}'"
    
    # 验证所有值都在数据字符串中
    for value in data.values():
        assert value in data_str, f"数据字符串应包含值 '{value}'"
    
    # 验证键值对数量
    # 通过计算冒号数量来验证（每个键值对有一个冒号）
    colon_count = data_str.count(": ")
    assert colon_count == len(data), f"应该有 {len(data)} 个键值对（冒号数量）"


@given(
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
        max_size=10
    )
)
def test_data_dictionary_format_in_formatter(data):
    """
    Property 13: LogFormatter 中的数据字典格式
    
    验证：LogFormatter.format_success 方法正确格式化数据字典
    
    **Validates: Requirements 3.5**
    """
    result = LogFormatter.format_success("操作完成", data)
    
    # 验证所有键值对都在输出中
    for key, value in data.items():
        assert f"{key}: {value}" in result, f"应该包含键值对 '{key}: {value}'"
    
    # 验证格式
    assert result.startswith("  ✓ "), "应该以成功标记开头"
    assert "(" in result and ")" in result, "数据应该在括号中"


@given(
    key_count=st.integers(min_value=1, max_value=10)
)
def test_data_dictionary_with_varying_sizes(key_count):
    """
    Property 13: 不同大小的数据字典
    
    验证：无论数据字典有多少个键值对，都应该完整显示
    
    **Validates: Requirements 3.5**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    # 创建指定大小的数据字典
    data = {f"键{i}": f"值{i}" for i in range(key_count)}
    
    logger.success("测试", data)
    
    gui_call_args = gui_logger.info.call_args[0][0]
    
    # 验证所有键值对都存在
    for key, value in data.items():
        assert f"{key}: {value}" in gui_call_args, f"应该包含键值对 '{key}: {value}'"
    
    # 验证分隔符数量
    if key_count > 1:
        data_part_start = gui_call_args.index("(")
        data_part_end = gui_call_args.rindex(")")
        data_part = gui_call_args[data_part_start:data_part_end]
        comma_count = data_part.count(", ")
        assert comma_count == key_count - 1, f"应该有 {key_count - 1} 个分隔符"


# ============================================================================
# Property 14: 错误堆栈信息记录
# **Validates: Requirements 9.5**
# ============================================================================

@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    ),
    error_message=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=100
    )
)
def test_error_stack_trace_logging(description, error_message):
    """
    Property 14: 错误堆栈信息记录
    
    验证：对于任意异常对象，当通过 error 方法传入时，
    文件日志应该包含完整的异常堆栈信息（包括异常类型、消息和调用栈）
    
    **Validates: Requirements 9.5**
    """
    file_logger = Mock()
    logger = ConciseLogger("test_module", None, file_logger)
    
    # 创建一个异常对象
    exception = Exception(error_message)
    
    logger.error(description, exception)
    
    # 验证文件日志器的 error 方法被调用
    file_logger.error.assert_called_once()
    
    # 获取调用参数
    call_args = file_logger.error.call_args
    
    # 验证第一个参数包含错误描述和异常消息
    error_msg = call_args[0][0]
    assert description in error_msg, "文件日志应包含错误描述"
    assert error_message in error_msg, "文件日志应包含异常消息"
    
    # 验证 exc_info=True 参数被传递（这会让 logging 记录完整堆栈）
    assert 'exc_info' in call_args[1], "应该传递 exc_info 参数"
    assert call_args[1]['exc_info'] == True, "exc_info 应该设置为 True 以记录堆栈信息"


@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_error_without_exception_no_stack_trace(description):
    """
    Property 14: 无异常时不记录堆栈
    
    验证：当 error 方法没有传入异常对象时，不应该设置 exc_info 参数
    
    **Validates: Requirements 9.5**
    """
    file_logger = Mock()
    logger = ConciseLogger("test_module", None, file_logger)
    
    logger.error(description)
    
    # 验证文件日志器被调用
    file_logger.error.assert_called_once()
    
    # 获取调用参数
    call_args = file_logger.error.call_args
    
    # 验证错误描述在消息中
    error_msg = call_args[0][0]
    assert description in error_msg, "文件日志应包含错误描述"
    
    # 验证没有传递 exc_info 参数（或者为 False）
    if len(call_args[1]) > 0:
        assert call_args[1].get('exc_info') != True, "没有异常时不应该设置 exc_info=True"


@given(
    error_message=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=100
    )
)
def test_different_exception_types_stack_trace(error_message):
    """
    Property 14: 不同异常类型的堆栈记录
    
    验证：不同类型的异常都应该正确记录堆栈信息
    
    **Validates: Requirements 9.5**
    """
    file_logger = Mock()
    logger = ConciseLogger("test_module", None, file_logger)
    
    # 测试不同类型的异常
    exception_types = [
        Exception(error_message),
        ValueError(error_message),
        RuntimeError(error_message),
        TypeError(error_message),
    ]
    
    for exception in exception_types:
        file_logger.reset_mock()
        
        logger.error("测试错误", exception)
        
        # 验证文件日志器被调用
        file_logger.error.assert_called_once()
        
        # 验证 exc_info=True
        call_args = file_logger.error.call_args
        assert call_args[1].get('exc_info') == True, f"{type(exception).__name__} 应该记录堆栈信息"
        
        # 验证异常消息在日志中
        error_msg = call_args[0][0]
        assert error_message in error_msg, f"{type(exception).__name__} 的消息应该在日志中"


@given(
    module_name=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t[]"),
        min_size=1,
        max_size=30
    ),
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    ),
    error_message=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=100
    )
)
def test_error_stack_trace_includes_module_info(module_name, description, error_message):
    """
    Property 14: 错误堆栈包含模块信息
    
    验证：文件日志中的错误信息应该包含模块名，便于定位问题
    
    **Validates: Requirements 9.5**
    """
    file_logger = Mock()
    logger = ConciseLogger(module_name, None, file_logger)
    
    exception = Exception(error_message)
    logger.error(description, exception)
    
    # 验证文件日志器被调用
    file_logger.error.assert_called_once()
    
    # 获取日志消息
    call_args = file_logger.error.call_args
    error_msg = call_args[0][0]
    
    # 验证包含模块名
    assert f"[{module_name}]" in error_msg, "错误日志应包含模块名"
    
    # 验证包含错误描述
    assert description in error_msg, "错误日志应包含错误描述"
    
    # 验证包含异常消息
    assert error_message in error_msg, "错误日志应包含异常消息"
    
    # 验证 exc_info=True
    assert call_args[1].get('exc_info') == True, "应该记录完整堆栈信息"


@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    ),
    error_message=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=100
    )
)
def test_gui_logger_does_not_show_stack_trace(description, error_message):
    """
    Property 14: GUI 日志不显示堆栈信息
    
    验证：即使文件日志记录了完整堆栈，GUI 日志也应该保持简洁，不显示堆栈
    
    **Validates: Requirements 9.5**
    """
    gui_logger = Mock()
    file_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, file_logger)
    
    exception = Exception(error_message)
    logger.error(description, exception)
    
    # 验证 GUI 日志器被调用
    gui_logger.error.assert_called_once()
    gui_call_args = gui_logger.error.call_args[0][0]
    
    # 验证 GUI 日志只包含简洁的错误描述，格式严格匹配
    expected_gui_format = f"  ✗ 错误: {description}"
    assert gui_call_args == expected_gui_format, "GUI 日志应该只包含简洁的错误描述"
    
    # 验证 GUI 日志不包含堆栈跟踪关键词
    assert "Traceback" not in gui_call_args, "GUI 日志不应该包含堆栈跟踪"
    assert "File " not in gui_call_args, "GUI 日志不应该包含文件路径信息"
    assert "line " not in gui_call_args, "GUI 日志不应该包含行号信息"
    
    # 验证文件日志包含异常信息
    file_logger.error.assert_called_once()
    file_call_args = file_logger.error.call_args
    file_msg = file_call_args[0][0]
    
    assert error_message in file_msg, "文件日志应该包含异常消息"
    assert file_call_args[1].get('exc_info') == True, "文件日志应该记录堆栈信息"
