"""
ConciseLogger 属性测试

使用 hypothesis 进行基于属性的测试，验证 ConciseLogger 类的双通道日志输出功能。
"""

import re
import logging
from unittest.mock import Mock, MagicMock, call
from hypothesis import given, strategies as st

from zdqd.src.concise_logger import ConciseLogger


# ============================================================================
# Property 1: 双通道日志输出
# **Validates: Requirements 1.2, 1.3**
# ============================================================================

@given(
    step_number=st.integers(min_value=1, max_value=100),
    title=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_dual_channel_logging_step(step_number, title):
    """
    Property 1: 双通道日志输出 - step 方法
    
    验证：对于任意步骤日志，当调用 ConciseLogger.step() 方法时，
    该消息应该同时输出到 GUI 日志器和文件日志器（如果它们都已配置）
    
    **Validates: Requirements 1.2, 1.3**
    """
    # 创建 Mock 日志器
    gui_logger = Mock()
    file_logger = Mock()
    
    # 创建 ConciseLogger 实例
    logger = ConciseLogger("test_module", gui_logger, file_logger)
    
    # 调用 step 方法
    logger.step(step_number, title)
    
    # 验证 GUI 日志器被调用
    gui_logger.info.assert_called_once()
    gui_call_args = gui_logger.info.call_args[0][0]
    assert f"步骤{step_number}: {title}" == gui_call_args
    
    # 验证文件日志器被调用
    file_logger.info.assert_called_once()
    file_call_args = file_logger.info.call_args[0][0]
    assert "[test_module]" in file_call_args
    assert str(step_number) in file_call_args
    assert title in file_call_args


@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_dual_channel_logging_action(description):
    """
    Property 1: 双通道日志输出 - action 方法
    
    验证：对于任意操作日志，当调用 ConciseLogger.action() 方法时，
    该消息应该同时输出到 GUI 日志器和文件日志器
    
    **Validates: Requirements 1.2, 1.3**
    """
    gui_logger = Mock()
    file_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, file_logger)
    
    logger.action(description)
    
    # 验证 GUI 日志器被调用
    gui_logger.info.assert_called_once()
    gui_call_args = gui_logger.info.call_args[0][0]
    assert f"  → {description}" == gui_call_args
    
    # 验证文件日志器被调用
    file_logger.info.assert_called_once()
    file_call_args = file_logger.info.call_args[0][0]
    assert "[test_module]" in file_call_args
    assert description in file_call_args


@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_dual_channel_logging_success(description):
    """
    Property 1: 双通道日志输出 - success 方法
    
    验证：对于任意成功日志，当调用 ConciseLogger.success() 方法时，
    该消息应该同时输出到 GUI 日志器和文件日志器
    
    **Validates: Requirements 1.2, 1.3**
    """
    gui_logger = Mock()
    file_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, file_logger)
    
    logger.success(description)
    
    # 验证 GUI 日志器被调用
    gui_logger.info.assert_called_once()
    gui_call_args = gui_logger.info.call_args[0][0]
    assert f"  ✓ {description}" == gui_call_args
    
    # 验证文件日志器被调用
    file_logger.info.assert_called_once()
    file_call_args = file_logger.info.call_args[0][0]
    assert "[test_module]" in file_call_args
    assert description in file_call_args



@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_dual_channel_logging_error(description):
    """
    Property 1: 双通道日志输出 - error 方法
    
    验证：对于任意错误日志，当调用 ConciseLogger.error() 方法时，
    该消息应该同时输出到 GUI 日志器和文件日志器
    
    **Validates: Requirements 1.2, 1.3**
    """
    gui_logger = Mock()
    file_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, file_logger)
    
    logger.error(description)
    
    # 验证 GUI 日志器被调用
    gui_logger.error.assert_called_once()
    gui_call_args = gui_logger.error.call_args[0][0]
    assert f"  ✗ 错误: {description}" == gui_call_args
    
    # 验证文件日志器被调用
    file_logger.error.assert_called_once()
    file_call_args = file_logger.error.call_args[0][0]
    assert "[test_module]" in file_call_args
    assert description in file_call_args


@given(
    module_name=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t[]"),
        min_size=1,
        max_size=30
    ),
    step_number=st.integers(min_value=1, max_value=100),
    title=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_dual_channel_with_both_loggers_configured(module_name, step_number, title):
    """
    Property 1: 双通道日志输出 - 两个日志器都配置时
    
    验证：当 GUI 日志器和文件日志器都配置时，日志消息应该同时输出到两个通道
    
    **Validates: Requirements 1.2, 1.3**
    """
    gui_logger = Mock()
    file_logger = Mock()
    logger = ConciseLogger(module_name, gui_logger, file_logger)
    
    logger.step(step_number, title)
    
    # 验证两个日志器都被调用
    assert gui_logger.info.called, "GUI 日志器应该被调用"
    assert file_logger.info.called, "文件日志器应该被调用"
    
    # 验证调用次数
    assert gui_logger.info.call_count == 1, "GUI 日志器应该被调用一次"
    assert file_logger.info.call_count == 1, "文件日志器应该被调用一次"



@given(
    step_number=st.integers(min_value=1, max_value=100),
    title=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_dual_channel_with_only_gui_logger(step_number, title):
    """
    Property 1: 双通道日志输出 - 只配置 GUI 日志器时
    
    验证：当只配置 GUI 日志器时，日志消息应该只输出到 GUI 通道
    
    **Validates: Requirements 1.2, 1.3**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    logger.step(step_number, title)
    
    # 验证 GUI 日志器被调用
    assert gui_logger.info.called, "GUI 日志器应该被调用"


@given(
    step_number=st.integers(min_value=1, max_value=100),
    title=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_dual_channel_with_only_file_logger(step_number, title):
    """
    Property 1: 双通道日志输出 - 只配置文件日志器时
    
    验证：当只配置文件日志器时，日志消息应该只输出到文件通道
    
    **Validates: Requirements 1.2, 1.3**
    """
    file_logger = Mock()
    logger = ConciseLogger("test_module", None, file_logger)
    
    logger.step(step_number, title)
    
    # 验证文件日志器被调用
    assert file_logger.info.called, "文件日志器应该被调用"


# ============================================================================
# Property 2: 文件日志包含完整信息
# **Validates: Requirements 1.4**
# ============================================================================

@given(
    module_name=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t[]"),
        min_size=1,
        max_size=30
    ),
    step_number=st.integers(min_value=1, max_value=100),
    title=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_file_log_contains_complete_info_step(module_name, step_number, title):
    """
    Property 2: 文件日志包含完整信息 - step 方法
    
    验证：对于任意步骤日志，文件日志输出应该包含模块名和详细消息
    （注：时间戳和日志级别由 logging 框架自动添加）
    
    **Validates: Requirements 1.4**
    """
    file_logger = Mock()
    logger = ConciseLogger(module_name, None, file_logger)
    
    logger.step(step_number, title)
    
    # 验证文件日志器被调用
    file_logger.info.assert_called_once()
    file_call_args = file_logger.info.call_args[0][0]
    
    # 验证包含模块名（用方括号包围）
    assert f"[{module_name}]" in file_call_args, "文件日志应包含模块名"
    
    # 验证包含详细消息
    assert str(step_number) in file_call_args, "文件日志应包含步骤编号"
    assert title in file_call_args, "文件日志应包含步骤标题"



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
    )
)
def test_file_log_contains_complete_info_action(module_name, description):
    """
    Property 2: 文件日志包含完整信息 - action 方法
    
    验证：对于任意操作日志，文件日志输出应该包含模块名和详细消息
    
    **Validates: Requirements 1.4**
    """
    file_logger = Mock()
    logger = ConciseLogger(module_name, None, file_logger)
    
    logger.action(description)
    
    file_logger.info.assert_called_once()
    file_call_args = file_logger.info.call_args[0][0]
    
    # 验证包含模块名
    assert f"[{module_name}]" in file_call_args, "文件日志应包含模块名"
    
    # 验证包含详细消息
    assert description in file_call_args, "文件日志应包含操作描述"
    assert "执行操作" in file_call_args, "文件日志应包含操作类型标识"


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
    data=st.dictionaries(
        keys=st.text(
            alphabet=st.characters(blacklist_characters="\n\r\t:,()=[]"),
            min_size=1,
            max_size=20
        ),
        values=st.text(
            alphabet=st.characters(blacklist_characters="\n\r\t,()[]"),
            min_size=1,
            max_size=30
        ),
        min_size=1,
        max_size=3
    )
)
def test_file_log_contains_complete_info_success_with_data(module_name, description, data):
    """
    Property 2: 文件日志包含完整信息 - success 方法（带数据）
    
    验证：对于任意成功日志（带数据），文件日志输出应该包含模块名、详细消息和数据
    
    **Validates: Requirements 1.4**
    """
    file_logger = Mock()
    logger = ConciseLogger(module_name, None, file_logger)
    
    logger.success(description, data)
    
    file_logger.info.assert_called_once()
    file_call_args = file_logger.info.call_args[0][0]
    
    # 验证包含模块名
    assert f"[{module_name}]" in file_call_args, "文件日志应包含模块名"
    
    # 验证包含详细消息
    assert description in file_call_args, "文件日志应包含成功描述"
    assert "成功" in file_call_args, "文件日志应包含成功标识"
    
    # 验证包含数据
    for key, value in data.items():
        # 文件日志使用 key=value 格式
        assert key in file_call_args, f"文件日志应包含数据键 '{key}'"
        assert value in file_call_args, f"文件日志应包含数据值 '{value}'"



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
    )
)
def test_file_log_contains_complete_info_error(module_name, description):
    """
    Property 2: 文件日志包含完整信息 - error 方法
    
    验证：对于任意错误日志，文件日志输出应该包含模块名和详细消息
    
    **Validates: Requirements 1.4**
    """
    file_logger = Mock()
    logger = ConciseLogger(module_name, None, file_logger)
    
    logger.error(description)
    
    file_logger.error.assert_called_once()
    file_call_args = file_logger.error.call_args[0][0]
    
    # 验证包含模块名
    assert f"[{module_name}]" in file_call_args, "文件日志应包含模块名"
    
    # 验证包含详细消息
    assert description in file_call_args, "文件日志应包含错误描述"
    assert "错误" in file_call_args, "文件日志应包含错误标识"


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
        max_size=50
    )
)
def test_file_log_contains_exception_info(module_name, description, error_message):
    """
    Property 2: 文件日志包含完整信息 - error 方法（带异常）
    
    验证：对于任意错误日志（带异常），文件日志输出应该包含模块名、详细消息和异常信息
    
    **Validates: Requirements 1.4**
    """
    file_logger = Mock()
    logger = ConciseLogger(module_name, None, file_logger)
    
    # 创建一个异常对象
    exception = Exception(error_message)
    
    logger.error(description, exception)
    
    file_logger.error.assert_called_once()
    
    # 验证调用参数
    call_args = file_logger.error.call_args
    file_call_args = call_args[0][0]
    
    # 验证包含模块名
    assert f"[{module_name}]" in file_call_args, "文件日志应包含模块名"
    
    # 验证包含详细消息
    assert description in file_call_args, "文件日志应包含错误描述"
    
    # 验证包含异常信息
    assert error_message in file_call_args, "文件日志应包含异常消息"
    
    # 验证 exc_info=True 参数被传递（用于记录堆栈信息）
    assert call_args[1].get('exc_info') == True, "文件日志应该设置 exc_info=True 以记录堆栈信息"


# ============================================================================
# Property 3: GUI日志简洁性
# **Validates: Requirements 1.5**
# ============================================================================

@given(
    step_number=st.integers(min_value=1, max_value=100),
    title=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_gui_log_conciseness_step(step_number, title):
    """
    Property 3: GUI日志简洁性 - step 方法
    
    验证：对于任意 GUI 日志输出，它应该不包含时间戳、模块名、日志级别等技术信息，
    只包含用户可读的简洁内容
    
    **Validates: Requirements 1.5**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    logger.step(step_number, title)
    
    gui_logger.info.assert_called_once()
    gui_call_args = gui_logger.info.call_args[0][0]
    
    # 验证不包含模块名（用方括号包围）
    assert "[test_module]" not in gui_call_args, "GUI 日志不应包含模块名"
    
    # 验证只包含简洁的步骤信息（这是最直接的验证方式）
    assert gui_call_args == f"步骤{step_number}: {title}", "GUI 日志应该只包含简洁的步骤信息"
    
    # 验证格式简洁：以"步骤"开头，不包含技术性前缀
    assert gui_call_args.startswith("步骤"), "GUI 日志应该以'步骤'开头"



@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_gui_log_conciseness_action(description):
    """
    Property 3: GUI日志简洁性 - action 方法
    
    验证：对于任意操作日志，GUI 输出应该不包含技术信息，只包含简洁的操作描述
    
    **Validates: Requirements 1.5**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    logger.action(description)
    
    gui_logger.info.assert_called_once()
    gui_call_args = gui_logger.info.call_args[0][0]
    
    # 验证不包含模块名
    assert "[test_module]" not in gui_call_args, "GUI 日志不应包含模块名"
    
    # 验证不包含技术性词汇
    assert "执行操作" not in gui_call_args, "GUI 日志不应包含技术性词汇"
    
    # 验证只包含简洁的操作信息
    assert gui_call_args == f"  → {description}", "GUI 日志应该只包含简洁的操作信息"


@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    ),
    data=st.dictionaries(
        keys=st.text(
            alphabet=st.characters(blacklist_characters="\n\r\t:,()=[]"),
            min_size=1,
            max_size=20
        ),
        values=st.text(
            alphabet=st.characters(blacklist_characters="\n\r\t,()[]"),
            min_size=1,
            max_size=30
        ),
        min_size=1,
        max_size=3
    )
)
def test_gui_log_conciseness_success(description, data):
    """
    Property 3: GUI日志简洁性 - success 方法
    
    验证：对于任意成功日志，GUI 输出应该不包含技术信息，只包含简洁的成功描述和数据
    
    **Validates: Requirements 1.5**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    logger.success(description, data)
    
    gui_logger.info.assert_called_once()
    gui_call_args = gui_logger.info.call_args[0][0]
    
    # 验证不包含模块名
    assert "[test_module]" not in gui_call_args, "GUI 日志不应包含模块名"
    
    # 验证不包含技术性词汇
    assert "成功:" not in gui_call_args or gui_call_args.startswith("  ✓"), "GUI 日志不应包含技术性格式"
    
    # 验证使用用户友好的格式（key: value 而不是 key=value）
    for key, value in data.items():
        assert f"{key}: {value}" in gui_call_args, "GUI 日志应使用用户友好的 'key: value' 格式"
        assert f"{key}={value}" not in gui_call_args, "GUI 日志不应使用技术性的 'key=value' 格式"


@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_gui_log_conciseness_error(description):
    """
    Property 3: GUI日志简洁性 - error 方法
    
    验证：对于任意错误日志，GUI 输出应该不包含技术信息，只包含简洁的错误描述
    
    **Validates: Requirements 1.5**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    logger.error(description)
    
    gui_logger.error.assert_called_once()
    gui_call_args = gui_logger.error.call_args[0][0]
    
    # 验证不包含模块名
    assert "[test_module]" not in gui_call_args, "GUI 日志不应包含模块名"
    
    # 验证只包含简洁的错误信息
    assert gui_call_args == f"  ✗ 错误: {description}", "GUI 日志应该只包含简洁的错误信息"


@given(
    description=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    ),
    error_message=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_gui_log_conciseness_error_with_exception(description, error_message):
    """
    Property 3: GUI日志简洁性 - error 方法（带异常）
    
    验证：即使传入了异常对象，GUI 输出也应该保持简洁，不显示堆栈信息
    
    **Validates: Requirements 1.5**
    """
    gui_logger = Mock()
    logger = ConciseLogger("test_module", gui_logger, None)
    
    exception = Exception(error_message)
    logger.error(description, exception)
    
    gui_logger.error.assert_called_once()
    gui_call_args = gui_logger.error.call_args[0][0]
    
    # 验证不包含模块名
    assert "[test_module]" not in gui_call_args, "GUI 日志不应包含模块名"
    
    # 验证不包含异常详细信息（异常信息应该只在文件日志中）
    # GUI 只显示用户提供的简洁描述
    assert gui_call_args == f"  ✗ 错误: {description}", "GUI 日志应该只包含简洁的错误描述"
    
    # 验证 GUI 日志不包含堆栈跟踪信息
    assert "Traceback" not in gui_call_args, "GUI 日志不应包含堆栈跟踪"
    assert "File" not in gui_call_args or not gui_call_args.startswith("  File"), "GUI 日志不应包含文件路径信息"


@given(
    module_name=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t[]"),
        min_size=1,
        max_size=30
    ),
    message=st.text(
        alphabet=st.characters(blacklist_characters="\n\r\t"),
        min_size=1,
        max_size=50
    )
)
def test_gui_log_not_called_for_debug(module_name, message):
    """
    Property 3: GUI日志简洁性 - debug 方法
    
    验证：调试信息应该只记录到文件日志，不应该输出到 GUI
    
    **Validates: Requirements 1.5**
    """
    gui_logger = Mock()
    file_logger = Mock()
    logger = ConciseLogger(module_name, gui_logger, file_logger)
    
    logger.debug(message)
    
    # 验证 GUI 日志器没有被调用
    assert not gui_logger.info.called, "debug 方法不应该调用 GUI 日志器的 info"
    assert not gui_logger.debug.called, "debug 方法不应该调用 GUI 日志器的 debug"
    assert not gui_logger.error.called, "debug 方法不应该调用 GUI 日志器的 error"
    
    # 验证文件日志器被调用
    file_logger.debug.assert_called_once()
    file_call_args = file_logger.debug.call_args[0][0]
    
    # 验证文件日志包含模块名和消息
    assert f"[{module_name}]" in file_call_args, "文件日志应包含模块名"
    assert message in file_call_args, "文件日志应包含调试消息"
