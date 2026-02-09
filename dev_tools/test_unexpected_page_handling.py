"""
测试主流程中非预期页面的处理逻辑

测试内容：
1. 文章页 - YOLO检测返回按钮 + 降级返回键
2. 搜索页 - YOLO检测返回按钮 + 降级返回键
3. 分类页 - YOLO检测首页按钮 + 降级默认坐标
4. 设置页 - YOLO检测返回按钮 + 降级返回键
5. 交易流水页 - YOLO检测返回按钮 + 降级返回键
6. 优惠券页 - YOLO检测返回按钮 + 降级返回键
7. 积分页 - YOLO检测返回按钮 + 降级返回键
8. 弹窗页面 - 关闭弹窗逻辑
9. 未知页面 - 返回键处理
"""

import sys
import os
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.page_state_guard import PageStateGuard
from src.page_state_dynamic import PageState
from src.adb_bridge import ADBBridge


class TestResult:
    """测试结果"""
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self):
        self.passed += 1
    
    def add_fail(self, error: str):
        self.failed += 1
        self.errors.append(error)
    
    def print_summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"测试: {self.name}")
        print(f"总计: {total} | 通过: {self.passed} | 失败: {self.failed}")
        if self.errors:
            print(f"\n失败详情:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        print(f"{'='*60}\n")


async def test_article_page_handling():
    """测试文章页处理逻辑"""
    result = TestResult("文章页处理")
    
    print("\n" + "="*60)
    print("测试: 文章页处理逻辑")
    print("="*60)
    
    # 创建mock对象
    mock_adb = Mock(spec=ADBBridge)
    mock_adb.tap = AsyncMock()
    mock_adb.press_back = AsyncMock()
    
    mock_detector = Mock()
    mock_detector.find_button_yolo = AsyncMock()
    mock_detector.detect_page = AsyncMock()
    
    guard = PageStateGuard(mock_adb, mock_detector)
    device_id = "test_device"
    
    # 测试: 文章页使用返回键（因为没有专用YOLO模型）
    print("\n[测试] 文章页使用返回键处理")
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.ARTICLE, 
            PageState.HOME, 
            "测试操作"
        )
        
        # 验证
        assert handled == True, "应该返回True表示处理成功"
        # 文章页没有传递model_name，所以不会调用YOLO，直接使用返回键
        assert mock_adb.press_back.called, "应该使用返回键"
        
        print("  ✓ 文章页正确使用返回键处理")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    result.print_summary()
    return result


async def test_search_page_handling():
    """测试搜索页处理逻辑"""
    result = TestResult("搜索页处理")
    
    print("\n" + "="*60)
    print("测试: 搜索页处理逻辑")
    print("="*60)
    
    mock_adb = Mock(spec=ADBBridge)
    mock_adb.tap = AsyncMock()
    mock_adb.press_back = AsyncMock()
    
    mock_detector = Mock()
    mock_detector.find_button_yolo = AsyncMock()
    
    guard = PageStateGuard(mock_adb, mock_detector)
    device_id = "test_device"
    
    # 测试: 搜索页使用返回键（因为没有专用YOLO模型）
    print("\n[测试] 搜索页使用返回键处理")
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.SEARCH, 
            PageState.HOME, 
            "测试操作"
        )
        
        assert handled == True, "应该返回True表示处理成功"
        assert mock_adb.press_back.called, "应该使用返回键"
        
        print("  ✓ 搜索页正确使用返回键处理")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    result.print_summary()
    return result


async def test_category_page_handling():
    """测试分类页处理逻辑"""
    result = TestResult("分类页处理")
    
    print("\n" + "="*60)
    print("测试: 分类页处理逻辑")
    print("="*60)
    
    mock_adb = Mock(spec=ADBBridge)
    mock_adb.tap = AsyncMock()
    mock_adb.press_back = AsyncMock()
    
    mock_detector = Mock()
    mock_detector.find_button_yolo = AsyncMock()
    
    guard = PageStateGuard(mock_adb, mock_detector)
    device_id = "test_device"
    
    # 测试1: YOLO成功检测首页按钮
    print("\n[测试1] YOLO成功检测到首页按钮")
    mock_detector.find_button_yolo.return_value = (90, 920)
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.CATEGORY, 
            PageState.HOME, 
            "测试操作"
        )
        
        assert handled == True
        assert mock_detector.find_button_yolo.called
        assert mock_adb.tap.called
        
        call_args = mock_detector.find_button_yolo.call_args
        assert call_args[0][1] == '分类页', "应该使用'分类页'模型"
        assert call_args[0][2] == '首页按钮', "应该查找'首页按钮'"
        
        print("  ✓ YOLO检测成功，正确点击首页按钮")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    # 重置mock
    mock_detector.find_button_yolo.reset_mock()
    mock_adb.tap.reset_mock()
    
    # 测试2: YOLO失败，使用默认坐标
    print("\n[测试2] YOLO失败，使用默认坐标")
    mock_detector.find_button_yolo.return_value = None
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.CATEGORY, 
            PageState.HOME, 
            "测试操作"
        )
        
        assert handled == True
        assert mock_detector.find_button_yolo.called
        assert mock_adb.tap.called
        
        tap_call_args = mock_adb.tap.call_args
        assert tap_call_args[0][1] == 90, "应该使用默认x坐标90"
        assert tap_call_args[0][2] == 920, "应该使用默认y坐标920"
        
        print("  ✓ YOLO失败后正确使用默认坐标")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    result.print_summary()
    return result


async def test_settings_page_handling():
    """测试设置页处理逻辑"""
    result = TestResult("设置页处理")
    
    print("\n" + "="*60)
    print("测试: 设置页处理逻辑")
    print("="*60)
    
    mock_adb = Mock(spec=ADBBridge)
    mock_adb.tap = AsyncMock()
    mock_adb.press_back = AsyncMock()
    
    mock_detector = Mock()
    mock_detector.find_button_yolo = AsyncMock()
    
    guard = PageStateGuard(mock_adb, mock_detector)
    device_id = "test_device"
    
    print("\n[测试] 设置页使用返回键处理")
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.SETTINGS, 
            PageState.PROFILE, 
            "测试操作"
        )
        
        assert handled == True, "应该返回True表示处理成功"
        assert mock_adb.press_back.called, "应该使用返回键"
        
        print("  ✓ 设置页正确使用返回键处理")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    result.print_summary()
    return result


async def test_transaction_history_page_handling():
    """测试交易流水页处理逻辑"""
    result = TestResult("交易流水页处理")
    
    print("\n" + "="*60)
    print("测试: 交易流水页处理逻辑")
    print("="*60)
    
    mock_adb = Mock(spec=ADBBridge)
    mock_adb.tap = AsyncMock()
    mock_adb.press_back = AsyncMock()
    
    mock_detector = Mock()
    mock_detector.find_button_yolo = AsyncMock()
    
    guard = PageStateGuard(mock_adb, mock_detector)
    device_id = "test_device"
    
    print("\n[测试] 交易流水页使用返回键处理")
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.TRANSACTION_HISTORY, 
            PageState.PROFILE, 
            "测试操作"
        )
        
        assert handled == True, "应该返回True表示处理成功"
        assert mock_adb.press_back.called, "应该使用返回键"
        
        print("  ✓ 交易流水页正确使用返回键处理")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    result.print_summary()
    return result


async def test_coupon_page_handling():
    """测试优惠券页处理逻辑"""
    result = TestResult("优惠券页处理")
    
    print("\n" + "="*60)
    print("测试: 优惠券页处理逻辑")
    print("="*60)
    
    mock_adb = Mock(spec=ADBBridge)
    mock_adb.tap = AsyncMock()
    mock_adb.press_back = AsyncMock()
    
    mock_detector = Mock()
    mock_detector.find_button_yolo = AsyncMock()
    
    guard = PageStateGuard(mock_adb, mock_detector)
    device_id = "test_device"
    
    print("\n[测试] 优惠券页使用返回键处理")
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.COUPON, 
            PageState.PROFILE, 
            "测试操作"
        )
        
        assert handled == True, "应该返回True表示处理成功"
        assert mock_adb.press_back.called, "应该使用返回键"
        
        print("  ✓ 优惠券页正确使用返回键处理")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    result.print_summary()
    return result


async def test_points_page_handling():
    """测试积分页处理逻辑"""
    result = TestResult("积分页处理")
    
    print("\n" + "="*60)
    print("测试: 积分页处理逻辑")
    print("="*60)
    
    mock_adb = Mock(spec=ADBBridge)
    mock_adb.tap = AsyncMock()
    mock_adb.press_back = AsyncMock()
    
    mock_detector = Mock()
    mock_detector.find_button_yolo = AsyncMock(return_value=(100, 100))
    
    guard = PageStateGuard(mock_adb, mock_detector)
    device_id = "test_device"
    
    print("\n[测试] 积分页返回按钮检测（特殊处理）")
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.POINTS_PAGE, 
            PageState.PROFILE, 
            "测试操作"
        )
        
        assert handled == True
        assert mock_detector.find_button_yolo.called
        
        call_args = mock_detector.find_button_yolo.call_args
        # 积分页使用特殊的模型名称
        assert call_args[0][1] == '积分页', "应该使用'积分页'模型"
        
        print("  ✓ 积分页处理逻辑正确")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    result.print_summary()
    return result


async def test_popup_handling():
    """测试弹窗处理逻辑"""
    result = TestResult("弹窗处理")
    
    print("\n" + "="*60)
    print("测试: 弹窗处理逻辑")
    print("="*60)
    
    mock_adb = Mock(spec=ADBBridge)
    mock_adb.tap = AsyncMock()
    
    mock_detector = Mock()
    mock_detector.close_popup = AsyncMock(return_value=True)
    
    guard = PageStateGuard(mock_adb, mock_detector)
    device_id = "test_device"
    
    print("\n[测试] 弹窗关闭逻辑")
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.POPUP, 
            PageState.HOME, 
            "测试操作"
        )
        
        assert handled == True, "应该返回True表示处理成功"
        assert mock_detector.close_popup.called, "应该调用close_popup"
        
        print("  ✓ 弹窗处理逻辑正确")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    result.print_summary()
    return result


async def test_unknown_page_handling():
    """测试未知页面处理逻辑"""
    result = TestResult("未知页面处理")
    
    print("\n" + "="*60)
    print("测试: 未知页面处理逻辑")
    print("="*60)
    
    mock_adb = Mock(spec=ADBBridge)
    mock_adb.press_back = AsyncMock()
    
    mock_detector = Mock()
    mock_detector.close_popup = AsyncMock(return_value=False)  # 添加close_popup mock
    
    # 创建mock的页面检测结果
    mock_page_result = Mock()
    mock_page_result.details = "异常页面: 商品列表"
    mock_detector.detect_page = AsyncMock(return_value=mock_page_result)
    
    guard = PageStateGuard(mock_adb, mock_detector)
    device_id = "test_device"
    
    # 测试1: 识别为异常页面（弹窗关闭失败后使用返回键）
    print("\n[测试1] UNKNOWN页面被当作POPUP处理，关闭失败后使用返回键")
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.UNKNOWN, 
            PageState.HOME, 
            "测试操作"
        )
        
        assert handled == True, "应该返回True表示处理成功"
        assert mock_detector.close_popup.called, "应该先尝试关闭弹窗"
        assert mock_adb.press_back.called, "弹窗关闭失败后应该按返回键"
        
        print("  ✓ UNKNOWN页面正确处理（弹窗关闭失败→返回键）")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    # 测试2: 未知页面类型（同样的处理方式）
    print("\n[测试2] UNKNOWN页面处理（降级策略）")
    mock_page_result.details = "未知页面"
    mock_detector.close_popup.reset_mock()
    mock_adb.press_back.reset_mock()
    
    try:
        handled = await guard._handle_unexpected_page(
            device_id, 
            PageState.UNKNOWN, 
            PageState.HOME, 
            "测试操作"
        )
        
        assert handled == True, "应该返回True表示处理成功"
        assert mock_adb.press_back.called, "应该使用返回键"
        
        print("  ✓ UNKNOWN页面正确使用降级策略")
        result.add_pass()
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        result.add_fail(str(e))
    
    result.print_summary()
    return result


async def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始测试: 主流程中非预期页面的处理逻辑")
    print("="*60)
    
    all_results = []
    
    # 运行所有测试
    all_results.append(await test_article_page_handling())
    all_results.append(await test_search_page_handling())
    all_results.append(await test_category_page_handling())
    all_results.append(await test_settings_page_handling())
    all_results.append(await test_transaction_history_page_handling())
    all_results.append(await test_coupon_page_handling())
    all_results.append(await test_points_page_handling())
    all_results.append(await test_popup_handling())
    all_results.append(await test_unknown_page_handling())
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed
    
    print(f"\n总测试数: {total_tests}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")
    print(f"成功率: {total_passed/total_tests*100:.1f}%")
    
    print("\n各测试详情:")
    for r in all_results:
        status = "✓" if r.failed == 0 else "✗"
        print(f"  {status} {r.name}: {r.passed}通过 / {r.failed}失败")
    
    if total_failed > 0:
        print("\n⚠️ 有测试失败，请检查上面的详细信息")
    else:
        print("\n✓ 所有测试通过！")
    
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
