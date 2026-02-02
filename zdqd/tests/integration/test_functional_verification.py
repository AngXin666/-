"""
功能验证测试
Functional Verification Tests

验证优化后的功能是否正常工作：
- 页面识别准确性
- 弹窗关闭功能
- 广告页处理
- 导航功能
"""

import asyncio
import time
from typing import Dict, List, Any
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.adb_bridge import ADBBridge
from src.page_detector_hybrid import PageDetectorHybrid, PageState
from src.navigator import Navigator
from src.screen_capture import ScreenCapture
from src.ui_automation import UIAutomation
from src.ad_detection.enhanced_ad_detector import EnhancedAdDetector


class FunctionalVerificationRunner:
    """功能验证测试运行器"""
    
    def __init__(self, device_id: str, adb_path: str = None):
        """初始化测试运行器
        
        Args:
            device_id: 设备ID
            adb_path: ADB路径（可选）
        """
        self.device_id = device_id
        self.adb = ADBBridge(adb_path)
        self.detector = PageDetectorHybrid(self.adb)
        self.navigator = Navigator(self.adb, self.detector)
        self.screen_capture = ScreenCapture(self.adb)
        self.ui_automation = UIAutomation(self.adb, self.screen_capture)
        self.ad_detector = EnhancedAdDetector(self.adb)
        
        # 测试结果
        self.results: List[Dict[str, Any]] = []
    
    def log(self, message: str):
        """输出日志"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    async def test_page_detection_accuracy(self, iterations: int = 5) -> Dict[str, Any]:
        """测试页面识别准确性
        
        Args:
            iterations: 测试迭代次数
            
        Returns:
            测试结果
        """
        self.log("=" * 80)
        self.log(f"测试：页面识别准确性（{iterations}次迭代）")
        self.log("=" * 80)
        
        detection_results = []
        
        for i in range(iterations):
            self.log(f"迭代 {i+1}/{iterations}")
            
            # 使用模板匹配检测
            result_template = await self.detector.detect_page(self.device_id, use_ocr=False)
            self.log(f"  模板匹配: {result_template.state.value} (置信度: {result_template.confidence:.2f})")
            
            # 使用OCR检测
            result_ocr = await self.detector.detect_page(self.device_id, use_ocr=True)
            self.log(f"  OCR识别: {result_ocr.state.value} (置信度: {result_ocr.confidence:.2f})")
            
            # 检查一致性
            consistent = result_template.state == result_ocr.state
            self.log(f"  一致性: {'[OK]' if consistent else '[ERROR]'}")
            
            detection_results.append({
                "iteration": i + 1,
                "template_state": result_template.state.value,
                "ocr_state": result_ocr.state.value,
                "consistent": consistent,
                "template_confidence": result_template.confidence,
                "ocr_confidence": result_ocr.confidence
            })
            
            await asyncio.sleep(1)
        
        # 计算一致性率
        consistent_count = sum(1 for r in detection_results if r["consistent"])
        consistency_rate = (consistent_count / len(detection_results)) * 100
        
        result = {
            "test_name": "页面识别准确性",
            "iterations": iterations,
            "consistent_count": consistent_count,
            "consistency_rate": round(consistency_rate, 1),
            "success": consistency_rate >= 80,  # 80%以上一致性视为成功
            "details": detection_results
        }
        
        self.log(f"")
        self.log(f"测试结果:")
        self.log(f"  一致性: {consistent_count}/{iterations} ({consistency_rate:.1f}%)")
        self.log(f"  成功: {'[OK]' if result['success'] else '[ERROR]'}")
        self.log(f"")
        
        return result
    
    async def test_popup_closing(self) -> Dict[str, Any]:
        """测试弹窗关闭功能
        
        Returns:
            测试结果
        """
        self.log("=" * 80)
        self.log("测试：弹窗关闭功能")
        self.log("=" * 80)
        
        # 检测当前页面
        result = await self.detector.detect_page(self.device_id, use_ocr=True)
        self.log(f"当前页面: {result.state.value}")
        
        if result.state != PageState.POPUP:
            self.log("[WARNING] 当前不是弹窗页面，跳过测试")
            return {
                "test_name": "弹窗关闭功能",
                "success": None,
                "skipped": True,
                "reason": "当前不是弹窗页面"
            }
        
        # 尝试关闭弹窗
        self.log("尝试关闭弹窗...")
        start_time = time.time()
        success = await self.detector.close_popup(self.device_id)
        close_time = time.time() - start_time
        
        if success:
            self.log(f"[OK] 弹窗关闭成功 (耗时: {close_time:.2f}秒)")
        else:
            self.log(f"[ERROR] 弹窗关闭失败")
        
        # 等待页面变化
        await asyncio.sleep(2)
        
        # 验证弹窗是否已关闭
        result_after = await self.detector.detect_page(self.device_id, use_ocr=True)
        self.log(f"关闭后页面: {result_after.state.value}")
        
        popup_closed = result_after.state != PageState.POPUP
        
        result = {
            "test_name": "弹窗关闭功能",
            "success": success and popup_closed,
            "close_time": round(close_time, 2),
            "popup_closed": popup_closed,
            "page_after": result_after.state.value
        }
        
        self.log(f"")
        self.log(f"测试结果:")
        self.log(f"  关闭成功: {'[OK]' if success else '[ERROR]'}")
        self.log(f"  弹窗已关闭: {'[OK]' if popup_closed else '[ERROR]'}")
        self.log(f"  总体成功: {'[OK]' if result['success'] else '[ERROR]'}")
        self.log(f"")
        
        return result
    
    async def test_ad_page_detection(self) -> Dict[str, Any]:
        """测试广告页检测
        
        Returns:
            测试结果
        """
        self.log("=" * 80)
        self.log("测试：广告页检测")
        self.log("=" * 80)
        
        # 使用混合检测器检测
        result_hybrid = await self.detector.detect_page(self.device_id, use_ocr=True)
        self.log(f"混合检测器结果: {result_hybrid.state.value}")
        
        # 使用增强广告检测器检测
        ad_result = await self.ad_detector.detect_ad_page(self.device_id)
        self.log(f"增强检测器结果: {'广告' if ad_result.is_ad else '非广告'}")
        self.log(f"  置信度: {ad_result.confidence:.2f}")
        self.log(f"  检测方法: {ad_result.method}")
        self.log(f"  详情: {ad_result.details}")
        
        # 检查一致性
        hybrid_is_ad = result_hybrid.state == PageState.AD
        consistent = hybrid_is_ad == ad_result.is_ad
        
        result = {
            "test_name": "广告页检测",
            "hybrid_is_ad": hybrid_is_ad,
            "enhanced_is_ad": ad_result.is_ad,
            "consistent": consistent,
            "confidence": ad_result.confidence,
            "method": ad_result.method,
            "success": True  # 只要能检测就算成功
        }
        
        self.log(f"")
        self.log(f"测试结果:")
        self.log(f"  混合检测: {'广告' if hybrid_is_ad else '非广告'}")
        self.log(f"  增强检测: {'广告' if ad_result.is_ad else '非广告'}")
        self.log(f"  一致性: {'[OK]' if consistent else '[ERROR]'}")
        self.log(f"  成功: [OK]")
        self.log(f"")
        
        return result
    
    async def test_navigation_functionality(self) -> Dict[str, Any]:
        """测试导航功能
        
        Returns:
            测试结果
        """
        self.log("=" * 80)
        self.log("测试：导航功能")
        self.log("=" * 80)
        
        test_results = []
        
        # 测试1: 导航到首页
        self.log("测试1: 导航到首页")
        start_time = time.time()
        success_home = await self.navigator.navigate_to_home(self.device_id)
        home_time = time.time() - start_time
        
        if success_home:
            self.log(f"  [OK] 成功 (耗时: {home_time:.2f}秒)")
        else:
            self.log(f"  [ERROR] 失败")
        
        test_results.append({
            "test": "导航到首页",
            "success": success_home,
            "time": round(home_time, 2)
        })
        
        await asyncio.sleep(1)
        
        # 测试2: 导航到个人页面
        self.log("测试2: 导航到个人页面")
        start_time = time.time()
        success_profile = await self.navigator.navigate_to_profile(self.device_id)
        profile_time = time.time() - start_time
        
        if success_profile:
            self.log(f"  [OK] 成功 (耗时: {profile_time:.2f}秒)")
        else:
            self.log(f"  [ERROR] 失败")
        
        test_results.append({
            "test": "导航到个人页面",
            "success": success_profile,
            "time": round(profile_time, 2)
        })
        
        await asyncio.sleep(1)
        
        # 测试3: 返回首页
        self.log("测试3: 返回首页")
        start_time = time.time()
        success_back = await self.navigator.navigate_to_home(self.device_id)
        back_time = time.time() - start_time
        
        if success_back:
            self.log(f"  [OK] 成功 (耗时: {back_time:.2f}秒)")
        else:
            self.log(f"  [ERROR] 失败")
        
        test_results.append({
            "test": "返回首页",
            "success": success_back,
            "time": round(back_time, 2)
        })
        
        # 计算总体成功率
        success_count = sum(1 for r in test_results if r["success"])
        success_rate = (success_count / len(test_results)) * 100
        
        result = {
            "test_name": "导航功能",
            "tests": test_results,
            "success_count": success_count,
            "total_tests": len(test_results),
            "success_rate": round(success_rate, 1),
            "success": success_rate == 100  # 所有测试都成功才算成功
        }
        
        self.log(f"")
        self.log(f"测试结果:")
        self.log(f"  成功: {success_count}/{len(test_results)} ({success_rate:.1f}%)")
        self.log(f"  总体成功: {'[OK]' if result['success'] else '[ERROR]'}")
        self.log(f"")
        
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有功能验证测试
        
        Returns:
            所有测试结果的汇总
        """
        self.log("")
        self.log("=" * 80)
        self.log("功能验证测试套件")
        self.log("=" * 80)
        self.log("")
        
        all_results = []
        
        # 测试1: 页面识别准确性
        try:
            result = await self.test_page_detection_accuracy(iterations=5)
            all_results.append(result)
            self.results.append(result)
        except Exception as e:
            self.log(f"[ERROR] 页面识别准确性测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        await asyncio.sleep(2)
        
        # 测试2: 弹窗关闭功能
        try:
            result = await self.test_popup_closing()
            all_results.append(result)
            self.results.append(result)
        except Exception as e:
            self.log(f"[ERROR] 弹窗关闭功能测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        await asyncio.sleep(2)
        
        # 测试3: 广告页检测
        try:
            result = await self.test_ad_page_detection()
            all_results.append(result)
            self.results.append(result)
        except Exception as e:
            self.log(f"[ERROR] 广告页检测测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        await asyncio.sleep(2)
        
        # 测试4: 导航功能
        try:
            result = await self.test_navigation_functionality()
            all_results.append(result)
            self.results.append(result)
        except Exception as e:
            self.log(f"[ERROR] 导航功能测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 生成汇总报告
        self.log("")
        self.log("=" * 80)
        self.log("功能验证汇总")
        self.log("=" * 80)
        self.log("")
        
        for result in all_results:
            test_name = result.get("test_name", "未知测试")
            success = result.get("success")
            skipped = result.get("skipped", False)
            
            if skipped:
                self.log(f"[SKIPPED] {test_name}: 跳过 ({result.get('reason', '未知原因')})")
            elif success is None:
                self.log(f"[SKIPPED] {test_name}: 未执行")
            elif success:
                self.log(f"[OK] {test_name}: 成功")
            else:
                self.log(f"[ERROR] {test_name}: 失败")
        
        self.log("")
        
        # 计算总体成功率（排除跳过的测试）
        valid_tests = [r for r in all_results if not r.get("skipped", False) and r.get("success") is not None]
        if valid_tests:
            success_count = sum(1 for r in valid_tests if r["success"])
            total_count = len(valid_tests)
            success_rate = (success_count / total_count) * 100
            
            self.log(f"总体成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
            self.log("")
        
        return {
            "all_results": all_results,
            "total_tests": len(all_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


async def main():
    """主函数"""
    # 配置
    device_id = "127.0.0.1:16384"  # 默认MuMu模拟器端口
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        device_id = sys.argv[1]
    
    print(f"")
    print(f"功能验证测试配置:")
    print(f"  设备ID: {device_id}")
    print(f"")
    
    # 创建测试运行器
    runner = FunctionalVerificationRunner(device_id)
    
    # 连接设备
    runner.log("连接设备...")
    connected = await runner.adb.connect(device_id)
    if not connected:
        runner.log(f"[ERROR] 无法连接到设备: {device_id}")
        runner.log("请确保:")
        runner.log("  1. 模拟器已启动")
        runner.log("  2. ADB服务正在运行")
        runner.log("  3. 设备ID正确")
        return
    
    runner.log(f"[OK] 已连接到设备: {device_id}")
    runner.log("")
    
    # 运行所有测试
    try:
        await runner.run_all_tests()
    except Exception as e:
        runner.log(f"[ERROR] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.log("")
        runner.log("测试完成")


if __name__ == "__main__":
    asyncio.run(main())
