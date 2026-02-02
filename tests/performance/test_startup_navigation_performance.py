"""
启动和导航性能测试
Performance Test for Startup and Navigation Optimization
"""

import asyncio
import time
from typing import Dict, List, Any
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.adb_bridge import ADBBridge
from src.page_detector_hybrid import PageDetectorHybrid
from src.ximeng_automation import XimengAutomation
from src.navigator import Navigator
from src.screen_capture import ScreenCapture
from src.ui_automation import UIAutomation
from src.auto_login import AutoLogin
from src.performance.performance_monitor import PerformanceMonitor
from src.performance.detection_cache import DetectionCache
from src.emulator_controller import EmulatorController


class PerformanceTestRunner:
    """性能测试运行器"""
    
    def __init__(self, device_id: str, adb_path: str = None):
        """初始化测试运行器
        
        Args:
            device_id: 设备ID（如 "127.0.0.1:16384"）
            adb_path: ADB路径（可选，默认使用系统ADB）
        """
        self.device_id = device_id
        self.adb = ADBBridge(adb_path)
        self.detector = PageDetectorHybrid(self.adb)
        self.navigator = Navigator(self.adb, self.detector)
        self.screen_capture = ScreenCapture(self.adb)
        self.ui_automation = UIAutomation(self.adb, self.screen_capture)
        self.auto_login = AutoLogin(self.adb, self.detector, self.ui_automation)
        self.ximeng = XimengAutomation(
            self.ui_automation,
            self.screen_capture,
            self.auto_login,
            self.adb
        )
        
        # 测试结果
        self.results: List[Dict[str, Any]] = []
    
    def log(self, message: str):
        """输出日志"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    async def test_startup_flow_optimized(self, package_name: str = "com.ry.xmsc") -> Dict[str, Any]:
        """测试优化后的启动流程
        
        Args:
            package_name: 应用包名
            
        Returns:
            测试结果字典
        """
        self.log("=" * 80)
        self.log("测试：优化后的启动流程")
        self.log("=" * 80)
        
        # 停止应用
        self.log("停止应用...")
        await self.adb.stop_app(self.device_id, package_name)
        await asyncio.sleep(2)
        
        # 启动应用
        self.log("启动应用...")
        start_time = time.time()
        await self.adb.start_app(self.device_id, package_name)
        await asyncio.sleep(2)
        
        # 运行启动流程
        success = await self.ximeng.handle_startup_flow(
            self.device_id,
            log_callback=self.log,
            package_name=package_name,
            max_retries=1,  # 只测试一次，不重试
            stuck_timeout=15,
            max_wait_time=60,
            enable_debug=False  # 关闭调试日志以提高性能
        )
        
        total_time = time.time() - start_time
        
        result = {
            "test_name": "优化后的启动流程",
            "success": success,
            "total_time": round(total_time, 2),
            "target_time": 15.0,
            "meets_target": total_time < 15.0
        }
        
        self.log(f"")
        self.log(f"测试结果:")
        self.log(f"  成功: {'[OK]' if success else '[ERROR]'}")
        self.log(f"  总耗时: {total_time:.2f} 秒")
        self.log(f"  目标耗时: < 15.0 秒")
        self.log(f"  达标: {'[OK]' if result['meets_target'] else '[ERROR]'}")
        self.log(f"")
        
        return result
    
    async def test_navigation_to_profile_optimized(self) -> Dict[str, Any]:
        """测试优化后的导航到个人页面
        
        Returns:
            测试结果字典
        """
        self.log("=" * 80)
        self.log("测试：优化后的导航到个人页面")
        self.log("=" * 80)
        
        # 确保在首页
        self.log("确保在首页...")
        await self.navigator.navigate_to_home(self.device_id)
        await asyncio.sleep(1)
        
        # 创建缓存
        cache = DetectionCache(ttl=0.5)
        
        # 测试导航
        self.log("开始导航到个人页面...")
        start_time = time.time()
        
        success = await self.navigator.navigate_to_profile(
            self.device_id,
            max_attempts=3
        )
        
        total_time = time.time() - start_time
        
        result = {
            "test_name": "优化后的导航到个人页面",
            "success": success,
            "total_time": round(total_time, 2),
            "target_time": 3.0,
            "meets_target": total_time < 3.0
        }
        
        self.log(f"")
        self.log(f"测试结果:")
        self.log(f"  成功: {'[OK]' if success else '[ERROR]'}")
        self.log(f"  总耗时: {total_time:.2f} 秒")
        self.log(f"  目标耗时: < 3.0 秒")
        self.log(f"  达标: {'[OK]' if result['meets_target'] else '[ERROR]'}")
        self.log(f"")
        
        return result
    
    async def test_page_detection_speed(self, iterations: int = 10) -> Dict[str, Any]:
        """测试页面检测速度
        
        Args:
            iterations: 测试迭代次数
            
        Returns:
            测试结果字典
        """
        self.log("=" * 80)
        self.log(f"测试：页面检测速度（{iterations}次迭代）")
        self.log("=" * 80)
        
        template_times = []
        ocr_times = []
        
        for i in range(iterations):
            # 测试模板匹配
            start_time = time.time()
            result = await self.detector.detect_page(self.device_id, use_ocr=False)
            template_time = time.time() - start_time
            template_times.append(template_time)
            
            self.log(f"  迭代 {i+1}/{iterations} - 模板匹配: {template_time:.3f}秒, 结果: {result.state.value}")
            
            await asyncio.sleep(0.1)
            
            # 测试OCR识别
            start_time = time.time()
            result = await self.detector.detect_page(self.device_id, use_ocr=True)
            ocr_time = time.time() - start_time
            ocr_times.append(ocr_time)
            
            self.log(f"  迭代 {i+1}/{iterations} - OCR识别: {ocr_time:.3f}秒, 结果: {result.state.value}")
            
            await asyncio.sleep(0.1)
        
        avg_template_time = sum(template_times) / len(template_times)
        avg_ocr_time = sum(ocr_times) / len(ocr_times)
        speedup = avg_ocr_time / avg_template_time if avg_template_time > 0 else 0
        
        result = {
            "test_name": "页面检测速度",
            "iterations": iterations,
            "avg_template_time": round(avg_template_time, 3),
            "avg_ocr_time": round(avg_ocr_time, 3),
            "speedup": round(speedup, 1),
            "template_target": 0.1,
            "ocr_target": 2.0,
            "template_meets_target": avg_template_time < 0.1,
            "ocr_meets_target": avg_ocr_time < 2.0
        }
        
        self.log(f"")
        self.log(f"测试结果:")
        self.log(f"  平均模板匹配耗时: {avg_template_time:.3f} 秒")
        self.log(f"  平均OCR识别耗时: {avg_ocr_time:.3f} 秒")
        self.log(f"  速度提升: {speedup:.1f}x")
        self.log(f"  模板匹配达标: {'[OK]' if result['template_meets_target'] else '[ERROR]'} (目标 < 0.1秒)")
        self.log(f"  OCR识别达标: {'[OK]' if result['ocr_meets_target'] else '[ERROR]'} (目标 < 2.0秒)")
        self.log(f"")
        
        return result
    
    async def test_cache_effectiveness(self, iterations: int = 5) -> Dict[str, Any]:
        """测试缓存有效性
        
        Args:
            iterations: 测试迭代次数
            
        Returns:
            测试结果字典
        """
        self.log("=" * 80)
        self.log(f"测试：缓存有效性（{iterations}次迭代）")
        self.log("=" * 80)
        
        cache = DetectionCache(ttl=0.5)
        
        cache_hit_times = []
        cache_miss_times = []
        
        for i in range(iterations):
            # 清除缓存，测试缓存未命中
            cache.clear(self.device_id)
            
            start_time = time.time()
            result = await self.detector.detect_page(self.device_id, use_ocr=False)
            miss_time = time.time() - start_time
            cache_miss_times.append(miss_time)
            
            # 设置缓存
            cache.set(self.device_id, result)
            
            self.log(f"  迭代 {i+1}/{iterations} - 缓存未命中: {miss_time:.3f}秒")
            
            # 测试缓存命中
            start_time = time.time()
            cached_result = cache.get(self.device_id)
            hit_time = time.time() - start_time
            cache_hit_times.append(hit_time)
            
            self.log(f"  迭代 {i+1}/{iterations} - 缓存命中: {hit_time:.6f}秒")
            
            await asyncio.sleep(0.1)
        
        avg_miss_time = sum(cache_miss_times) / len(cache_miss_times)
        avg_hit_time = sum(cache_hit_times) / len(cache_hit_times)
        speedup = avg_miss_time / avg_hit_time if avg_hit_time > 0 else 0
        
        result = {
            "test_name": "缓存有效性",
            "iterations": iterations,
            "avg_cache_miss_time": round(avg_miss_time, 3),
            "avg_cache_hit_time": round(avg_hit_time, 6),
            "speedup": round(speedup, 1)
        }
        
        self.log(f"")
        self.log(f"测试结果:")
        self.log(f"  平均缓存未命中耗时: {avg_miss_time:.3f} 秒")
        self.log(f"  平均缓存命中耗时: {avg_hit_time:.6f} 秒")
        self.log(f"  速度提升: {speedup:.1f}x")
        self.log(f"")
        
        return result
    
    async def run_all_tests(self, package_name: str = "com.ry.xmsc") -> Dict[str, Any]:
        """运行所有性能测试
        
        Args:
            package_name: 应用包名
            
        Returns:
            所有测试结果的汇总
        """
        self.log("")
        self.log("=" * 80)
        self.log("启动和导航性能测试套件")
        self.log("=" * 80)
        self.log("")
        
        all_results = []
        
        # 测试1: 页面检测速度
        try:
            result = await self.test_page_detection_speed(iterations=10)
            all_results.append(result)
            self.results.append(result)
        except Exception as e:
            self.log(f"[ERROR] 页面检测速度测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        await asyncio.sleep(2)
        
        # 测试2: 缓存有效性
        try:
            result = await self.test_cache_effectiveness(iterations=5)
            all_results.append(result)
            self.results.append(result)
        except Exception as e:
            self.log(f"[ERROR] 缓存有效性测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        await asyncio.sleep(2)
        
        # 测试3: 导航到个人页面（优化后）
        try:
            result = await self.test_navigation_to_profile_optimized()
            all_results.append(result)
            self.results.append(result)
        except Exception as e:
            self.log(f"[ERROR] 导航测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        await asyncio.sleep(2)
        
        # 测试4: 启动流程（优化后）
        try:
            result = await self.test_startup_flow_optimized(package_name)
            all_results.append(result)
            self.results.append(result)
        except Exception as e:
            self.log(f"[ERROR] 启动流程测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 生成汇总报告
        self.log("")
        self.log("=" * 80)
        self.log("测试汇总")
        self.log("=" * 80)
        self.log("")
        
        for result in all_results:
            test_name = result.get("test_name", "未知测试")
            self.log(f"测试: {test_name}")
            
            if "success" in result:
                self.log(f"  成功: {'[OK]' if result['success'] else '[ERROR]'}")
            
            if "total_time" in result:
                self.log(f"  总耗时: {result['total_time']:.2f} 秒")
            
            if "target_time" in result:
                self.log(f"  目标耗时: < {result['target_time']:.1f} 秒")
            
            if "meets_target" in result:
                self.log(f"  达标: {'[OK]' if result['meets_target'] else '[ERROR]'}")
            
            if "speedup" in result:
                self.log(f"  速度提升: {result['speedup']:.1f}x")
            
            self.log("")
        
        # 计算总体达标率
        tests_with_target = [r for r in all_results if "meets_target" in r]
        if tests_with_target:
            met_targets = sum(1 for r in tests_with_target if r["meets_target"])
            total_targets = len(tests_with_target)
            success_rate = (met_targets / total_targets) * 100
            
            self.log(f"总体达标率: {met_targets}/{total_targets} ({success_rate:.1f}%)")
            self.log("")
        
        return {
            "all_results": all_results,
            "total_tests": len(all_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def print_summary(self):
        """打印测试摘要"""
        if not self.results:
            self.log("没有测试结果")
            return
        
        self.log("")
        self.log("=" * 80)
        self.log("性能测试摘要")
        self.log("=" * 80)
        self.log("")
        
        # 关键指标
        for result in self.results:
            test_name = result.get("test_name", "未知测试")
            self.log(f"【{test_name}】")
            
            if "total_time" in result and "target_time" in result:
                total_time = result["total_time"]
                target_time = result["target_time"]
                improvement = ((target_time - total_time) / target_time) * 100
                
                self.log(f"  实际耗时: {total_time:.2f}秒")
                self.log(f"  目标耗时: {target_time:.2f}秒")
                self.log(f"  性能提升: {improvement:.1f}%")
                self.log(f"  达标: {'[OK]' if result.get('meets_target') else '[ERROR]'}")
            
            if "speedup" in result:
                self.log(f"  速度提升: {result['speedup']:.1f}x")
            
            self.log("")
        
        # 生成性能对比报告
        try:
            from tests.performance.performance_comparison import PerformanceComparison
            
            analyzer = PerformanceComparison()
            report = analyzer.generate_comparison_report(self.results)
            print(report)
            
            # 保存报告
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_comparison_report_{timestamp}.txt"
            analyzer.save_report(report, filename)
            
        except Exception as e:
            self.log(f"[WARNING] 生成对比报告失败: {e}")


async def main():
    """主函数"""
    print("")
    print("=" * 80)
    print("启动和导航性能测试")
    print("=" * 80)
    print("")
    
    # 1. 自动检测模拟器
    print("🔍 正在检测模拟器...")
    emulator = EmulatorController()
    
    if not emulator.is_available():
        print("[FAILED] 未检测到模拟器！")
        print("")
        print("请确保已安装MuMu模拟器")
        return
    
    print(f"[PASSED] {emulator.get_emulator_info()}")
    print("")
    
    # 2. 启动模拟器（如果未运行）
    print("🚀 检查模拟器运行状态...")
    instance_index = 0
    
    if not await emulator._is_running(instance_index):
        print("模拟器未运行，正在启动...")
        if await emulator.launch_instance(instance_index):
            print("[PASSED] 模拟器启动成功")
        else:
            print("[FAILED] 模拟器启动失败")
            return
    else:
        print("[PASSED] 模拟器已在运行")
    
    print("")
    
    # 3. 获取设备ID和ADB路径
    adb_port = await emulator.get_adb_port(instance_index)
    device_id = f"127.0.0.1:{adb_port}"
    adb_path = emulator.get_adb_path()
    
    print(f"📱 设备信息:")
    print(f"  设备ID: {device_id}")
    print(f"  ADB路径: {adb_path}")
    print("")
    
    # 4. 检查应用包名
    package_name = "com.ry.xmsc"
    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    
    print(f"📦 应用包名: {package_name}")
    print("")
    
    # 5. 创建测试运行器
    runner = PerformanceTestRunner(device_id, adb_path)
    
    # 6. 连接设备
    print("🔗 连接设备...")
    connected = await runner.adb.connect(device_id)
    if not connected:
        print(f"[FAILED] 无法连接到设备: {device_id}")
        print("")
        print("尝试重启ADB服务...")
        try:
            runner.adb._run_adb("kill-server")
            await asyncio.sleep(2)
            runner.adb._run_adb("start-server")
            await asyncio.sleep(2)
            connected = await runner.adb.connect(device_id)
            if not connected:
                print("[FAILED] 重启ADB后仍无法连接")
                return
            print("[PASSED] 重启ADB后连接成功")
        except Exception as e:
            print(f"[FAILED] ADB重启失败: {e}")
            return
    else:
        print(f"[PASSED] 已连接到设备: {device_id}")
    
    print("")
    
    # 7. 检查应用是否已安装
    print(f"🔍 检查应用是否已安装...")
    is_installed = await runner.adb.is_app_installed(device_id, package_name)
    if not is_installed:
        print(f"[FAILED] 应用未安装: {package_name}")
        print(f"请先安装应用后再运行测试")
        return
    
    print(f"[PASSED] 应用已安装: {package_name}")
    print("")
    
    # 8. 启动应用到首页
    print("🚀 启动应用到首页...")
    try:
        # 先停止应用
        await runner.adb.stop_app(device_id, package_name)
        await asyncio.sleep(2)
        
        # 启动应用
        await runner.adb.start_app(device_id, package_name)
        await asyncio.sleep(3)
        
        # 运行启动流程（处理广告、弹窗等）
        success = await runner.ximeng.handle_startup_flow(
            device_id,
            log_callback=lambda msg: print(f"  {msg}"),
            package_name=package_name,
            max_retries=2,
            stuck_timeout=15,
            max_wait_time=60,
            enable_debug=False
        )
        
        if success:
            print("[PASSED] 应用已启动到首页")
        else:
            print("[WARNING]️ 应用启动可能未完全到达首页，但将继续测试")
        
        print("")
    except Exception as e:
        print(f"[WARNING]️ 启动应用时出错: {e}")
        print("将继续运行测试...")
        print("")
    
    # 9. 运行所有测试
    try:
        await runner.run_all_tests(package_name)
        runner.print_summary()
    except Exception as e:
        print(f"[FAILED] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("")
        print("=" * 80)
        print("测试完成")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
