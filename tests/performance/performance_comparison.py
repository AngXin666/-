"""
性能对比分析工具
Performance Comparison Analysis Tool

用于对比优化前后的性能数据，生成对比报告
"""

import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class PerformanceBaseline:
    """性能基准数据"""
    startup_time: float  # 启动流程耗时（秒）
    navigation_time: float  # 导航耗时（秒）
    page_detection_time: float  # 页面检测耗时（秒）
    total_wait_time: float  # 总等待时间（秒）
    ocr_count: int  # OCR调用次数
    template_count: int  # 模板匹配次数


class PerformanceComparison:
    """性能对比分析器"""
    
    # 优化前的基准数据（根据需求文档）
    BASELINE_BEFORE = PerformanceBaseline(
        startup_time=45.0,  # 30-60秒，取中间值
        navigation_time=12.5,  # 10-15秒，取中间值
        page_detection_time=2.0,  # OCR识别 1-3秒，取中间值
        total_wait_time=25.0,  # 估算：广告10秒 + 弹窗3秒 + 其他等待
        ocr_count=15,  # 估算：启动流程中多次使用OCR
        template_count=0  # 优化前主要使用OCR
    )
    
    # 性能目标（根据需求文档）
    PERFORMANCE_TARGETS = {
        "startup_time": 15.0,  # < 15秒
        "navigation_time": 3.0,  # < 3秒
        "template_detection_time": 0.1,  # < 0.1秒
        "ocr_detection_time": 2.0,  # < 2秒
    }
    
    def __init__(self):
        """初始化对比分析器"""
        self.results: List[Dict[str, Any]] = []
    
    def add_test_result(self, result: Dict[str, Any]):
        """添加测试结果
        
        Args:
            result: 测试结果字典
        """
        self.results.append(result)
    
    def calculate_improvement(self, before: float, after: float) -> float:
        """计算性能提升百分比
        
        Args:
            before: 优化前的值
            after: 优化后的值
            
        Returns:
            提升百分比（正数表示提升，负数表示下降）
        """
        if before == 0:
            return 0.0
        return ((before - after) / before) * 100
    
    def analyze_startup_performance(self, actual_time: float) -> Dict[str, Any]:
        """分析启动流程性能
        
        Args:
            actual_time: 实际耗时（秒）
            
        Returns:
            分析结果
        """
        baseline = self.BASELINE_BEFORE.startup_time
        target = self.PERFORMANCE_TARGETS["startup_time"]
        
        improvement = self.calculate_improvement(baseline, actual_time)
        meets_target = actual_time < target
        
        return {
            "metric": "启动流程耗时",
            "baseline": baseline,
            "actual": actual_time,
            "target": target,
            "improvement": round(improvement, 1),
            "meets_target": meets_target,
            "time_saved": round(baseline - actual_time, 2)
        }
    
    def analyze_navigation_performance(self, actual_time: float) -> Dict[str, Any]:
        """分析导航性能
        
        Args:
            actual_time: 实际耗时（秒）
            
        Returns:
            分析结果
        """
        baseline = self.BASELINE_BEFORE.navigation_time
        target = self.PERFORMANCE_TARGETS["navigation_time"]
        
        improvement = self.calculate_improvement(baseline, actual_time)
        meets_target = actual_time < target
        
        return {
            "metric": "导航到个人页耗时",
            "baseline": baseline,
            "actual": actual_time,
            "target": target,
            "improvement": round(improvement, 1),
            "meets_target": meets_target,
            "time_saved": round(baseline - actual_time, 2)
        }
    
    def analyze_detection_performance(
        self, 
        template_time: float, 
        ocr_time: float
    ) -> Dict[str, Any]:
        """分析页面检测性能
        
        Args:
            template_time: 模板匹配平均耗时（秒）
            ocr_time: OCR识别平均耗时（秒）
            
        Returns:
            分析结果
        """
        baseline = self.BASELINE_BEFORE.page_detection_time
        template_target = self.PERFORMANCE_TARGETS["template_detection_time"]
        ocr_target = self.PERFORMANCE_TARGETS["ocr_detection_time"]
        
        template_improvement = self.calculate_improvement(baseline, template_time)
        speedup = baseline / template_time if template_time > 0 else 0
        
        return {
            "metric": "页面检测速度",
            "baseline_ocr": baseline,
            "actual_template": template_time,
            "actual_ocr": ocr_time,
            "template_target": template_target,
            "ocr_target": ocr_target,
            "template_improvement": round(template_improvement, 1),
            "speedup": round(speedup, 1),
            "template_meets_target": template_time < template_target,
            "ocr_meets_target": ocr_time < ocr_target
        }
    
    def generate_comparison_report(self, test_results: List[Dict[str, Any]]) -> str:
        """生成对比报告
        
        Args:
            test_results: 测试结果列表
            
        Returns:
            格式化的对比报告文本
        """
        report_lines = []
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("性能对比分析报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 提取测试结果
        startup_result = None
        navigation_result = None
        detection_result = None
        
        for result in test_results:
            test_name = result.get("test_name", "")
            
            if "启动流程" in test_name:
                startup_result = result
            elif "导航" in test_name:
                navigation_result = result
            elif "检测速度" in test_name:
                detection_result = result
        
        # 1. 启动流程性能对比
        if startup_result and "total_time" in startup_result:
            report_lines.append("【1. 启动流程性能】")
            report_lines.append("")
            
            analysis = self.analyze_startup_performance(startup_result["total_time"])
            
            report_lines.append(f"  优化前基准: {analysis['baseline']:.1f} 秒")
            report_lines.append(f"  优化后实际: {analysis['actual']:.1f} 秒")
            report_lines.append(f"  性能目标:   < {analysis['target']:.1f} 秒")
            report_lines.append(f"  性能提升:   {analysis['improvement']:.1f}%")
            report_lines.append(f"  节省时间:   {analysis['time_saved']:.1f} 秒")
            report_lines.append(f"  达标状态:   {'[OK] 达标' if analysis['meets_target'] else '[ERROR] 未达标'}")
            report_lines.append("")
        
        # 2. 导航性能对比
        if navigation_result and "total_time" in navigation_result:
            report_lines.append("【2. 导航性能】")
            report_lines.append("")
            
            analysis = self.analyze_navigation_performance(navigation_result["total_time"])
            
            report_lines.append(f"  优化前基准: {analysis['baseline']:.1f} 秒")
            report_lines.append(f"  优化后实际: {analysis['actual']:.1f} 秒")
            report_lines.append(f"  性能目标:   < {analysis['target']:.1f} 秒")
            report_lines.append(f"  性能提升:   {analysis['improvement']:.1f}%")
            report_lines.append(f"  节省时间:   {analysis['time_saved']:.1f} 秒")
            report_lines.append(f"  达标状态:   {'[OK] 达标' if analysis['meets_target'] else '[ERROR] 未达标'}")
            report_lines.append("")
        
        # 3. 页面检测性能对比
        if detection_result:
            report_lines.append("【3. 页面检测性能】")
            report_lines.append("")
            
            template_time = detection_result.get("avg_template_time", 0)
            ocr_time = detection_result.get("avg_ocr_time", 0)
            
            analysis = self.analyze_detection_performance(template_time, ocr_time)
            
            report_lines.append(f"  优化前（OCR）:      {analysis['baseline_ocr']:.3f} 秒")
            report_lines.append(f"  优化后（模板匹配）: {analysis['actual_template']:.3f} 秒")
            report_lines.append(f"  优化后（OCR）:      {analysis['actual_ocr']:.3f} 秒")
            report_lines.append(f"  模板匹配目标:       < {analysis['template_target']:.3f} 秒")
            report_lines.append(f"  OCR目标:            < {analysis['ocr_target']:.3f} 秒")
            report_lines.append(f"  速度提升:           {analysis['speedup']:.1f}x")
            report_lines.append(f"  模板匹配达标:       {'[OK] 达标' if analysis['template_meets_target'] else '[ERROR] 未达标'}")
            report_lines.append(f"  OCR达标:            {'[OK] 达标' if analysis['ocr_meets_target'] else '[ERROR] 未达标'}")
            report_lines.append("")
        
        # 4. 总体评估
        report_lines.append("【4. 总体评估】")
        report_lines.append("")
        
        # 计算总体达标率
        all_metrics = []
        
        if startup_result and "total_time" in startup_result:
            analysis = self.analyze_startup_performance(startup_result["total_time"])
            all_metrics.append(("启动流程", analysis["meets_target"], analysis["improvement"]))
        
        if navigation_result and "total_time" in navigation_result:
            analysis = self.analyze_navigation_performance(navigation_result["total_time"])
            all_metrics.append(("导航", analysis["meets_target"], analysis["improvement"]))
        
        if detection_result:
            template_time = detection_result.get("avg_template_time", 0)
            ocr_time = detection_result.get("avg_ocr_time", 0)
            analysis = self.analyze_detection_performance(template_time, ocr_time)
            all_metrics.append(("模板检测", analysis["template_meets_target"], analysis["template_improvement"]))
            all_metrics.append(("OCR检测", analysis["ocr_meets_target"], 0))
        
        if all_metrics:
            met_count = sum(1 for _, meets, _ in all_metrics if meets)
            total_count = len(all_metrics)
            success_rate = (met_count / total_count) * 100
            
            report_lines.append(f"  达标指标: {met_count}/{total_count}")
            report_lines.append(f"  达标率:   {success_rate:.1f}%")
            report_lines.append("")
            
            # 计算平均性能提升
            improvements = [imp for _, _, imp in all_metrics if imp > 0]
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                report_lines.append(f"  平均性能提升: {avg_improvement:.1f}%")
                report_lines.append("")
            
            # 详细指标
            report_lines.append("  各指标达标情况:")
            for metric_name, meets, improvement in all_metrics:
                status = "[OK]" if meets else "[ERROR]"
                imp_str = f"(+{improvement:.1f}%)" if improvement > 0 else ""
                report_lines.append(f"    {status} {metric_name} {imp_str}")
            report_lines.append("")
        
        # 5. 优化点贡献度分析
        report_lines.append("【5. 优化点贡献度分析】")
        report_lines.append("")
        
        # 根据设计文档中的预期贡献度
        contributions = [
            ("模板匹配替代OCR", "40%", "10-20秒"),
            ("轮询检测替代固定等待", "30%", "8-12秒"),
            ("检测缓存", "15%", "3-5秒"),
            ("导航路径优化", "15%", "3-5秒")
        ]
        
        for opt_name, contribution, time_saved in contributions:
            report_lines.append(f"  {opt_name}:")
            report_lines.append(f"    预期贡献度: {contribution}")
            report_lines.append(f"    预期节省时间: {time_saved}")
        
        report_lines.append("")
        
        # 6. 结论
        report_lines.append("【6. 结论】")
        report_lines.append("")
        
        if all_metrics:
            if success_rate >= 80:
                report_lines.append("  [OK] 性能优化目标基本达成")
                report_lines.append("  [OK] 大部分指标达到或超过预期目标")
            elif success_rate >= 50:
                report_lines.append("  [WARNING] 性能优化部分达成")
                report_lines.append("  [WARNING] 部分指标需要进一步优化")
            else:
                report_lines.append("  [ERROR] 性能优化未达预期")
                report_lines.append("  [ERROR] 需要重新评估优化策略")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_report(self, report: str, filename: str = "performance_comparison_report.txt"):
        """保存报告到文件
        
        Args:
            report: 报告文本
            filename: 文件名
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"报告已保存到: {filename}")


def main():
    """主函数 - 示例用法"""
    # 创建对比分析器
    analyzer = PerformanceComparison()
    
    # 示例测试结果
    test_results = [
        {
            "test_name": "优化后的启动流程",
            "total_time": 12.5,
            "target_time": 15.0,
            "meets_target": True
        },
        {
            "test_name": "优化后的导航到个人页面",
            "total_time": 2.3,
            "target_time": 3.0,
            "meets_target": True
        },
        {
            "test_name": "页面检测速度",
            "avg_template_time": 0.08,
            "avg_ocr_time": 1.5,
            "speedup": 25.0,
            "template_meets_target": True,
            "ocr_meets_target": True
        }
    ]
    
    # 生成报告
    report = analyzer.generate_comparison_report(test_results)
    print(report)
    
    # 保存报告
    analyzer.save_report(report)


if __name__ == "__main__":
    main()
