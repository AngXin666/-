"""
运行所有测试套件
包括单元测试和属性测试
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """运行命令并返回结果"""
    print("\n" + "=" * 80)
    print(f"运行: {description}")
    print("=" * 80)
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("错误输出:", result.stderr)
    
    return result.returncode == 0

def main():
    """主测试流程"""
    print("=" * 80)
    print("模型单例优化 - 完整测试套件")
    print("=" * 80)
    
    # 测试结果统计
    results = {}
    
    # 1. 单元测试列表
    unit_tests = [
        ("test_model_manager_basic.py", "基础单例模式测试"),
        ("test_task2_implementation.py", "任务2: 模型加载逻辑测试"),
        ("test_task3_1_model_access.py", "任务3.1: 模型访问接口测试"),
        ("test_task3_4_status_query.py", "任务3.4: 状态查询测试"),
        ("test_task3_comprehensive.py", "任务3: 综合测试"),
        ("test_task4_config_and_dataclasses.py", "任务4: 配置和数据类测试"),
        ("test_task6_component_integration.py", "任务6: 组件集成测试"),
        ("test_task7_orchestrator.py", "任务7: Orchestrator测试"),
        ("test_task8_cleanup.py", "任务8: 资源清理测试"),
        ("test_task8_2_exit_integration.py", "任务8.2: 退出集成测试"),
        ("test_task9_performance_monitoring.py", "任务9: 性能监控测试"),
    ]
    
    # 2. 属性测试列表
    property_tests = [
        ("tests/property/test_property_config_driven_loading.py", "属性测试: 配置驱动加载"),
        ("tests/property/test_property_initialization_order.py", "属性测试: 初始化顺序"),
    ]
    
    # 3. 集成测试
    integration_tests = [
        ("tests/integration/test_e2e_model_manager.py", "端到端集成测试"),
    ]
    
    print("\n[INFO] 测试计划:")
    print(f"  - 单元测试: {len(unit_tests)} 个")
    print(f"  - 属性测试: {len(property_tests)} 个")
    print(f"  - 集成测试: {len(integration_tests)} 个")
    print(f"  - 总计: {len(unit_tests) + len(property_tests) + len(integration_tests)} 个测试套件")
    
    # 运行单元测试
    print("\n" + "=" * 80)
    print("第一阶段: 单元测试")
    print("=" * 80)
    
    for test_file, description in unit_tests:
        if os.path.exists(test_file):
            success = run_command(f"python {test_file}", description)
            results[description] = "[OK] 通过" if success else "[ERROR] 失败"
        else:
            print(f"[WARNING] 跳过 {test_file} (文件不存在)")
            results[description] = "[SKIPPED] 跳过"
    
    # 运行属性测试
    print("\n" + "=" * 80)
    print("第二阶段: 属性测试")
    print("=" * 80)
    
    for test_file, description in property_tests:
        if os.path.exists(test_file):
            success = run_command(f"python {test_file}", description)
            results[description] = "[OK] 通过" if success else "[ERROR] 失败"
        else:
            print(f"[WARNING] 跳过 {test_file} (文件不存在)")
            results[description] = "[SKIPPED] 跳过"
    
    # 运行集成测试
    print("\n" + "=" * 80)
    print("第三阶段: 集成测试")
    print("=" * 80)
    
    for test_file, description in integration_tests:
        if os.path.exists(test_file):
            success = run_command(f"python {test_file}", description)
            results[description] = "[OK] 通过" if success else "[ERROR] 失败"
        else:
            print(f"[WARNING] 跳过 {test_file} (文件不存在)")
            results[description] = "[SKIPPED] 跳过"
    
    # 生成测试报告
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if "通过" in v)
    failed = sum(1 for v in results.values() if "失败" in v)
    skipped = sum(1 for v in results.values() if "跳过" in v)
    total = len(results)
    
    print(f"\n总计: {total} 个测试套件")
    print(f"  [OK] 通过: {passed}")
    print(f"  [ERROR] 失败: {failed}")
    print(f"  [SKIPPED] 跳过: {skipped}")
    
    print("\n详细结果:")
    for name, status in results.items():
        print(f"  {status} - {name}")
    
    # 返回状态
    if failed > 0:
        print("\n[FAILED] 测试失败！请检查失败的测试。")
        return 1
    elif passed == 0:
        print("\n[WARNING] 没有测试通过！")
        return 1
    else:
        print("\n[PASSED] 所有测试通过！")
        return 0

if __name__ == "__main__":
    sys.exit(main())
