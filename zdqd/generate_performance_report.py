"""
生成性能测试报告

读取基准测试结果并生成详细的性能报告
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def format_memory(bytes_value: int) -> str:
    """格式化内存显示"""
    mb = bytes_value / 1024 / 1024
    return f"{mb:.1f}MB"


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}秒"
    else:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"


def load_latest_result(mode: str) -> Dict[str, Any]:
    """加载最新的测试结果"""
    results_dir = Path("benchmark_results")
    if not results_dir.exists():
        return None
    
    files = sorted(results_dir.glob(f"benchmark_{mode}_*.json"), reverse=True)
    if not files:
        return None
    
    with open(files[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_report():
    """生成性能报告"""
    print("=" * 80)
    print("ModelManager 性能优化报告")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载测试结果
    with_manager = load_latest_result("with_manager")
    without_manager = load_latest_result("without_manager")
    
    if not with_manager:
        print("错误: 未找到使用ModelManager的测试结果")
        print("请先运行: python benchmark_model_manager.py --mode with_manager")
        return
    
    # 1. 模型加载性能
    print("1. 模型加载性能")
    print("-" * 80)
    
    if 'model_loading' in with_manager:
        ml = with_manager['model_loading']
        print(f"总加载时间: {format_time(ml['total_time'])}")
        print(f"内存占用: {format_memory(ml['memory_delta'])}")
        print(f"已加载模型: {len(ml['models_loaded'])} 个")
        print()
        
        if ml.get('model_times'):
            print("各模型加载时间:")
            for model_name, model_time in ml['model_times'].items():
                print(f"  - {model_name}: {format_time(model_time)}")
        print()
    
    # 2. 单账号处理性能
    print("2. 单账号处理性能")
    print("-" * 80)
    
    if 'single_account' in with_manager:
        sa = with_manager['single_account']
        print(f"总处理时间: {format_time(sa['total_time'])}")
        print(f"  - 模型加载时间: {format_time(sa['model_loading_time'])}")
        print(f"  - 模型访问时间: {format_time(sa['model_access_time'])}")
        print(f"内存增量: {format_memory(sa['memory_delta'])}")
        print()
        
        if without_manager and 'single_account' in without_manager:
            wo_sa = without_manager['single_account']
            time_saved = wo_sa['total_time'] - sa['total_time']
            memory_saved = wo_sa['memory_delta'] - sa['memory_delta']
            
            print("对比优化前:")
            print(f"  - 时间节省: {format_time(time_saved)}")
            print(f"  - 内存节省: {format_memory(memory_saved)}")
            print()
    
    # 3. 多账号处理性能
    print("3. 多账号处理性能")
    print("-" * 80)
    
    if 'multiple_accounts' in with_manager:
        ma = with_manager['multiple_accounts']
        num_accounts = ma['num_accounts']
        print(f"{num_accounts}账号总时间: {format_time(ma['total_time'])}")
        print(f"平均每账号: {format_time(ma['avg_time_per_account'])}")
        print(f"最快: {format_time(ma['min_time'])}")
        print(f"最慢: {format_time(ma['max_time'])}")
        print(f"内存增量: {format_memory(ma['memory_delta'])}")
        print()
        
        if without_manager and 'multiple_accounts' in without_manager:
            wo_ma = without_manager['multiple_accounts']
            time_saved = wo_ma['total_time'] - ma['total_time']
            memory_saved = wo_ma['memory_delta'] - ma['memory_delta']
            
            print("对比优化前:")
            print(f"  - 时间节省: {format_time(time_saved)}")
            print(f"  - 内存节省: {format_memory(memory_saved)}")
            
            if wo_ma['total_time'] > 0:
                percent = (time_saved / wo_ma['total_time']) * 100
                print(f"  - 性能提升: {percent:.1f}%")
            print()
    
    # 4. 内存使用情况
    print("4. 内存使用情况")
    print("-" * 80)
    
    if 'memory' in with_manager:
        mem = with_manager['memory']
        print(f"初始内存: {format_memory(mem['memory_initial'])}")
        print(f"加载模型后: {format_memory(mem['memory_after_loading'])} (+{format_memory(mem['loading_delta'])})")
        print(f"处理账号后: {format_memory(mem['memory_after_accounts'])} (+{format_memory(mem['accounts_delta'])})")
        print(f"总增量: {format_memory(mem['total_delta'])}")
        print()
        
        if without_manager and 'memory' in without_manager:
            wo_mem = without_manager['memory']
            memory_saved = wo_mem['total_delta'] - mem['total_delta']
            
            print("对比优化前:")
            print(f"  - 内存节省: {format_memory(memory_saved)}")
            
            if wo_mem['total_delta'] > 0:
                percent = (memory_saved / wo_mem['total_delta']) * 100
                print(f"  - 内存优化: {percent:.1f}%")
            print()
    
    # 5. 总结
    print("5. 优化总结")
    print("-" * 80)
    
    if with_manager and without_manager:
        # 计算总体改善
        improvements = []
        
        if 'single_account' in with_manager and 'single_account' in without_manager:
            time_saved = without_manager['single_account']['total_time'] - with_manager['single_account']['total_time']
            improvements.append(f"单账号节省 {format_time(time_saved)}")
        
        if 'multiple_accounts' in with_manager and 'multiple_accounts' in without_manager:
            time_saved = without_manager['multiple_accounts']['total_time'] - with_manager['multiple_accounts']['total_time']
            num_accounts = with_manager['multiple_accounts']['num_accounts']
            improvements.append(f"{num_accounts}账号节省 {format_time(time_saved)}")
        
        if 'memory' in with_manager and 'memory' in without_manager:
            memory_saved = without_manager['memory']['total_delta'] - with_manager['memory']['total_delta']
            improvements.append(f"内存节省 {format_memory(memory_saved)}")
        
        print("✓ 优化效果:")
        for improvement in improvements:
            print(f"  - {improvement}")
        print()
        
        print("✓ 优化优势:")
        print("  - 启动时预加载所有模型，用户看到界面时模型已就绪")
        print("  - 所有账号共享同一个模型实例，消除重复加载")
        print("  - 大幅减少内存占用，支持更多并发账号")
        print("  - 日志更清晰，无重复的模型加载信息")
        print()
    else:
        print("注意: 需要运行两种模式的测试才能对比优化效果")
        print("  python benchmark_model_manager.py --mode with_manager")
        print("  python benchmark_model_manager.py --mode without_manager")
        print()
    
    # 6. 性能指标表
    print("6. 性能指标对比表")
    print("-" * 80)
    
    if with_manager and without_manager:
        print(f"{'指标':<30} {'优化前':<20} {'优化后':<20} {'改善':<20}")
        print("-" * 80)
        
        # 模型加载时间
        if 'model_loading' in with_manager and 'model_loading' in without_manager:
            wo_time = without_manager['model_loading']['total_time']
            w_time = with_manager['model_loading']['total_time']
            diff = wo_time - w_time
            print(f"{'模型加载时间':<30} {format_time(wo_time):<20} {format_time(w_time):<20} {format_time(diff):<20}")
        
        # 单账号处理时间
        if 'single_account' in with_manager and 'single_account' in without_manager:
            wo_time = without_manager['single_account']['total_time']
            w_time = with_manager['single_account']['total_time']
            diff = wo_time - w_time
            print(f"{'单账号处理时间':<30} {format_time(wo_time):<20} {format_time(w_time):<20} {format_time(diff):<20}")
        
        # 30账号总时间
        if 'multiple_accounts' in with_manager and 'multiple_accounts' in without_manager:
            wo_time = without_manager['multiple_accounts']['total_time']
            w_time = with_manager['multiple_accounts']['total_time']
            diff = wo_time - w_time
            num = with_manager['multiple_accounts']['num_accounts']
            print(f"{f'{num}账号总时间':<30} {format_time(wo_time):<20} {format_time(w_time):<20} {format_time(diff):<20}")
        
        # 内存占用
        if 'memory' in with_manager and 'memory' in without_manager:
            wo_mem = without_manager['memory']['total_delta']
            w_mem = with_manager['memory']['total_delta']
            diff = wo_mem - w_mem
            print(f"{'内存占用':<30} {format_memory(wo_mem):<20} {format_memory(w_mem):<20} {format_memory(diff):<20}")
        
        print()
    
    print("=" * 80)
    print("报告生成完成")
    print("=" * 80)


if __name__ == '__main__':
    generate_report()
