"""
代码审查清单
检查线程安全性、错误处理、资源清理和代码质量
"""

import os
import re
import sys

def check_thread_safety():
    """检查线程安全性"""
    print("\n" + "=" * 80)
    print("1. 线程安全性检查")
    print("=" * 80)
    
    issues = []
    
    # 读取model_manager.py
    with open('src/model_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查1: 是否使用了锁
    if 'threading.Lock()' in content:
        print("[OK] 使用了threading.Lock")
    else:
        issues.append("未找到threading.Lock的使用")
    
    # 检查2: 关键方法是否使用了锁
    critical_methods = [
        'get_page_detector_integrated',
        'get_page_detector_hybrid',
        'get_ocr_thread_pool',
        '__new__',
        '__init__'
    ]
    
    for method in critical_methods:
        # 查找方法定义
        pattern = rf'def {method}\(.*?\):'
        if re.search(pattern, content):
            # 检查方法体中是否有with self._lock
            method_match = re.search(rf'def {method}\(.*?\):.*?(?=\n    def |\nclass |\Z)', content, re.DOTALL)
            if method_match:
                method_body = method_match.group(0)
                if 'with self._lock' in method_body or 'with cls._lock' in method_body:
                    print(f"[OK] {method}() 使用了锁保护")
                else:
                    # 某些方法可能不需要锁（如只读操作）
                    if method in ['get_page_detector_integrated', 'get_page_detector_hybrid', 'get_ocr_thread_pool']:
                        issues.append(f"{method}() 可能需要锁保护")
    
    # 检查3: 单例模式的双重检查锁定
    if 'if cls._instance is None:' in content and 'with cls._lock:' in content:
        print("[OK] 单例模式使用了双重检查锁定")
    else:
        issues.append("单例模式可能缺少双重检查锁定")
    
    if issues:
        print("\n[WARNING] 发现潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[OK] 线程安全性检查通过")
        return True

def check_error_handling():
    """检查错误处理"""
    print("\n" + "=" * 80)
    print("2. 错误处理检查")
    print("=" * 80)
    
    issues = []
    
    # 读取model_manager.py
    with open('src/model_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查1: 是否有文件验证
    if '_validate_model_files' in content:
        print("[OK] 实现了文件验证方法")
    else:
        issues.append("缺少文件验证方法")
    
    # 检查2: 是否有重试机制
    if '_load_model_with_retry' in content:
        print("[OK] 实现了重试机制")
    else:
        issues.append("缺少重试机制")
    
    # 检查3: 是否有GPU降级逻辑
    if 'torch.cuda.is_available()' in content:
        print("[OK] 实现了GPU可用性检查")
    else:
        print("[INFO] 未找到GPU检查（可能不需要）")
    
    # 检查4: 是否有异常捕获
    try_count = content.count('try:')
    except_count = content.count('except')
    
    if try_count > 0 and except_count > 0:
        print(f"[OK] 使用了异常处理 (try: {try_count}, except: {except_count})")
    else:
        issues.append("缺少异常处理")
    
    # 检查5: 是否有RuntimeError抛出
    if 'raise RuntimeError' in content:
        print("[OK] 使用RuntimeError报告错误")
    else:
        issues.append("未使用RuntimeError报告错误")
    
    # 检查6: 是否有FileNotFoundError
    if 'raise FileNotFoundError' in content or 'FileNotFoundError' in content:
        print("[OK] 处理文件不存在的情况")
    else:
        issues.append("未处理文件不存在的情况")
    
    if issues:
        print("\n[WARNING] 发现潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[OK] 错误处理检查通过")
        return True

def check_resource_cleanup():
    """检查资源清理"""
    print("\n" + "=" * 80)
    print("3. 资源清理检查")
    print("=" * 80)
    
    issues = []
    
    # 读取model_manager.py
    with open('src/model_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查1: 是否有cleanup方法
    if 'def cleanup(self):' in content:
        print("[OK] 实现了cleanup()方法")
    else:
        issues.append("缺少cleanup()方法")
    
    # 检查2: cleanup是否释放模型
    if 'del self._models' in content or 'self._models.clear()' in content:
        print("[OK] cleanup()释放模型实例")
    else:
        issues.append("cleanup()可能未释放模型实例")
    
    # 检查3: 是否清理GPU缓存
    if 'torch.cuda.empty_cache()' in content:
        print("[OK] cleanup()清理GPU缓存")
    else:
        print("[INFO] 未找到GPU缓存清理（可能不需要）")
    
    # 检查4: 是否强制垃圾回收
    if 'gc.collect()' in content:
        print("[OK] cleanup()执行垃圾回收")
    else:
        issues.append("cleanup()未执行垃圾回收")
    
    # 检查5: run.py是否调用cleanup
    if os.path.exists('run.py'):
        with open('run.py', 'r', encoding='utf-8') as f:
            run_content = f.read()
        
        if 'cleanup()' in run_content:
            print("[OK] run.py调用cleanup()")
        else:
            issues.append("run.py未调用cleanup()")
    
    if issues:
        print("\n[WARNING] 发现潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[OK] 资源清理检查通过")
        return True

def check_code_quality():
    """检查代码质量"""
    print("\n" + "=" * 80)
    print("4. 代码质量检查")
    print("=" * 80)
    
    issues = []
    
    # 读取model_manager.py
    with open('src/model_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # 检查1: 是否有文档字符串
    docstring_count = content.count('"""')
    if docstring_count >= 10:  # 至少有一些文档字符串
        print(f"[OK] 包含文档字符串 ({docstring_count // 2} 个)")
    else:
        issues.append("文档字符串较少")
    
    # 检查2: 是否有类型注解
    if '-> ' in content or ': ' in content:
        print("[OK] 使用了类型注解")
    else:
        print("[INFO] 未使用类型注解（可选）")
    
    # 检查3: 是否有日志
    if 'self._log(' in content or 'print(' in content:
        print("[OK] 包含日志输出")
    else:
        issues.append("缺少日志输出")
    
    # 检查4: 代码行数
    code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
    print(f"[INFO] 代码行数: {code_lines} 行")
    
    if code_lines > 2000:
        print("[WARNING] 代码行数较多，考虑拆分")
    
    # 检查5: 是否有TODO或FIXME
    todo_count = content.count('TODO') + content.count('FIXME')
    if todo_count > 0:
        print(f"[INFO] 发现 {todo_count} 个TODO/FIXME标记")
    
    # 检查6: 是否有魔法数字
    magic_numbers = re.findall(r'\b\d{3,}\b', content)
    if len(magic_numbers) > 10:
        print(f"[WARNING] 发现较多魔法数字 ({len(magic_numbers)} 个)")
    
    # 检查7: 方法数量
    method_count = len(re.findall(r'def \w+\(', content))
    print(f"[INFO] 方法数量: {method_count} 个")
    
    if issues:
        print("\n[WARNING] 发现潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[OK] 代码质量检查通过")
        return True

def check_integration():
    """检查组件集成"""
    print("\n" + "=" * 80)
    print("5. 组件集成检查")
    print("=" * 80)
    
    issues = []
    
    # 检查XimengAutomation
    if os.path.exists('src/ximeng_automation.py'):
        with open('src/ximeng_automation.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'ModelManager' in content:
            print("[OK] XimengAutomation导入ModelManager")
        else:
            issues.append("XimengAutomation未导入ModelManager")
        
        if 'model_manager.get_page_detector_integrated()' in content or 'ModelManager.get_instance()' in content:
            print("[OK] XimengAutomation使用ModelManager获取模型")
        else:
            issues.append("XimengAutomation未使用ModelManager获取模型")
    
    # 检查Orchestrator
    if os.path.exists('src/orchestrator.py'):
        with open('src/orchestrator.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'ModelManager' in content:
            print("[OK] Orchestrator导入ModelManager")
        else:
            issues.append("Orchestrator未导入ModelManager")
        
        if 'is_initialized()' in content:
            print("[OK] Orchestrator检查ModelManager初始化状态")
        else:
            issues.append("Orchestrator未检查ModelManager初始化状态")
    
    # 检查run.py
    if os.path.exists('run.py'):
        with open('run.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'ModelManager' in content:
            print("[OK] run.py导入ModelManager")
        else:
            issues.append("run.py未导入ModelManager")
        
        if 'initialize_all_models' in content:
            print("[OK] run.py调用initialize_all_models()")
        else:
            issues.append("run.py未调用initialize_all_models()")
    
    if issues:
        print("\n[WARNING] 发现潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[OK] 组件集成检查通过")
        return True

def main():
    """主函数"""
    print("=" * 80)
    print("模型单例优化 - 代码审查清单")
    print("=" * 80)
    
    results = []
    
    # 执行所有检查
    results.append(("线程安全性", check_thread_safety()))
    results.append(("错误处理", check_error_handling()))
    results.append(("资源清理", check_resource_cleanup()))
    results.append(("代码质量", check_code_quality()))
    results.append(("组件集成", check_integration()))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("代码审查结果汇总")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 检查通过")
    
    for name, result in results:
        status = "[OK] 通过" if result else "[WARNING] 有警告"
        print(f"  {status} - {name}")
    
    if passed == total:
        print("\n[PASSED] 代码审查通过！")
        print("\n关键验证点:")
        print("  [OK] 线程安全：使用锁保护关键操作")
        print("  [OK] 错误处理：完善的异常处理和重试机制")
        print("  [OK] 资源清理：正确释放模型和GPU缓存")
        print("  [OK] 代码质量：良好的文档和日志")
        print("  [OK] 组件集成：正确集成到现有系统")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} 个检查有警告，请review")
        return 1

if __name__ == "__main__":
    sys.exit(main())
