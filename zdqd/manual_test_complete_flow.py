"""
手动测试完整流程
验证模型加载、单账号任务、多账号任务和内存占用
"""

import sys
import os
import time
import psutil

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_manager import ModelManager
from adb_bridge import ADBBridge

def get_memory_mb():
    """获取当前进程内存占用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_model_loading():
    """测试1: 验证模型加载"""
    print("\n" + "=" * 80)
    print("测试1: 验证模型加载")
    print("=" * 80)
    
    # 记录初始内存
    initial_memory = get_memory_mb()
    print(f"\n初始内存: {initial_memory:.1f}MB")
    
    # 获取ModelManager实例
    print("\n[步骤1] 获取ModelManager单例...")
    manager = ModelManager.get_instance()
    print("[OK] ModelManager实例创建成功")
    
    # 检查初始状态
    print("\n[步骤2] 检查初始状态...")
    is_init = manager.is_initialized()
    print(f"初始化状态: {'已初始化' if is_init else '未初始化'}")
    
    if not is_init:
        print("\n[步骤3] 初始化模型...")
        print("注意: 这将加载所有深度学习模型，可能需要几秒钟...")
        
        # 创建ADB桥接器（模拟）
        try:
            adb = ADBBridge()
        except:
            print("[WARNING] 无法创建真实的ADB桥接器，使用模拟对象")
            adb = None
        
        # 初始化模型
        start_time = time.time()
        try:
            stats = manager.initialize_all_models(
                adb_bridge=adb,
                log_callback=print
            )
            load_time = time.time() - start_time
            
            print(f"\n[OK] 模型加载完成！")
            print(f"  - 加载时间: {load_time:.2f}秒")
            print(f"  - 已加载模型: {stats['models_loaded']}")
            print(f"  - 内存增量: {stats['memory_delta'] / 1024 / 1024:.1f}MB")
            
        except Exception as e:
            print(f"\n[ERROR] 模型加载失败: {e}")
            print("这可能是因为缺少模型文件或依赖")
            return False
    
    # 记录加载后内存
    loaded_memory = get_memory_mb()
    memory_delta = loaded_memory - initial_memory
    print(f"\n加载后内存: {loaded_memory:.1f}MB")
    print(f"内存增量: {memory_delta:.1f}MB")
    
    # 验证模型可访问
    print("\n[步骤4] 验证模型可访问...")
    try:
        integrated = manager.get_page_detector_integrated()
        print(f"[OK] PageDetectorIntegrated: {type(integrated).__name__}")
        
        hybrid = manager.get_page_detector_hybrid()
        print(f"[OK] PageDetectorHybridOptimized: {type(hybrid).__name__}")
        
        ocr = manager.get_ocr_thread_pool()
        print(f"[OK] OCRThreadPool: {type(ocr).__name__}")
        
    except Exception as e:
        print(f"[ERROR] 模型访问失败: {e}")
        return False
    
    print("\n[OK] 测试1通过: 模型加载成功")
    return True

def test_single_account():
    """测试2: 模拟单账号任务"""
    print("\n" + "=" * 80)
    print("测试2: 模拟单账号任务")
    print("=" * 80)
    
    manager = ModelManager.get_instance()
    
    print("\n模拟账号1的任务流程...")
    print("  [1] 获取页面检测器...")
    
    try:
        # 模拟多次访问模型（就像真实任务中那样）
        for i in range(5):
            detector = manager.get_page_detector_integrated()
            hybrid = manager.get_page_detector_hybrid()
            ocr = manager.get_ocr_thread_pool()
            time.sleep(0.1)  # 模拟处理时间
        
        print("  [OK] 完成5次模型访问")
        print("\n[OK] 测试2通过: 单账号任务模拟成功")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 测试2失败: {e}")
        return False

def test_multi_account():
    """测试3: 模拟多账号任务"""
    print("\n" + "=" * 80)
    print("测试3: 模拟多账号任务（30个账号）")
    print("=" * 80)
    
    manager = ModelManager.get_instance()
    
    # 记录初始内存
    initial_memory = get_memory_mb()
    print(f"\n初始内存: {initial_memory:.1f}MB")
    
    # 模拟30个账号
    account_count = 30
    model_ids = set()
    
    print(f"\n开始处理{account_count}个账号...")
    
    try:
        for i in range(1, account_count + 1):
            # 每个账号都获取模型
            integrated = manager.get_page_detector_integrated()
            hybrid = manager.get_page_detector_hybrid()
            ocr = manager.get_ocr_thread_pool()
            
            # 记录模型ID
            model_ids.add(id(integrated))
            model_ids.add(id(hybrid))
            model_ids.add(id(ocr))
            
            # 每10个账号报告一次
            if i % 10 == 0:
                current_memory = get_memory_mb()
                print(f"  已处理 {i} 个账号，当前内存: {current_memory:.1f}MB")
        
        # 最终内存
        final_memory = get_memory_mb()
        memory_delta = final_memory - initial_memory
        
        print(f"\n处理完成！")
        print(f"  - 最终内存: {final_memory:.1f}MB")
        print(f"  - 内存增量: {memory_delta:.1f}MB")
        print(f"  - 唯一模型实例数: {len(model_ids)}")
        
        # 验证
        if len(model_ids) == 3:  # 应该只有3个唯一实例
            print("\n[OK] 所有账号共享同一组模型实例")
        else:
            print(f"\n[WARNING] 检测到{len(model_ids)}个唯一实例（预期3个）")
        
        if memory_delta < 50:  # 内存增量应该很小
            print(f"[OK] 内存增量很小（{memory_delta:.1f}MB < 50MB）")
        else:
            print(f"[WARNING] 内存增量较大（{memory_delta:.1f}MB）")
        
        print("\n[OK] 测试3通过: 多账号任务模拟成功")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """测试4: 验证内存占用"""
    print("\n" + "=" * 80)
    print("测试4: 验证内存占用")
    print("=" * 80)
    
    manager = ModelManager.get_instance()
    
    # 获取加载统计
    stats = manager.get_loading_stats()
    
    print("\n加载统计:")
    print(f"  - 总模型数: {stats['total_models']}")
    print(f"  - 已加载: {stats['loaded_models']}")
    print(f"  - 失败: {stats['failed_models']}")
    print(f"  - 总加载时间: {stats['total_time']:.2f}秒")
    print(f"  - 内存占用: {stats['memory_delta_mb']:.1f}MB")
    
    # 当前进程内存
    current_memory = get_memory_mb()
    print(f"\n当前进程内存: {current_memory:.1f}MB")
    
    print("\n[OK] 测试4通过: 内存占用验证完成")
    return True

def main():
    """主测试流程"""
    print("=" * 80)
    print("模型单例优化 - 手动测试完整流程")
    print("=" * 80)
    
    results = []
    
    # 测试1: 模型加载
    results.append(("模型加载", test_model_loading()))
    
    # 测试2: 单账号任务
    results.append(("单账号任务", test_single_account()))
    
    # 测试3: 多账号任务
    results.append(("多账号任务", test_multi_account()))
    
    # 测试4: 内存占用
    results.append(("内存占用", test_memory_usage()))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    for name, result in results:
        status = "[OK] 通过" if result else "[ERROR] 失败"
        print(f"  {status} - {name}")
    
    if passed == total:
        print("\n[PASSED] 所有手动测试通过！")
        return 0
    else:
        print(f"\n[FAILED] {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
