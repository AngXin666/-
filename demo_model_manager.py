"""
ModelManager 使用演示

展示如何使用ModelManager管理模型
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_manager import ModelManager


def demo_basic_usage():
    """演示基本使用"""
    print("\n" + "=" * 60)
    print("演示1: 基本使用")
    print("=" * 60)
    
    # 获取单例实例
    manager = ModelManager.get_instance()
    print(f"✓ 获取ModelManager实例: {id(manager)}")
    
    # 检查初始化状态
    is_init = manager.is_initialized()
    print(f"✓ 模型是否已初始化: {is_init}")
    
    # 查看配置
    config = manager._config
    print(f"✓ 配置加载成功")
    print(f"  - 模型配置数量: {len(config['models'])}")
    print(f"  - 启用的模型:")
    for model_name, model_config in config['models'].items():
        enabled = model_config.get('enabled', True)
        print(f"    - {model_name}: {'启用' if enabled else '禁用'}")


def demo_singleton_pattern():
    """演示单例模式"""
    print("\n" + "=" * 60)
    print("演示2: 单例模式")
    print("=" * 60)
    
    # 多种方式获取实例
    instance1 = ModelManager.get_instance()
    instance2 = ModelManager()
    instance3 = ModelManager.get_instance()
    
    print(f"✓ 通过get_instance()获取: {id(instance1)}")
    print(f"✓ 通过构造函数获取: {id(instance2)}")
    print(f"✓ 再次通过get_instance()获取: {id(instance3)}")
    print(f"✓ 所有实例都是同一个对象: {instance1 is instance2 is instance3}")


def demo_config_loading():
    """演示配置加载"""
    print("\n" + "=" * 60)
    print("演示3: 配置加载")
    print("=" * 60)
    
    manager = ModelManager.get_instance()
    
    # 显示配置详情
    print("✓ 深度学习页面分类器配置:")
    integrated_config = manager._config['models']['page_detector_integrated']
    for key, value in integrated_config.items():
        print(f"  - {key}: {value}")
    
    print("\n✓ YOLO检测器配置:")
    hybrid_config = manager._config['models']['page_detector_hybrid']
    for key, value in hybrid_config.items():
        print(f"  - {key}: {value}")
    
    print("\n✓ OCR线程池配置:")
    ocr_config = manager._config['models']['ocr_thread_pool']
    for key, value in ocr_config.items():
        print(f"  - {key}: {value}")
    
    print("\n✓ 启动配置:")
    startup_config = manager._config['startup']
    for key, value in startup_config.items():
        print(f"  - {key}: {value}")


def demo_helper_methods():
    """演示辅助方法"""
    print("\n" + "=" * 60)
    print("演示4: 辅助方法")
    print("=" * 60)
    
    manager = ModelManager.get_instance()
    
    # 测试_is_model_enabled
    print("✓ 检查模型是否启用:")
    models = ['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool', 'nonexistent']
    for model_name in models:
        enabled = manager._is_model_enabled(model_name)
        print(f"  - {model_name}: {'启用' if enabled else '禁用/不存在'}")
    
    # 测试_is_critical_model
    print("\n✓ 检查是否是关键模型:")
    for model_name in models[:3]:
        critical = manager._is_critical_model(model_name)
        print(f"  - {model_name}: {'关键模型' if critical else '可选模型'}")
    
    # 测试is_initialized
    print(f"\n✓ 模型是否已初始化: {manager.is_initialized()}")


def demo_loading_stats():
    """演示加载统计"""
    print("\n" + "=" * 60)
    print("演示5: 加载统计")
    print("=" * 60)
    
    manager = ModelManager.get_instance()
    
    # 获取加载统计
    stats = manager.get_loading_stats()
    
    print("✓ 加载统计信息:")
    print(f"  - 总模型数: {stats['total_models']}")
    print(f"  - 已加载模型数: {stats['loaded_models']}")
    print(f"  - 加载失败模型数: {stats['failed_models']}")
    print(f"  - 总加载时间: {stats['total_time']:.2f}秒")
    print(f"  - 加载前内存: {stats['memory_before'] / 1024 / 1024:.1f}MB")
    print(f"  - 加载后内存: {stats['memory_after'] / 1024 / 1024:.1f}MB")
    print(f"  - 内存增量: {stats['memory_delta'] / 1024 / 1024:.1f}MB")
    
    if stats['errors']:
        print(f"  - 错误列表:")
        for error in stats['errors']:
            print(f"    - {error}")
    
    if stats['model_times']:
        print(f"  - 各模型加载时间:")
        for model_name, load_time in stats['model_times'].items():
            print(f"    - {model_name}: {load_time:.2f}秒")


def demo_config_merge():
    """演示配置合并"""
    print("\n" + "=" * 60)
    print("演示6: 配置合并")
    print("=" * 60)
    
    manager = ModelManager.get_instance()
    
    # 示例配置
    default = {
        'models': {
            'model_a': {'enabled': True, 'path': 'default.pth'},
            'model_b': {'enabled': True, 'path': 'default.pth'}
        },
        'startup': {
            'show_progress': True
        }
    }
    
    user = {
        'models': {
            'model_a': {'path': 'custom.pth'},  # 只覆盖path
            'model_c': {'enabled': True, 'path': 'new.pth'}  # 新增模型
        },
        'startup': {
            'log_loading_time': True  # 新增配置
        }
    }
    
    merged = manager._merge_config(default, user)
    
    print("✓ 默认配置:")
    print(f"  {default}")
    
    print("\n✓ 用户配置:")
    print(f"  {user}")
    
    print("\n✓ 合并后配置:")
    print(f"  {merged}")
    
    print("\n✓ 合并结果验证:")
    print(f"  - model_a.enabled保留默认值: {merged['models']['model_a']['enabled'] == True}")
    print(f"  - model_a.path使用用户值: {merged['models']['model_a']['path'] == 'custom.pth'}")
    print(f"  - model_b完全保留: {merged['models']['model_b'] == default['models']['model_b']}")
    print(f"  - model_c新增成功: {'model_c' in merged['models']}")
    print(f"  - startup.show_progress保留: {merged['startup']['show_progress'] == True}")
    print(f"  - startup.log_loading_time新增: {merged['startup']['log_loading_time'] == True}")


def main():
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("ModelManager 使用演示")
    print("=" * 60)
    
    try:
        demo_basic_usage()
        demo_singleton_pattern()
        demo_config_loading()
        demo_helper_methods()
        demo_loading_stats()
        demo_config_merge()
        
        print("\n" + "=" * 60)
        print("✓ 演示完成！")
        print("=" * 60)
        print("\n提示：")
        print("  - 查看 src/MODEL_MANAGER_README.md 了解详细使用说明")
        print("  - 复制 model_config.json.example 为 model_config.json 自定义配置")
        print("  - 运行 test_model_manager_basic.py 查看测试结果")
        
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
