"""
初始化ModelManager用于测试
"""

import sys
import asyncio
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.model_manager import ModelManager
from src.adb_bridge import ADBBridge


async def init_models():
    """初始化所有模型"""
    print("\n" + "="*60)
    print("初始化ModelManager")
    print("="*60)
    
    # 创建ADB桥接器
    adb = ADBBridge()
    
    # 获取ModelManager实例
    model_manager = ModelManager.get_instance()
    print(f"✓ ModelManager实例: {id(model_manager)}")
    
    # 初始化所有模型
    print("\n开始加载模型...")
    
    def progress_callback(message, current, total):
        print(f"[{current}/{total}] {message}")
    
    def log_callback(message):
        print(f"  {message}")
    
    try:
        # initialize_all_models是同步方法，不需要await
        stats = model_manager.initialize_all_models(
            adb_bridge=adb,
            log_callback=log_callback,
            progress_callback=progress_callback
        )
        
        print("\n" + "="*60)
        print("模型加载完成")
        print("="*60)
        print(f"✓ 成功: {stats['success']}")
        print(f"✓ 已加载模型: {', '.join(stats['models_loaded'])}")
        print(f"✓ 总耗时: {stats['total_time']:.2f}秒")
        print(f"✓ 内存占用: {stats['memory_after'] / 1024 / 1024:.1f}MB")
        
        if stats['errors']:
            print(f"\n⚠ 错误:")
            for error in stats['errors']:
                print(f"  - {error}")
        
        return stats['success']
        
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 运行初始化（不需要asyncio.run，因为initialize_all_models是同步的）
    # 但是为了保持一致性，我们还是使用async函数
    success = asyncio.run(init_models())
    
    if success:
        print("\n✓ ModelManager初始化成功，可以运行测试了")
    else:
        print("\n✗ ModelManager初始化失败")


if __name__ == '__main__':
    main()
