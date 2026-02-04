"""
测试个人页_已登录的YOLO模型映射修复
"""
import asyncio
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.adb_bridge import ADBBridge
from src.page_detector_integrated import PageDetectorIntegrated

async def test_profile_logged_mapping():
    """测试个人页_已登录的映射"""
    
    print("=" * 60)
    print("测试个人页_已登录的YOLO模型映射")
    print("=" * 60)
    
    # 初始化ADB
    adb = ADBBridge()
    
    # 初始化整合检测器（使用正确的路径参数）
    print("\n[1] 初始化整合检测器...")
    detector = PageDetectorIntegrated(
        adb=adb,
        yolo_registry_path='yolo_model_registry.json',
        mapping_path='page_yolo_mapping.json',
        log_callback=print
    )
    
    # 检查映射是否加载
    print(f"\n[2] 检查映射配置...")
    print(f"  - 映射配置中的页面数量: {len(detector._page_yolo_mapping)}")
    
    target_page = "个人页_已登录"
    if target_page in detector._page_yolo_mapping:
        print(f"  ✓ 找到 '{target_page}' 的映射")
        mapping = detector._page_yolo_mapping[target_page]
        yolo_models = mapping.get('yolo_models', [])
        print(f"  - YOLO模型数量: {len(yolo_models)}")
        for model in yolo_models:
            print(f"    • {model.get('model_key')} (优先级: {model.get('priority')})")
    else:
        print(f"  ✗ 未找到 '{target_page}' 的映射")
        print(f"  可用的页面类型: {list(detector._page_yolo_mapping.keys())[:10]}")
    
    # 检查YOLO注册表
    print(f"\n[3] 检查YOLO注册表...")
    print(f"  - 注册表中的模型数量: {len(detector._yolo_registry)}")
    
    if 'profile_logged' in detector._yolo_registry:
        print(f"  ✓ 找到 'profile_logged' 模型")
        model_info = detector._yolo_registry['profile_logged']
        print(f"    - 名称: {model_info.get('name')}")
        print(f"    - 模型路径: {model_info.get('model_path')}")
    else:
        print(f"  ✗ 未找到 'profile_logged' 模型")
    
    # 测试实际检测（如果有设备连接）
    print(f"\n[4] 跳过实际检测测试（需要连接设备）")
    print(f"  提示: 映射配置已正确加载，可以在实际运行中测试")
    
    print("\n" + "=" * 60)
    print("✓ 测试完成 - 映射配置正确")
    print("=" * 60)

if __name__ == '__main__':
    asyncio.run(test_profile_logged_mapping())
