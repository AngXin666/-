#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断打包后的程序问题
"""

import sys
import os

print("=" * 60)
print("诊断打包后的程序")
print("=" * 60)

# 设置工作目录
packed_dir = r"D:\溪盟商城自动化助手_打包_新"
os.chdir(packed_dir)
sys.path.insert(0, os.path.join(packed_dir, '_internal'))

print(f"\n当前目录: {os.getcwd()}")

# 测试PageState加载
print("\n[测试1] PageState配置加载...")
try:
    from src.page_state_dynamic import PageState
    
    # 检查配置路径
    print(f"  配置路径: {PageState._config_path}")
    print(f"  已加载: {PageState._loaded}")
    print(f"  状态数量: {len(PageState._states)}")
    
    # 列出前10个状态
    print(f"  前10个状态:")
    for i, (name, state) in enumerate(list(PageState._states.items())[:10]):
        print(f"    {name}: {state.value} ({state.chinese_name})")
    
    # 测试关键状态
    print(f"\n  测试关键状态:")
    print(f"    HOME_NOTICE: {PageState.HOME_NOTICE}")
    print(f"    PROFILE: {PageState.PROFILE}")
    print(f"    CHECKIN: {PageState.CHECKIN}")
    print(f"    POPUP: {PageState.POPUP}")
    
except Exception as e:
    print(f"  ✗ PageState加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试整合检测器初始化
print("\n[测试2] 整合检测器初始化...")
try:
    from src.page_detector_integrated import PageDetectorIntegrated
    from src.adb_bridge import ADBBridge
    
    # 创建ADB（不连接设备）
    adb = ADBBridge()
    
    # 创建检测器
    print("  创建检测器...")
    detector = PageDetectorIntegrated(adb)
    
    print(f"  ✓ 检测器创建成功")
    print(f"  分类器已加载: {detector._classifier_model is not None}")
    print(f"  类别数量: {len(detector._classes) if detector._classes else 0}")
    print(f"  状态映射数量: {len(detector._class_to_state)}")
    
    # 检查映射
    if detector._class_to_state:
        print(f"\n  前10个映射:")
        for i, (class_name, state) in enumerate(list(detector._class_to_state.items())[:10]):
            print(f"    {class_name} -> {state.name}")
    else:
        print(f"  ⚠️ 警告: 状态映射为空！")
    
    # 检查类别列表
    if detector._classes:
        print(f"\n  前10个类别:")
        for i, class_name in enumerate(detector._classes[:10]):
            print(f"    {i}: {class_name}")
    
except Exception as e:
    print(f"  ✗ 检测器初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试配置文件
print("\n[测试3] 配置文件检查...")
config_files = {
    'page_state_mapping.json': 'config/page_state_mapping.json',
    'page_classes.json': 'models/page_classes.json',
    'yolo_model_registry.json': 'config/yolo_model_registry.json',
}

for name, path in config_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  ✓ {name}: {size} 字节")
        
        # 读取并显示内容摘要
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if name == 'page_state_mapping.json':
                mappings = data.get('mappings', {})
                print(f"    映射数量: {len(mappings)}")
                print(f"    前5个映射: {list(mappings.keys())[:5]}")
            elif name == 'page_classes.json':
                print(f"    类别数量: {len(data)}")
                print(f"    前5个类别: {data[:5]}")
        except Exception as e:
            print(f"    ⚠️ 读取失败: {e}")
    else:
        print(f"  ✗ {name}: 不存在")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
