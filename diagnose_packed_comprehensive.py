#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全面诊断打包后的程序 - 检查所有关键模块
"""

import sys
import os
import traceback

# 设置工作目录
packed_dir = r"D:\溪盟商城自动化助手_打包_新"
os.chdir(packed_dir)
sys.path.insert(0, os.path.join(packed_dir, '_internal'))

print("=" * 80)
print("全面诊断打包后的程序")
print("=" * 80)
print(f"\n当前目录: {os.getcwd()}")
print(f"Python路径: {sys.path[:3]}")

# 测试结果统计
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_module(name, test_func):
    """测试模块并记录结果"""
    print(f"\n{'='*80}")
    print(f"[测试] {name}")
    print('='*80)
    try:
        result = test_func()
        if result:
            test_results['passed'].append(name)
            print(f"✓ {name} - 通过")
        else:
            test_results['failed'].append(name)
            print(f"✗ {name} - 失败")
        return result
    except Exception as e:
        test_results['failed'].append(name)
        print(f"✗ {name} - 异常: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# 测试1: 基础模块导入
# ============================================================================
def test_basic_imports():
    """测试基础模块导入"""
    modules = [
        'yaml',
        'cv2',
        'PIL',
        'numpy',
        'pandas',
        'torch',
        'ultralytics',
        'rapidocr_onnxruntime',
        'cryptography',
        'psutil',
        'imagehash',
        'tkinter',
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n  失败的模块: {failed}")
        return False
    return True

# ============================================================================
# 测试2: src模块导入
# ============================================================================
def test_src_imports():
    """测试src模块导入"""
    modules = [
        'src.page_state_dynamic',
        'src.page_detector_integrated',
        'src.page_detector',
        'src.page_detector_dl',
        'src.page_detector_cache',
        'src.yolo_button_detector',
        'src.smart_button_clicker',
        'src.ocr_thread_pool',
        'src.ocr_enhancer',
        'src.model_manager',
        'src.adb_bridge',
        'src.navigator',
        'src.auto_login',
        'src.daily_checkin',
        'src.wait_helper',
        'src.performance.smart_waiter',
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n  失败的模块: {failed}")
        return False
    return True

# ============================================================================
# 测试3: 配置文件检查
# ============================================================================
def test_config_files():
    """测试配置文件"""
    config_files = {
        'config.yaml': 'config.yaml',
        'page_state_mapping.json': 'config/page_state_mapping.json',
        'page_classes.json': 'models/page_classes.json',
        'yolo_model_registry.json': 'config/yolo_model_registry.json',
        'page_yolo_mapping.json': 'models/page_yolo_mapping.json',
    }
    
    missing = []
    for name, path in config_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {name}: {size} 字节")
        else:
            print(f"  ✗ {name}: 不存在 ({path})")
            missing.append(name)
    
    if missing:
        print(f"\n  缺失的配置文件: {missing}")
        return False
    return True

# ============================================================================
# 测试4: 模型文件检查
# ============================================================================
def test_model_files():
    """测试模型文件"""
    model_files = {
        '页面分类器': 'models/page_classifier_pytorch_best.pth',
        'YOLO模型1': 'models/yolo26n.pt',
        'YOLO模型2': 'models/yolov8n.pt',
        'OCR检测模型': 'models/ch_PP-OCRv4_det_infer.onnx',
        'OCR识别模型': 'models/ch_PP-OCRv4_rec_infer.onnx',
        'OCR方向分类': 'models/ch_ppocr_mobile_v2.0_cls_infer.onnx',
    }
    
    missing = []
    for name, path in model_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024  # MB
            print(f"  ✓ {name}: {size:.2f} MB")
        else:
            print(f"  ✗ {name}: 不存在 ({path})")
            missing.append(name)
    
    if missing:
        print(f"\n  缺失的模型文件: {missing}")
        return False
    return True

# ============================================================================
# 测试5: PageState加载
# ============================================================================
def test_page_state():
    """测试PageState加载"""
    from src.page_state_dynamic import PageState
    
    print(f"  配置路径: {PageState._config_path}")
    print(f"  已加载: {PageState._loaded}")
    print(f"  状态数量: {len(PageState._states)}")
    
    # 测试关键状态
    critical_states = ['UNKNOWN', 'POPUP', 'HOME', 'PROFILE', 'CHECKIN', 'LOGIN']
    missing = []
    for state_name in critical_states:
        state = getattr(PageState, state_name, None)
        if state and state.value != 'unknown':
            print(f"  ✓ {state_name}: {state.value}")
        else:
            print(f"  ✗ {state_name}: 缺失或错误")
            missing.append(state_name)
    
    if missing:
        print(f"\n  缺失的关键状态: {missing}")
        return False
    return True

# ============================================================================
# 测试6: 整合检测器初始化
# ============================================================================
def test_integrated_detector():
    """测试整合检测器"""
    from src.page_detector_integrated import PageDetectorIntegrated
    from src.adb_bridge import ADBBridge
    
    adb = ADBBridge()
    detector = PageDetectorIntegrated(adb)
    
    print(f"  分类器已加载: {detector._classifier_model is not None}")
    print(f"  类别数量: {len(detector._classes) if detector._classes else 0}")
    print(f"  状态映射数量: {len(detector._class_to_state)}")
    print(f"  YOLO检测器数量: {len(detector._yolo_detectors)}")
    
    # 检查关键属性
    checks = {
        '分类器模型': detector._classifier_model is not None,
        '类别列表': detector._classes is not None and len(detector._classes) > 0,
        '状态映射': len(detector._class_to_state) > 0,
        'YOLO检测器': len(detector._yolo_detectors) > 0,
    }
    
    failed = [k for k, v in checks.items() if not v]
    if failed:
        print(f"\n  失败的检查: {failed}")
        return False
    return True

# ============================================================================
# 测试7: OCR线程池
# ============================================================================
def test_ocr_thread_pool():
    """测试OCR线程池"""
    from src.ocr_thread_pool import OCRThreadPool
    
    pool = OCRThreadPool()
    print(f"  OCR引擎: {pool.ocr_engine}")
    print(f"  线程池大小: {pool.max_workers}")
    print(f"  已初始化: {pool.ocr_engine is not None}")
    
    if pool.ocr_engine is None:
        print(f"  ✗ OCR引擎未初始化")
        return False
    
    return True

# ============================================================================
# 测试8: YOLO按钮检测器
# ============================================================================
def test_yolo_detector():
    """测试YOLO按钮检测器"""
    from src.yolo_button_detector import YOLOButtonDetector
    
    # 检查YOLO模型注册表
    registry_path = 'config/yolo_model_registry.json'
    if not os.path.exists(registry_path):
        print(f"  ✗ YOLO模型注册表不存在")
        return False
    
    import json
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    print(f"  注册的YOLO模型数量: {len(registry)}")
    print(f"  前5个模型: {list(registry.keys())[:5]}")
    
    # 尝试创建检测器
    try:
        detector = YOLOButtonDetector('首页公告')
        print(f"  ✓ 成功创建YOLO检测器")
        return True
    except Exception as e:
        print(f"  ✗ 创建YOLO检测器失败: {e}")
        return False

# ============================================================================
# 测试9: 智能等待器
# ============================================================================
def test_smart_waiter():
    """测试智能等待器"""
    from src.performance.smart_waiter import SmartWaiter
    from src.adb_bridge import ADBBridge
    
    adb = ADBBridge()
    
    try:
        waiter = SmartWaiter(adb)
        print(f"  ✓ 智能等待器创建成功")
        print(f"  检测器: {waiter.detector}")
        return True
    except Exception as e:
        print(f"  ✗ 智能等待器创建失败: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# 测试10: 模型管理器
# ============================================================================
def test_model_manager():
    """测试模型管理器"""
    from src.model_manager import ModelManager
    
    manager = ModelManager.get_instance()
    print(f"  模型管理器实例: {manager}")
    print(f"  配置文件: {manager.config_file}")
    
    # 检查配置
    if not os.path.exists(manager.config_file):
        print(f"  ⚠ 配置文件不存在，使用默认配置")
    
    # 检查模型路径
    print(f"  模型目录: {manager.models_dir}")
    print(f"  模型目录存在: {os.path.exists(manager.models_dir)}")
    
    return True

# ============================================================================
# 测试11: 导航器
# ============================================================================
def test_navigator():
    """测试导航器"""
    from src.navigator import Navigator
    from src.adb_bridge import ADBBridge
    from src.model_manager import ModelManager
    
    adb = ADBBridge()
    manager = ModelManager.get_instance()
    
    try:
        # 尝试初始化模型
        print(f"  尝试初始化模型...")
        manager.initialize_all_models(adb)
        print(f"  ✓ 模型初始化成功")
        
        # 创建导航器
        nav = Navigator(adb)
        print(f"  ✓ 导航器创建成功")
        print(f"  检测器: {nav.detector}")
        return True
    except Exception as e:
        print(f"  ✗ 导航器测试失败: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# 测试12: 检查PyTorch和CUDA
# ============================================================================
def test_pytorch_cuda():
    """测试PyTorch和CUDA"""
    import torch
    
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  当前GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠ CUDA不可用，将使用CPU")
    
    return True

# ============================================================================
# 测试13: 检查Ultralytics YOLO
# ============================================================================
def test_ultralytics():
    """测试Ultralytics YOLO"""
    from ultralytics import YOLO
    
    # 检查YOLO模型文件
    yolo_model = 'models/yolo26n.pt'
    if not os.path.exists(yolo_model):
        print(f"  ✗ YOLO模型不存在: {yolo_model}")
        return False
    
    try:
        model = YOLO(yolo_model)
        print(f"  ✓ YOLO模型加载成功")
        print(f"  模型类型: {type(model)}")
        return True
    except Exception as e:
        print(f"  ✗ YOLO模型加载失败: {e}")
        return False

# ============================================================================
# 执行所有测试
# ============================================================================
print("\n开始执行测试...\n")

test_module("1. 基础模块导入", test_basic_imports)
test_module("2. src模块导入", test_src_imports)
test_module("3. 配置文件检查", test_config_files)
test_module("4. 模型文件检查", test_model_files)
test_module("5. PageState加载", test_page_state)
test_module("6. 整合检测器初始化", test_integrated_detector)
test_module("7. OCR线程池", test_ocr_thread_pool)
test_module("8. YOLO按钮检测器", test_yolo_detector)
test_module("9. 智能等待器", test_smart_waiter)
test_module("10. 模型管理器", test_model_manager)
test_module("11. PyTorch和CUDA", test_pytorch_cuda)
test_module("12. Ultralytics YOLO", test_ultralytics)
test_module("13. 导航器", test_navigator)

# ============================================================================
# 输出测试结果
# ============================================================================
print("\n" + "=" * 80)
print("测试结果汇总")
print("=" * 80)

print(f"\n✓ 通过: {len(test_results['passed'])} 项")
for test in test_results['passed']:
    print(f"  - {test}")

if test_results['failed']:
    print(f"\n✗ 失败: {len(test_results['failed'])} 项")
    for test in test_results['failed']:
        print(f"  - {test}")

if test_results['warnings']:
    print(f"\n⚠ 警告: {len(test_results['warnings'])} 项")
    for test in test_results['warnings']:
        print(f"  - {test}")

print("\n" + "=" * 80)
if test_results['failed']:
    print("✗ 诊断发现问题，需要修复")
else:
    print("✓ 所有测试通过")
print("=" * 80)
