"""
测试整合检测器
Test Integrated Detector
"""

import sys
from pathlib import Path
import os

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("❌ PIL未安装")
    sys.exit(1)

from src.page_detector_integrated import PageDetectorIntegrated


class MockADB:
    """模拟ADB，用于测试"""
    pass


def test_integrated_detector():
    """测试整合检测器"""
    print("=" * 70)
    print("测试整合检测器")
    print("=" * 70)
    
    # 创建模拟ADB
    adb = MockADB()
    
    # 初始化整合检测器
    print("\n[1] 初始化整合检测器...")
    detector = PageDetectorIntegrated(adb)
    
    # 检查初始化状态
    print(f"  - 页面分类器: {'✓' if detector._classifier_model else '✗'}")
    print(f"  - 类别列表: {'✓' if detector._classes else '✗'} ({len(detector._classes) if detector._classes else 0} 个类别)")
    print(f"  - YOLO注册表: {'✓' if detector._yolo_registry else '✗'} ({len(detector._yolo_registry) if detector._yolo_registry else 0} 个模型)")
    print(f"  - 页面-YOLO映射: {'✓' if detector._page_yolo_mapping else '✗'} ({len(detector._page_yolo_mapping) if detector._page_yolo_mapping else 0} 个页面)")
    
    # 检查"个人页_已登录"的映射
    print("\n[2] 检查'个人页_已登录'的YOLO映射...")
    page_class = '个人页_已登录'
    
    if page_class in detector._page_yolo_mapping:
        mapping = detector._page_yolo_mapping[page_class]
        yolo_models = mapping.get('yolo_models', [])
        
        print(f"  ✓ 找到映射配置")
        print(f"  - 页面状态: {mapping.get('page_state')}")
        print(f"  - YOLO模型数量: {len(yolo_models)}")
        
        for i, model_info in enumerate(yolo_models, 1):
            print(f"  - 模型{i}: {model_info.get('model_key')}")
            print(f"    目的: {model_info.get('purpose')}")
            print(f"    优先级: {model_info.get('priority')}")
    else:
        print(f"  ❌ 未找到映射配置")
        return
    
    # 加载测试图片
    print("\n[3] 加载测试图片...")
    test_image_path = None
    test_dirs = [
        '原始标注图/个人页_已登录_余额积分/images',
        '原始标注图/个人页_已登录_头像首页/images',
    ]
    
    for img_dir in test_dirs:
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image_path = os.path.join(img_dir, images[0])
                break
    
    if not test_image_path:
        print("  ❌ 未找到测试图片")
        return
    
    print(f"  ✓ 测试图片: {test_image_path}")
    image = Image.open(test_image_path)
    print(f"  ✓ 图片尺寸: {image.size}")
    
    # 测试页面分类
    print("\n[4] 测试页面分类...")
    page_class, confidence = detector._classify_page(image)
    
    if page_class:
        print(f"  ✓ 分类结果: {page_class}")
        print(f"  ✓ 置信度: {confidence:.2%}")
    else:
        print(f"  ❌ 分类失败")
        return
    
    # 测试元素检测
    print("\n[5] 测试YOLO元素检测...")
    print("-" * 70)
    
    elements = detector._detect_elements(image, page_class)
    
    print(f"\n检测结果:")
    print(f"  - 检测到的元素数量: {len(elements)}")
    
    if elements:
        print(f"\n元素详情:")
        for i, elem in enumerate(elements, 1):
            print(f"  {i}. {elem.class_name}")
            print(f"     - 置信度: {elem.confidence:.2%}")
            print(f"     - 边界框: {elem.bbox}")
            print(f"     - 中心点: {elem.center}")
        
        print(f"\n✅ 整合检测器YOLO检测成功")
    else:
        print(f"\n❌ 整合检测器未检测到任何元素")
        print(f"\n可能的原因:")
        print(f"  1. YOLO模型加载失败")
        print(f"  2. 页面-YOLO映射配置错误")
        print(f"  3. YOLO检测时出现异常")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == '__main__':
    test_integrated_detector()
