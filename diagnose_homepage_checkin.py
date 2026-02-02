"""
诊断首页识别和签到按钮检测问题
"""
import asyncio
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adb_bridge import ADBBridge
from model_manager import ModelManager

async def diagnose():
    """诊断首页和签到功能"""
    print("=" * 60)
    print("诊断首页识别和签到按钮检测")
    print("=" * 60)
    
    # 初始化ADB
    adb = ADBBridge()
    devices = adb.list_devices()
    
    if not devices:
        print("❌ 没有找到设备")
        return
    
    device_id = devices[0]
    print(f"\n使用设备: {device_id}")
    
    # 初始化ModelManager
    print("\n[1/4] 初始化模型管理器...")
    model_manager = ModelManager.get_instance()
    await model_manager.initialize()
    
    # 获取整合检测器
    detector = model_manager.get_integrated_detector()
    print("✓ 整合检测器已加载")
    
    # 检测当前页面
    print("\n[2/4] 检测当前页面状态...")
    page_result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
    
    if page_result:
        print(f"  页面类型: {page_result.state.value}")
        print(f"  置信度: {page_result.confidence:.2%}")
        print(f"  检测详情: {page_result.details}")
    else:
        print("  ❌ 无法检测页面")
        return
    
    # 如果是首页,检测签到按钮
    if page_result.state.value == "HOME":
        print("\n[3/4] 检测首页签到按钮...")
        
        # 使用整合检测器检测元素
        detection_result = await detector.detect_page(
            device_id,
            use_cache=False,
            detect_elements=True
        )
        
        if detection_result and detection_result.elements:
            print(f"  检测到 {len(detection_result.elements)} 个元素:")
            for element in detection_result.elements:
                print(f"    - {element.class_name}: {element.center} (置信度: {element.confidence:.2%})")
                
            # 查找签到按钮
            checkin_button = None
            for element in detection_result.elements:
                if '每日签到' in element.class_name or '签到按钮' in element.class_name:
                    checkin_button = element
                    break
            
            if checkin_button:
                print(f"\n  ✓ 找到签到按钮: {checkin_button.center}")
                print(f"    置信度: {checkin_button.confidence:.2%}")
            else:
                print(f"\n  ❌ 未找到签到按钮")
        else:
            print("  ❌ 未检测到任何元素")
    else:
        print(f"\n[3/4] 当前不在首页,在 {page_result.state.value}")
    
    # 检查模型配置
    print("\n[4/4] 检查模型配置...")
    import json
    
    with open('yolo_model_registry.json', 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    with open('page_yolo_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    # 检查首页模型
    if '首页' in registry['models']:
        homepage_model = registry['models']['首页']
        print(f"  首页模型:")
        print(f"    - 路径: {homepage_model['model_path']}")
        print(f"    - 类别: {homepage_model['classes']}")
        print(f"    - mAP50: {homepage_model['performance']['mAP50']:.3f}")
        
        # 检查模型文件是否存在
        model_path = Path(homepage_model['model_path'])
        if model_path.exists():
            print(f"    - 文件: ✓ 存在")
        else:
            print(f"    - 文件: ✗ 不存在")
    
    # 检查映射
    if '首页' in mapping['mapping']:
        homepage_mapping = mapping['mapping']['首页']
        print(f"  首页映射:")
        print(f"    - 使用模型: {homepage_mapping['yolo_models'][0]['model_key']}")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == '__main__':
    asyncio.run(diagnose())
