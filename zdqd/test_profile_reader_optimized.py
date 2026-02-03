"""
优化版 ProfileReader 测试 - 只对 YOLO 检测区域做 OCR
Optimized ProfileReader Test - OCR only on YOLO detected regions
"""

import sys
from pathlib import Path
import os
import asyncio
import time
import re
from io import BytesIO

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
from src.profile_reader import ProfileReader


class MockADB:
    """模拟ADB，用于测试"""
    
    def __init__(self, test_image_path):
        self.test_image_path = test_image_path
    
    async def screencap(self, device_id: str) -> bytes:
        """返回测试图片的字节数据"""
        with open(self.test_image_path, 'rb') as f:
            return f.read()


async def test_optimized_profile_reader():
    """测试优化版 ProfileReader"""
    print("=" * 70)
    print("优化版 ProfileReader 测试 - 区域 OCR")
    print("=" * 70)
    
    # 查找测试图片
    print("\n[1] 查找测试图片...")
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
        print("❌ 未找到测试图片")
        return
    
    print(f"✓ 测试图片: {test_image_path}")
    
    # 创建模拟ADB
    adb = MockADB(test_image_path)
    
    # 使用ModelManager初始化所有模型
    print("\n[2] 初始化ModelManager...")
    from src.model_manager import ModelManager
    
    try:
        manager = ModelManager.get_instance()
        
        # 初始化所有模型
        print("  正在加载模型...")
        stats = manager.initialize_all_models(
            adb_bridge=adb,
            log_callback=lambda msg: None
        )
        
        print(f"  ✓ ModelManager初始化完成")
        print(f"    - 加载模型数: {stats['models_loaded']}")
        print(f"    - 总耗时: {stats['total_time']:.2f}秒")
        
        # 从ModelManager获取整合检测器
        detector = manager.get_page_detector_integrated()
        ocr_pool = manager.get_ocr_thread_pool()
        
    except Exception as e:
        print(f"  ❌ ModelManager初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试优化方案
    print("\n[3] 测试优化方案：只对 YOLO 区域做 OCR")
    print("=" * 70)
    
    device_id = "test_device"
    
    try:
        total_start = time.time()
        
        # 步骤1：获取截图
        screenshot_data = await adb.screencap(device_id)
        image = Image.open(BytesIO(screenshot_data))
        print(f"✓ 截图完成")
        
        # 步骤2：YOLO 检测元素位置
        yolo_start = time.time()
        detection_result = await detector.detect_page(
            device_id, 
            use_cache=False, 
            detect_elements=True
        )
        yolo_time = time.time() - yolo_start
        print(f"✓ YOLO 检测完成: {len(detection_result.elements)} 个元素, 耗时 {yolo_time:.3f}秒")
        
        # 步骤3：对比两种方案
        from src.ocr_image_processor import enhance_for_ocr
        
        result = {
            'nickname': None,
            'user_id': None,
            'balance': None,
            'points': None,
            'vouchers': None,
            'coupons': None
        }
        
        # === 方案A：全屏 OCR（当前方案）===
        print(f"\n[方案A] 全屏 OCR...")
        full_ocr_start = time.time()
        enhanced_image = enhance_for_ocr(image)
        full_ocr_result = await ocr_pool.recognize(enhanced_image, timeout=10.0)
        full_ocr_time = time.time() - full_ocr_start
        print(f"  耗时: {full_ocr_time:.3f}秒, 识别到 {len(full_ocr_result.texts) if full_ocr_result else 0} 个文本")
        
        # === 方案B：区域 OCR（优化方案）===
        print(f"\n[方案B] 区域 OCR（只识别 YOLO 检测区域）...")
        region_ocr_start = time.time()
        
        # 并行 OCR 识别所有区域
        ocr_tasks = []
        element_map = []
        
        for element in detection_result.elements:
            x1, y1, x2, y2 = element.bbox
            # 从全屏图片裁剪（不需要重新截图）
            region = image.crop((x1, y1, x2, y2))
            region_enhanced = enhance_for_ocr(region)
            
            # 记录元素类型和 OCR 任务
            element_map.append((element.class_name, element))
            ocr_tasks.append(ocr_pool.recognize(region_enhanced, timeout=3.0))
        
        # 并行执行所有 OCR
        ocr_results = await asyncio.gather(*ocr_tasks)
        
        region_ocr_time = time.time() - region_ocr_start
        print(f"  耗时: {region_ocr_time:.3f}秒, 识别了 {len(ocr_results)} 个区域")
        
        # 计算加速比
        speedup = full_ocr_time / region_ocr_time if region_ocr_time > 0 else 0
        print(f"\n  🚀 区域 OCR 比全屏 OCR 快 {speedup:.1f}x")
        
        # 步骤4：解析区域 OCR 结果
        for i, (class_name, element) in enumerate(element_map):
            ocr_result = ocr_results[i]
            
            if not ocr_result or not ocr_result.texts:
                continue
            
            texts = ocr_result.texts
            combined_text = ' '.join(texts)
            
            # 处理昵称
            if '昵称' in class_name and result['nickname'] is None:
                # 提取昵称（去除数字和特殊字符）
                nickname = combined_text.strip()
                # 移除常见的干扰字符
                nickname = re.sub(r'[0-9\s]+', '', nickname)
                if nickname:
                    result['nickname'] = nickname
            
            # 处理用户ID
            elif 'ID' in class_name and result['user_id'] is None:
                match = re.search(r'(\d{6,})', combined_text)
                if match:
                    result['user_id'] = match.group(1)
            
            # 处理数字字段
            else:
                numbers = re.findall(r'(\d+\.?\d*)', combined_text)
                if numbers:
                    try:
                        value = float(numbers[0])
                        
                        if '余额' in class_name and result['balance'] is None:
                            result['balance'] = value
                        elif '积分' in class_name and result['points'] is None:
                            result['points'] = int(value)
                        elif '抵扣' in class_name and result['vouchers'] is None:
                            result['vouchers'] = value
                        elif '优惠' in class_name and result['coupons'] is None:
                            result['coupons'] = int(value)
                    except ValueError:
                        pass
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*70}")
        print(f"性能对比总结:")
        print(f"  方案A（全屏 OCR）: {full_ocr_time:.3f}秒")
        print(f"  方案B（区域 OCR）: {region_ocr_time:.3f}秒")
        print(f"  加速比: {speedup:.1f}x")
        print(f"  总耗时: {total_time:.3f}秒 (包含 YOLO {yolo_time:.3f}秒)")
        print(f"{'='*70}")
        
        # 显示结果
        print(f"\n获取结果:")
        print(f"  - 昵称: {result.get('nickname')}")
        print(f"  - 用户ID: {result.get('user_id')}")
        print(f"  - 余额: {result.get('balance')}")
        print(f"  - 积分: {result.get('points')}")
        print(f"  - 抵扣券: {result.get('vouchers')}")
        print(f"  - 优惠券: {result.get('coupons')}")
        
        # 统计成功率
        success_count = sum(1 for v in result.values() if v is not None)
        print(f"\n✅ 成功获取: {success_count}/6 个字段")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(test_optimized_profile_reader())
