"""
测试使用YOLO详细标注模型获取昵称
Test Nickname Recognition using YOLO Detailed Annotation Model
"""
import sys
from pathlib import Path
import os
import asyncio
import re

sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("❌ PIL未安装")
    sys.exit(1)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("❌ YOLO未安装")
    sys.exit(1)

# 导入项目的OCR系统
try:
    from src.ocr_image_processor import enhance_for_ocr
    from src.ocr_thread_pool import get_ocr_pool
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️  OCR系统未安装")


async def get_nickname_from_image(image_path, model, ocr_pool):
    """从图片中获取昵称"""
    try:
        # 加载图片
        image = Image.open(image_path)
        
        # YOLO检测
        results = model.predict(image, conf=0.25, verbose=False)
        
        nickname = None
        nickname_conf = 0.0
        
        # 查找昵称区域
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                cls = int(box.cls[0])
                class_name = r.names[cls]
                conf = float(box.conf[0])
                
                # 只处理昵称文字区域
                if class_name == '昵称文字':
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 裁剪昵称区域
                    region = image.crop((int(x1), int(y1), int(x2), int(y2)))
                    
                    # OCR识别
                    enhanced_image = enhance_for_ocr(region)
                    ocr_result = await ocr_pool.recognize(enhanced_image, timeout=5.0)
                    
                    if ocr_result and ocr_result.texts:
                        # 提取第一行作为昵称
                        text = '\n'.join(ocr_result.texts)
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        if lines:
                            nickname = lines[0]
                            nickname_conf = conf
                            break
        
        return {
            'success': nickname is not None,
            'nickname': nickname,
            'confidence': nickname_conf
        }
        
    except Exception as e:
        return {
            'success': False,
            'nickname': None,
            'confidence': 0.0,
            'error': str(e)
        }


async def test_nickname_recognition():
    """测试昵称识别"""
    
    print("=" * 70)
    print("测试使用YOLO详细标注模型获取昵称")
    print("=" * 70)
    
    # 初始化OCR系统
    print("\n[1] 初始化OCR系统...")
    ocr_pool = None
    if HAS_OCR:
        try:
            ocr_pool = get_ocr_pool()
            print("✓ OCR线程池已初始化")
        except Exception as e:
            print(f"⚠️  OCR初始化失败: {e}")
            return
    else:
        print("⚠️  OCR系统未找到")
        return
    
    # 查找测试图片
    print("\n[2] 查找测试图片...")
    test_images = []
    test_dir = 'training_data/新已登陆页'
    
    if os.path.exists(test_dir):
        images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
        test_images.extend(images)
    
    if not test_images:
        print("❌ 未找到测试图片")
        return
    
    print(f"✓ 找到 {len(test_images)} 张测试图片")
    
    # 加载模型
    model_path = "runs/detect/runs/detect/profile_detailed_detector/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"\n❌ 模型不存在: {model_path}")
        return
    
    print(f"\n[3] 加载YOLO详细标注检测模型...")
    model = YOLO(model_path)
    print(f"✓ 模型已加载")
    
    # 开始测试
    print(f"\n[4] 开始测试昵称识别...")
    print(f"{'='*70}")
    
    results = []
    success_count = 0
    fail_count = 0
    
    for i, image_path in enumerate(test_images, 1):
        result = await get_nickname_from_image(image_path, model, ocr_pool)
        results.append(result)
        
        if result['success']:
            success_count += 1
            status = "✓"
        else:
            fail_count += 1
            status = "✗"
        
        # 显示前10个和每10个
        if i <= 10 or i % 10 == 0:
            print(f"  [{i}/{len(test_images)}] {status} {os.path.basename(image_path)}")
            print(f"    昵称: {result['nickname']}")
            print(f"    置信度: {result['confidence']:.3f}")
            if 'error' in result:
                print(f"    错误: {result['error']}")
    
    # 统计结果
    print(f"\n{'='*70}")
    print(f"测试完成！")
    print(f"{'='*70}")
    
    print(f"\n【识别准确率】")
    print(f"  成功: {success_count}/{len(test_images)} ({success_count/len(test_images)*100:.1f}%)")
    print(f"  失败: {fail_count}/{len(test_images)} ({fail_count/len(test_images)*100:.1f}%)")
    
    # 统计昵称分布
    print(f"\n【昵称分布】")
    nickname_counts = {}
    for result in results:
        if result['success']:
            nickname = result['nickname']
            nickname_counts[nickname] = nickname_counts.get(nickname, 0) + 1
    
    for nickname, count in sorted(nickname_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {nickname}: {count}次")
    
    # 置信度统计
    confidences = [r['confidence'] for r in results if r['success']]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        print(f"\n【置信度统计】")
        print(f"  平均: {avg_conf:.3f}")
        print(f"  最低: {min_conf:.3f}")
        print(f"  最高: {max_conf:.3f}")
    
    # 失败案例
    if fail_count > 0:
        print(f"\n【失败案例】")
        for i, result in enumerate(results, 1):
            if not result['success']:
                print(f"  {i}. {os.path.basename(test_images[i-1])}")
                if 'error' in result:
                    print(f"     错误: {result['error']}")
    
    print(f"\n{'='*70}")
    
    # OCR缓存统计
    print(f"\n【OCR缓存统计】")
    ocr_stats = ocr_pool.get_stats()
    print(f"  总请求数: {ocr_stats['total_requests']}")
    print(f"  缓存命中: {ocr_stats['cache_hits']}")
    print(f"  缓存未命中: {ocr_stats['cache_misses']}")
    print(f"  缓存命中率: {ocr_stats['cache_hit_rate']:.1%}")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    asyncio.run(test_nickname_recognition())
