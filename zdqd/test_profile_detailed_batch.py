"""
批量测试个人页详细标注YOLO模型 - 100次识别测试
Batch Test Profile Detailed Annotation YOLO Model - 100 Recognition Tests
"""

import sys
from pathlib import Path
import os
import time
import asyncio
import random
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

# 导入项目的OCR系统和图像预处理
try:
    from src.ocr_image_processor import enhance_for_ocr
    from src.ocr_thread_pool import get_ocr_pool
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️  OCR系统未安装")


async def ocr_region_async(image, ocr_pool, region_name=""):
    """OCR识别区域（异步版本，使用项目的OCR线程池）"""
    if not HAS_OCR or not ocr_pool:
        return "[OCR未初始化]"
    
    try:
        # 使用项目的图像预处理（灰度图 + 对比度增强2倍）
        enhanced_image = enhance_for_ocr(image)
        
        # 使用OCR线程池识别（异步，带超时）
        ocr_result = await ocr_pool.recognize(enhanced_image, timeout=5.0)
        
        if not ocr_result or not ocr_result.texts:
            return ""
        
        # 返回识别的文本
        return '\n'.join(ocr_result.texts)
        
    except Exception as e:
        return f"[OCR错误: {e}]"


async def recognize_single_image(image_path, model, ocr_pool, verbose=False):
    """识别单张图片"""
    try:
        # 加载图片
        image = Image.open(image_path)
        
        # YOLO检测
        yolo_start = time.time()
        results = model.predict(image, conf=0.25, verbose=False)
        yolo_time = time.time() - yolo_start
        
        # 解析结果
        profile_data = {
            'nickname': None,
            'user_id': None,
            'balance': None,
            'points': None,
            'vouchers': None,
            'coupons': None,
            'homepage_button': None,
            'my_button': None
        }
        
        ocr_total_time = 0
        detection_count = 0
        
        for r in results:
            boxes = r.boxes
            detection_count = len(boxes)
            
            for box in boxes:
                cls = int(box.cls[0])
                class_name = r.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 裁剪区域
                region = image.crop((int(x1), int(y1), int(x2), int(y2)))
                
                # 对文字区域进行OCR识别
                if '文字' in class_name or '数字' in class_name:
                    ocr_start = time.time()
                    text = await ocr_region_async(region, ocr_pool, class_name)
                    ocr_time = time.time() - ocr_start
                    ocr_total_time += ocr_time
                    
                    # 根据区域类型保存数据
                    if class_name == '昵称文字':
                        # 提取第一行作为昵称
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        if lines:
                            profile_data['nickname'] = lines[0]
                    elif class_name == 'ID文字':
                        # 提取纯数字
                        match = re.search(r'(\d+)', text)
                        if match:
                            profile_data['user_id'] = match.group(1)
                    elif class_name == '余额数字':
                        # 提取数字
                        match = re.search(r'(\d+\.?\d*)', text)
                        if match:
                            try:
                                profile_data['balance'] = float(match.group(1))
                            except ValueError:
                                pass
                    elif class_name == '积分数字':
                        # 提取数字
                        match = re.search(r'(\d+\.?\d*)', text)
                        if match:
                            try:
                                profile_data['points'] = int(float(match.group(1)))
                            except ValueError:
                                pass
                    elif class_name == '抵扣券数字':
                        # 提取数字
                        match = re.search(r'(\d+\.?\d*)', text)
                        if match:
                            try:
                                profile_data['vouchers'] = float(match.group(1))
                            except ValueError:
                                pass
                    elif class_name == '优惠券数字':
                        # 提取数字
                        match = re.search(r'(\d+\.?\d*)', text)
                        if match:
                            try:
                                profile_data['coupons'] = int(float(match.group(1)))
                            except ValueError:
                                pass
                else:
                    # 按钮区域不需要OCR
                    if class_name == '首页' and profile_data['homepage_button'] is None:
                        profile_data['homepage_button'] = (int(x1), int(y1), int(x2), int(y2))
                    elif class_name == '我的' and profile_data['my_button'] is None:
                        profile_data['my_button'] = (int(x1), int(y1), int(x2), int(y2))
        
        total_time = yolo_time + ocr_total_time
        
        # 判断识别是否成功（至少要有昵称、用户ID、余额、积分）
        success = (
            profile_data['nickname'] is not None and
            profile_data['user_id'] is not None and
            profile_data['balance'] is not None and
            profile_data['points'] is not None
        )
        
        if verbose:
            print(f"  图片: {os.path.basename(image_path)}")
            print(f"    检测区域数: {detection_count}")
            print(f"    昵称: {profile_data['nickname']}")
            print(f"    用户ID: {profile_data['user_id']}")
            print(f"    余额: {profile_data['balance']}")
            print(f"    积分: {profile_data['points']}")
            print(f"    抵扣券: {profile_data['vouchers']}")
            print(f"    优惠券: {profile_data['coupons']}")
            print(f"    首页按钮: {profile_data['homepage_button']}")
            print(f"    我的按钮: {profile_data['my_button']}")
            print(f"    耗时: {total_time:.3f}秒 (YOLO: {yolo_time:.3f}s, OCR: {ocr_total_time:.3f}s)")
            print(f"    状态: {'✓ 成功' if success else '✗ 失败'}")
        
        return {
            'success': success,
            'yolo_time': yolo_time,
            'ocr_time': ocr_total_time,
            'total_time': total_time,
            'detection_count': detection_count,
            'profile_data': profile_data
        }
        
    except Exception as e:
        if verbose:
            print(f"  图片: {os.path.basename(image_path)} - 错误: {e}")
        return {
            'success': False,
            'yolo_time': 0,
            'ocr_time': 0,
            'total_time': 0,
            'detection_count': 0,
            'profile_data': {},
            'error': str(e)
        }


async def test_batch():
    """批量测试100次识别"""
    
    print("=" * 70)
    print("批量测试个人页详细标注YOLO模型 - 100次识别")
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
    test_dirs = [
        'training_data/新已登陆页',
        'yolo_dataset/profile_detailed/images/val',
        'yolo_dataset/profile_detailed/images/train',
    ]
    
    for img_dir in test_dirs:
        if os.path.exists(img_dir):
            images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
            test_images.extend(images)
    
    if not test_images:
        print("❌ 未找到测试图片")
        return
    
    print(f"✓ 找到 {len(test_images)} 张测试图片")
    
    # 如果图片少于100张，重复使用
    if len(test_images) < 100:
        print(f"  图片数量不足100张，将重复使用现有图片")
        test_images = test_images * (100 // len(test_images) + 1)
    
    # 随机选择100张
    test_images = random.sample(test_images, 100)
    
    # 加载模型
    model_path = "runs/detect/runs/detect/profile_detailed_detector/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"\n❌ 模型不存在: {model_path}")
        return
    
    print(f"\n[3] 加载YOLO详细标注检测模型...")
    model = YOLO(model_path)
    print(f"✓ 模型已加载")
    
    # 开始批量测试
    print(f"\n[4] 开始批量测试 (100次识别)...")
    print(f"{'='*70}")
    
    start_time = time.time()
    results = []
    
    for i, image_path in enumerate(test_images, 1):
        verbose = (i <= 5 or i % 20 == 0)  # 前5张和每20张显示详细信息
        
        if verbose:
            print(f"\n[测试 {i}/100]")
        
        result = await recognize_single_image(image_path, model, ocr_pool, verbose=verbose)
        results.append(result)
        
        if not verbose:
            # 显示进度
            if i % 10 == 0:
                print(f"  进度: {i}/100 ({i}%)")
    
    total_time = time.time() - start_time
    
    # 统计结果
    print(f"\n{'='*70}")
    print(f"测试完成！")
    print(f"{'='*70}")
    
    success_count = sum(1 for r in results if r['success'])
    fail_count = 100 - success_count
    
    yolo_times = [r['yolo_time'] for r in results if r['yolo_time'] > 0]
    ocr_times = [r['ocr_time'] for r in results if r['ocr_time'] > 0]
    total_times = [r['total_time'] for r in results if r['total_time'] > 0]
    
    avg_yolo = sum(yolo_times) / len(yolo_times) if yolo_times else 0
    avg_ocr = sum(ocr_times) / len(ocr_times) if ocr_times else 0
    avg_total = sum(total_times) / len(total_times) if total_times else 0
    
    min_total = min(total_times) if total_times else 0
    max_total = max(total_times) if total_times else 0
    
    print(f"\n【识别准确率】")
    print(f"  成功: {success_count}/100 ({success_count}%)")
    print(f"  失败: {fail_count}/100 ({fail_count}%)")
    
    print(f"\n【性能统计】")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均单次耗时: {avg_total:.3f}秒")
    print(f"    - YOLO检测: {avg_yolo:.3f}秒")
    print(f"    - OCR识别: {avg_ocr:.3f}秒")
    print(f"  最快: {min_total:.3f}秒")
    print(f"  最慢: {max_total:.3f}秒")
    print(f"  吞吐量: {100/total_time:.2f} 张/秒")
    
    # OCR缓存统计
    print(f"\n【OCR缓存统计】")
    ocr_stats = ocr_pool.get_stats()
    print(f"  总请求数: {ocr_stats['total_requests']}")
    print(f"  缓存命中: {ocr_stats['cache_hits']}")
    print(f"  缓存未命中: {ocr_stats['cache_misses']}")
    print(f"  缓存命中率: {ocr_stats['cache_hit_rate']:.1%}")
    print(f"  当前缓存大小: {ocr_stats['cache_size']}")
    
    # 失败案例分析
    if fail_count > 0:
        print(f"\n【失败案例】")
        fail_reasons = {}
        for i, result in enumerate(results, 1):
            if not result['success']:
                data = result.get('profile_data', {})
                missing = []
                if not data.get('nickname'):
                    missing.append('昵称')
                if not data.get('user_id'):
                    missing.append('用户ID')
                if data.get('balance') is None:
                    missing.append('余额')
                if data.get('points') is None:
                    missing.append('积分')
                
                reason = '、'.join(missing) if missing else '未知'
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                
                if i <= 10 or len([r for r in results[:i] if not r['success']]) <= 5:
                    print(f"  测试 {i}: 缺失字段 - {reason}")
                    print(f"    昵称: {data.get('nickname')}")
                    print(f"    用户ID: {data.get('user_id')}")
                    print(f"    余额: {data.get('balance')}")
                    print(f"    积分: {data.get('points')}")
        
        print(f"\n【失败原因统计】")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}次")
    
    print(f"\n{'='*70}")
    
    # 性能对比
    if avg_total > 0:
        original_time = 3.8  # 原方案耗时
        speedup = original_time / avg_total
        print(f"\n【性能对比】")
        print(f"  原方案平均耗时: {original_time:.3f}秒")
        print(f"  新方案平均耗时: {avg_total:.3f}秒")
        if speedup >= 1.0:
            print(f"  🚀 新方案快了 {speedup:.2f}x")
        else:
            print(f"  ⚠️  新方案慢了 {1/speedup:.2f}x")
    
    print(f"{'='*70}")


if __name__ == '__main__':
    asyncio.run(test_batch())
