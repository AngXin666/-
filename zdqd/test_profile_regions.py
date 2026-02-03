"""
测试个人页区域检测模型
Test Profile Regions Detection Model
"""

import sys
from pathlib import Path
import os
import time
from io import BytesIO
import asyncio

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
        print(f"  ⚠️  OCR失败: {e}")
        import traceback
        traceback.print_exc()
        return f"[OCR错误: {e}]"


async def test_profile_regions():
    """测试个人页区域检测"""
    
    print("=" * 70)
    print("测试个人页区域检测模型")
    print("=" * 70)
    
    # 初始化OCR系统（使用项目的OCR线程池）
    print("\n[0] 初始化OCR系统...")
    ocr_pool = None
    if HAS_OCR:
        try:
            ocr_pool = get_ocr_pool()
            print("✓ OCR线程池已初始化")
        except Exception as e:
            print(f"⚠️  OCR初始化失败: {e}")
    else:
        print("⚠️  OCR系统未找到，将跳过OCR识别")
    
    # 查找测试图片
    print("\n[1] 查找测试图片...")
    test_image_path = None
    test_dirs = [
        'training_data/新已登陆页',
        'yolo_dataset/profile_regions/images/val',
        '原始标注图/个人页_已登录_余额积分/images',
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
    
    # 加载图片
    image = Image.open(test_image_path)
    print(f"  图片尺寸: {image.size}")
    
    # 检查模型
    model_path = "runs/detect/runs/detect/profile_regions_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"\n❌ 模型不存在: {model_path}")
        return
    
    print(f"\n[2] 加载YOLO区域检测模型...")
    model = YOLO(model_path)
    print(f"✓ 模型已加载")
    
    # YOLO检测
    print(f"\n[3] YOLO检测区域...")
    yolo_start = time.time()
    
    results = model.predict(image, conf=0.5, verbose=False)
    
    yolo_time = time.time() - yolo_start
    print(f"✓ YOLO检测完成，耗时 {yolo_time:.3f}秒")
    
    # 解析结果
    profile_data = {
        'nickname': None,
        'user_id': None,
        'balance': None,
        'points': None,
        'vouchers': None,
        'coupons': None
    }
    
    ocr_total_time = 0
    
    for r in results:
        boxes = r.boxes
        print(f"\n检测到 {len(boxes)} 个区域:")
        
        for box in boxes:
            cls = int(box.cls[0])
            class_name = r.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            print(f"\n  [{class_name}]")
            print(f"    置信度: {conf:.2%}")
            print(f"    位置: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")
            
            # 裁剪区域
            region = image.crop((int(x1), int(y1), int(x2), int(y2)))
            
            # 保存裁剪的区域（用于调试）
            debug_dir = Path("debug_regions")
            debug_dir.mkdir(exist_ok=True)
            region_filename = f"{class_name.replace('/', '_')}_{int(conf*100)}.png"
            region.save(debug_dir / region_filename)
            print(f"    已保存区域: {debug_dir / region_filename}")
            
            # OCR识别（使用异步版本）
            ocr_start = time.time()
            text = await ocr_region_async(region, ocr_pool, class_name)
            ocr_time = time.time() - ocr_start
            ocr_total_time += ocr_time
            
            print(f"    OCR耗时: {ocr_time:.3f}秒")
            print(f"    识别结果: {text[:200] if text else '(空)'}")
            
            # 根据区域类型解析内容
            if '确认按钮' in class_name:
                # 这个区域包含昵称和ID
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # 提取昵称（第一行）
                if len(lines) >= 1:
                    profile_data['nickname'] = lines[0]
                
                # 提取用户ID（查找包含"ID:"的行，或者纯数字行）
                import re
                for line in lines:
                    # 查找ID:后面的数字
                    if 'ID' in line or 'id' in line:
                        match = re.search(r'(\d{6,})', line)
                        if match:
                            profile_data['user_id'] = match.group(1)
                            break
                    # 或者查找纯数字行（6位以上）
                    elif re.match(r'^\d{6,}$', line):
                        profile_data['user_id'] = line
                        break
                
                print(f"    → 昵称: {profile_data['nickname']}")
                print(f"    → 用户ID: {profile_data['user_id']}")
            
            elif '数据区域' in class_name:
                # 这个区域包含余额、积分、抵扣劵、优惠劵
                # OCR结果格式：数字和标签混在一起
                # 需要根据标签匹配对应的数字
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # 构建文本用于查找
                full_text = '\n'.join(lines)
                
                # 提取数字（按顺序）
                import re
                numbers = re.findall(r'(\d+\.?\d*)', full_text)
                
                # 根据标签位置匹配数字
                # 通常顺序是：余额、积分、抵扣券、青元宝、优惠券
                if '余额' in full_text and len(numbers) >= 1:
                    try:
                        profile_data['balance'] = float(numbers[0])
                    except ValueError:
                        pass
                
                if '积分' in full_text and len(numbers) >= 2:
                    try:
                        profile_data['points'] = int(float(numbers[1]))
                    except ValueError:
                        pass
                
                if '抵扣' in full_text and len(numbers) >= 3:
                    try:
                        profile_data['vouchers'] = float(numbers[2])
                    except ValueError:
                        pass
                
                if '优惠' in full_text and len(numbers) >= 5:
                    try:
                        profile_data['coupons'] = int(float(numbers[4]))
                    except ValueError:
                        pass
                
                print(f"    → 识别到 {len(numbers)} 个数字")
                print(f"    → 余额: {profile_data['balance']}")
                print(f"    → 积分: {profile_data['points']}")
                print(f"    → 抵扣券: {profile_data['vouchers']}")
                print(f"    → 优惠券: {profile_data['coupons']}")
    
    # 性能统计
    total_time = yolo_time + ocr_total_time
    
    print(f"\n{'='*70}")
    print(f"性能统计:")
    print(f"  YOLO检测: {yolo_time:.3f}秒")
    print(f"  OCR识别: {ocr_total_time:.3f}秒")
    print(f"  总耗时: {total_time:.3f}秒")
    print(f"\n  🚀 相比原方案（3.8秒），快了 {3.8/total_time:.1f}x")
    print(f"{'='*70}")
    
    print(f"\n[识别结果]")
    print(f"  昵称: {profile_data['nickname']}")
    print(f"  用户ID: {profile_data['user_id']}")
    print(f"  余额: {profile_data['balance']}")
    print(f"  积分: {profile_data['points']}")
    print(f"  抵扣劵: {profile_data['vouchers']}")
    print(f"  优惠劵: {profile_data['coupons']}")


if __name__ == '__main__':
    asyncio.run(test_profile_regions())
