"""
测试个人页详细标注YOLO模型
Test Profile Detailed Annotation YOLO Model
"""

import sys
from pathlib import Path
import os
import time
from io import BytesIO
import asyncio

sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from PIL import Image, ImageDraw, ImageFont
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
        return f"[OCR错误: {e}]"


async def test_profile_detailed():
    """测试个人页详细标注检测"""
    
    print("=" * 70)
    print("测试个人页详细标注YOLO模型")
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
        'yolo_dataset/profile_detailed/images/val',
        'yolo_dataset/profile_detailed/images/train',
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
    model_path = "runs/detect/runs/detect/profile_detailed_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"\n❌ 模型不存在: {model_path}")
        return
    
    print(f"\n[2] 加载YOLO详细标注检测模型...")
    model = YOLO(model_path)
    print(f"✓ 模型已加载")
    
    # YOLO检测
    print(f"\n[3] YOLO检测区域...")
    yolo_start = time.time()
    
    results = model.predict(image, conf=0.1, verbose=False)
    
    yolo_time = time.time() - yolo_start
    print(f"✓ YOLO检测完成，耗时 {yolo_time:.3f}秒")
    
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
    detections = []
    
    for r in results:
        boxes = r.boxes
        print(f"\n检测到 {len(boxes)} 个区域:")
        
        for box in boxes:
            cls = int(box.cls[0])
            class_name = r.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            detection = {
                'class': class_name,
                'conf': conf,
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            }
            detections.append(detection)
            
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
            
            # 对文字区域进行OCR识别
            if '文字' in class_name or '数字' in class_name:
                ocr_start = time.time()
                text = await ocr_region_async(region, ocr_pool, class_name)
                ocr_time = time.time() - ocr_start
                ocr_total_time += ocr_time
                
                print(f"    OCR耗时: {ocr_time:.3f}秒")
                print(f"    识别结果: {text[:200] if text else '(空)'}")
                
                detection['ocr_text'] = text
                
                # 根据区域类型保存数据
                if class_name == '昵称文字':
                    profile_data['nickname'] = text.strip()
                elif class_name == 'ID文字':
                    # 提取纯数字
                    import re
                    match = re.search(r'(\d+)', text)
                    if match:
                        profile_data['user_id'] = match.group(1)
                    else:
                        profile_data['user_id'] = text.strip()
                elif class_name == '余额数字':
                    try:
                        profile_data['balance'] = float(text.strip())
                    except ValueError:
                        profile_data['balance'] = text.strip()
                elif class_name == '积分数字':
                    try:
                        profile_data['points'] = int(float(text.strip()))
                    except ValueError:
                        profile_data['points'] = text.strip()
                elif class_name == '抵扣券数字':
                    try:
                        profile_data['vouchers'] = float(text.strip())
                    except ValueError:
                        profile_data['vouchers'] = text.strip()
                elif class_name == '优惠券数字':
                    try:
                        profile_data['coupons'] = int(float(text.strip()))
                    except ValueError:
                        profile_data['coupons'] = text.strip()
            else:
                # 按钮区域不需要OCR
                if class_name == '首页':
                    profile_data['homepage_button'] = (int(x1), int(y1), int(x2), int(y2))
                elif class_name == '我的':
                    profile_data['my_button'] = (int(x1), int(y1), int(x2), int(y2))
    
    # 性能统计
    total_time = yolo_time + ocr_total_time
    
    print(f"\n{'='*70}")
    print(f"性能统计:")
    print(f"  YOLO检测: {yolo_time:.3f}秒")
    print(f"  OCR识别: {ocr_total_time:.3f}秒")
    print(f"  总耗时: {total_time:.3f}秒")
    print(f"{'='*70}")
    
    print(f"\n[识别结果]")
    print(f"  昵称: {profile_data['nickname']}")
    print(f"  用户ID: {profile_data['user_id']}")
    print(f"  余额: {profile_data['balance']}")
    print(f"  积分: {profile_data['points']}")
    print(f"  抵扣劵: {profile_data['vouchers']}")
    print(f"  优惠劵: {profile_data['coupons']}")
    print(f"  首页按钮: {profile_data['homepage_button']}")
    print(f"  我的按钮: {profile_data['my_button']}")
    
    # 创建可视化图片
    print(f"\n[4] 生成可视化图片...")
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # 定义颜色
    colors = {
        '首页': '#FF0000',
        '我的': '#00FF00',
        '抵扣券数字': '#0000FF',
        '优惠券数字': '#FFFF00',
        '余额数字': '#FF00FF',
        '积分数字': '#00FFFF',
        '昵称文字': '#FFA500',
        'ID文字': '#800080'
    }
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = colors.get(det['class'], '#FFFFFF')
        
        # 绘制边框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # 绘制标签
        label = f"{det['class']} {det['conf']:.2f}"
        if 'ocr_text' in det and det['ocr_text']:
            label += f"\n{det['ocr_text'][:20]}"
        
        # 绘制标签背景
        try:
            font = ImageFont.truetype("msyh.ttc", 12)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((x1, y1-15), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1-15), label, fill='#000000', font=font)
    
    # 保存可视化图片
    vis_path = "debug_regions/visualization.png"
    vis_image.save(vis_path)
    print(f"✓ 可视化图片已保存: {vis_path}")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    asyncio.run(test_profile_detailed())
