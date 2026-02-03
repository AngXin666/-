"""
测试 YOLO 数字识别模型
Test YOLO Numbers Recognition Model

对比：
- 方案A：YOLO位置 + OCR识别（当前方案，3.8秒）
- 方案B：YOLO直接识别数字（目标方案，~0.6秒）
"""

import sys
from pathlib import Path
import os
import asyncio
import time
import re
from io import BytesIO

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


class MockADB:
    """模拟ADB"""
    def __init__(self, test_image_path):
        self.test_image_path = test_image_path
    
    async def screencap(self, device_id: str) -> bytes:
        with open(self.test_image_path, 'rb') as f:
            return f.read()


async def test_yolo_numbers():
    """测试 YOLO 数字识别"""
    
    print("=" * 70)
    print("测试 YOLO 数字识别模型")
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
    
    # 加载图片
    adb = MockADB(test_image_path)
    screenshot_data = await adb.screencap("test")
    image = Image.open(BytesIO(screenshot_data))
    
    # 检查模型是否存在
    model_path = "runs/detect/runs/detect/yolo_runs/profile_numbers_detector2/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"\n❌ 模型不存在: {model_path}")
        print(f"\n请先训练模型：")
        print(f"  1. 创建数据集模板:")
        print(f"     python train_profile_numbers_yolo.py --create-template")
        print(f"  2. 准备训练数据（标注整个数字区域）")
        print(f"  3. 训练模型:")
        print(f"     python train_profile_numbers_yolo.py")
        print(f"\n或者使用现有的 balance 模型测试:")
        print(f"     python test_yolo_models.py")
        return
    
    print(f"\n[2] 加载 YOLO 数字识别模型...")
    model = YOLO(model_path)
    print(f"✓ 模型已加载")
    
    # YOLO 检测
    print(f"\n[3] YOLO 检测数字...")
    yolo_start = time.time()
    
    results = model.predict(image, conf=0.25, verbose=False)
    
    yolo_time = time.time() - yolo_start
    print(f"✓ YOLO 检测完成，耗时 {yolo_time:.3f}秒")
    
    # 解析结果
    result = {
        'nickname': None,
        'user_id': None,
        'balance': None,
        'points': None,
        'vouchers': None,
        'coupons': None
    }
    
    for r in results:
        boxes = r.boxes
        print(f"\n检测到 {len(boxes)} 个目标:")
        
        for box in boxes:
            cls = int(box.cls[0])
            class_name = r.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            print(f"  - {class_name}: 置信度={conf:.2f}, bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
            
            # 裁剪区域
            region = image.crop((int(x1), int(y1), int(x2), int(y2)))
            
            # 简单的数字提取（这里用正则，实际可以用轻量级OCR）
            # 方法1：使用 pytesseract 数字模式（很快，~50ms）
            # 方法2：使用 EasyOCR 数字模式
            # 方法3：训练一个小的数字分类器
            
            # 这里演示用 PIL 转文本（实际需要OCR）
            # 为了演示，我们假设能提取到数字
            
            if '余额' in class_name:
                # 实际应该用 OCR 提取数字
                # text = pytesseract.image_to_string(region, config='--psm 7 digits')
                # result['balance'] = float(text.strip())
                print(f"    → 余额区域已检测（需要轻量级数字提取）")
            
            elif '积分' in class_name:
                print(f"    → 积分区域已检测（需要轻量级数字提取）")
            
            elif '抵扣' in class_name:
                print(f"    → 抵扣券区域已检测（需要轻量级数字提取）")
            
            elif '优惠' in class_name:
                print(f"    → 优惠券区域已检测（需要轻量级数字提取）")
            
            elif '昵称' in class_name:
                print(f"    → 昵称区域已检测（需要轻量级文字提取）")
            
            elif 'ID' in class_name:
                print(f"    → 用户ID区域已检测（需要轻量级数字提取）")
    
    print(f"\n{'='*70}")
    print(f"性能预估:")
    print(f"  - YOLO 检测: {yolo_time:.3f}秒")
    print(f"  - 轻量级数字提取: ~0.05秒 × 6个区域 = ~0.3秒")
    print(f"  - 预计总耗时: ~{yolo_time + 0.3:.1f}秒")
    print(f"\n  🚀 相比当前方案（3.8秒），快了 {3.8/(yolo_time + 0.3):.1f}x")
    print(f"{'='*70}")
    
    print(f"\n💡 优化建议:")
    print(f"  1. 使用 pytesseract 数字模式（--psm 7 digits）")
    print(f"  2. 或使用 EasyOCR 数字模式")
    print(f"  3. 或训练一个轻量级数字分类器（CNN）")
    print(f"  4. 数字提取比完整OCR快10-20倍")


if __name__ == '__main__':
    asyncio.run(test_yolo_numbers())
