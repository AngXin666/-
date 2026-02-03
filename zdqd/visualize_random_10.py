"""
随机选择10张图片并可视化YOLO检测结果
"""
import sys
from pathlib import Path
import random
import os

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


def visualize_detection(image_path, model, output_dir):
    """可视化单张图片的检测结果"""
    try:
        # 加载图片
        image = Image.open(image_path)
        
        # YOLO检测
        results = model.predict(image, conf=0.25, verbose=False)
        
        # 创建绘图对象
        draw = ImageDraw.Draw(image)
        
        # 尝试加载中文字体
        try:
            font = ImageFont.truetype("msyh.ttc", 20)  # 微软雅黑
            font_small = ImageFont.truetype("msyh.ttc", 16)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 定义颜色
        colors = {
            '首页': '#FF0000',      # 红色
            '我的': '#00FF00',      # 绿色
            '抵扣券数字': '#0000FF',  # 蓝色
            '优惠券数字': '#FFFF00',  # 黄色
            '余额数字': '#FF00FF',    # 品红
            '积分数字': '#00FFFF',    # 青色
            '昵称文字': '#FFA500',    # 橙色
            'ID文字': '#800080'      # 紫色
        }
        
        detection_count = 0
        class_counts = {}
        
        # 绘制检测框
        for r in results:
            boxes = r.boxes
            detection_count = len(boxes)
            
            for box in boxes:
                cls = int(box.cls[0])
                class_name = r.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 统计类别
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # 获取颜色
                color = colors.get(class_name, '#FFFFFF')
                
                # 绘制边框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # 绘制标签背景
                label = f"{class_name} {conf:.2f}"
                bbox = draw.textbbox((x1, y1 - 25), label, font=font_small)
                draw.rectangle(bbox, fill=color)
                
                # 绘制标签文字
                draw.text((x1, y1 - 25), label, fill='black', font=font_small)
        
        # 在图片顶部添加统计信息
        stats_text = f"检测区域: {detection_count} | "
        stats_text += " | ".join([f"{name}: {count}" for name, count in sorted(class_counts.items())])
        
        # 绘制统计信息背景
        stats_bbox = draw.textbbox((10, 10), stats_text, font=font)
        draw.rectangle(stats_bbox, fill='white')
        draw.text((10, 10), stats_text, fill='black', font=font)
        
        # 保存结果
        output_path = Path(output_dir) / f"visualized_{Path(image_path).name}"
        image.save(output_path)
        
        print(f"  ✓ {Path(image_path).name}")
        print(f"    检测区域: {detection_count}")
        print(f"    类别分布: {class_counts}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ {Path(image_path).name} - 错误: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("随机选择10张图片并可视化YOLO检测结果")
    print("=" * 70)
    
    # 查找测试图片
    print("\n[1] 查找测试图片...")
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
    
    # 随机选择10张
    if len(test_images) > 10:
        selected_images = random.sample(test_images, 10)
    else:
        selected_images = test_images
    
    print(f"✓ 随机选择 {len(selected_images)} 张图片")
    
    # 创建输出目录
    output_dir = "debug_annotations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录: {output_dir}")
    
    # 加载模型
    model_path = "runs/detect/runs/detect/profile_detailed_detector/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"\n❌ 模型不存在: {model_path}")
        return
    
    print(f"\n[2] 加载YOLO详细标注检测模型...")
    model = YOLO(model_path)
    print(f"✓ 模型已加载")
    
    # 开始可视化
    print(f"\n[3] 开始可视化...")
    print(f"{'='*70}")
    
    success_count = 0
    for i, image_path in enumerate(selected_images, 1):
        print(f"\n[{i}/{len(selected_images)}]")
        if visualize_detection(image_path, model, output_dir):
            success_count += 1
    
    # 统计结果
    print(f"\n{'='*70}")
    print(f"可视化完成！")
    print(f"{'='*70}")
    print(f"\n成功: {success_count}/{len(selected_images)}")
    print(f"输出目录: {output_dir}")
    print(f"\n请查看 {output_dir} 目录中的可视化结果")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
