"""
分析昵称检测置信度差异的原因
"""
import sys
from pathlib import Path
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


def analyze_nickname_detection(image_path, model, output_dir):
    """分析单张图片的昵称检测"""
    try:
        # 加载图片
        image = Image.open(image_path)
        
        # YOLO检测
        results = model.predict(image, conf=0.25, verbose=False)
        
        # 查找昵称区域
        nickname_detections = []
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                cls = int(box.cls[0])
                class_name = r.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                if class_name == '昵称文字':
                    nickname_detections.append({
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'area': int((x2 - x1) * (y2 - y1))
                    })
        
        if not nickname_detections:
            return None
        
        # 取置信度最高的
        detection = max(nickname_detections, key=lambda x: x['confidence'])
        
        # 可视化
        draw = ImageDraw.Draw(image)
        
        # 尝试加载中文字体
        try:
            font = ImageFont.truetype("msyh.ttc", 24)
        except:
            font = ImageFont.load_default()
        
        # 绘制昵称区域
        x1, y1, x2, y2 = detection['bbox']
        draw.rectangle([x1, y1, x2, y2], outline='#FFA500', width=4)
        
        # 绘制信息
        info_text = f"置信度: {detection['confidence']:.3f}\n"
        info_text += f"尺寸: {detection['width']}x{detection['height']}\n"
        info_text += f"面积: {detection['area']}px²"
        
        # 绘制信息背景
        info_lines = info_text.split('\n')
        y_offset = 10
        for line in info_lines:
            bbox = draw.textbbox((10, y_offset), line, font=font)
            draw.rectangle(bbox, fill='white')
            draw.text((10, y_offset), line, fill='black', font=font)
            y_offset += 30
        
        # 保存结果
        output_path = Path(output_dir) / f"analyzed_{Path(image_path).name}"
        image.save(output_path)
        
        return detection
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return None


def main():
    """主函数"""
    print("=" * 70)
    print("分析昵称检测置信度差异")
    print("=" * 70)
    
    # 查找测试图片
    print("\n[1] 查找测试图片...")
    test_dir = 'training_data/新已登陆页'
    
    if not os.path.exists(test_dir):
        print("❌ 测试目录不存在")
        return
    
    # 找到每个账号的一张原始图片（不是副本）
    test_images = []
    for f in os.listdir(test_dir):
        if f.endswith('.png') and '副本' not in f:
            test_images.append(os.path.join(test_dir, f))
    
    print(f"✓ 找到 {len(test_images)} 张原始图片")
    
    # 创建输出目录
    output_dir = "debug_annotations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model_path = "runs/detect/runs/detect/profile_detailed_detector/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"\n❌ 模型不存在: {model_path}")
        return
    
    print(f"\n[2] 加载YOLO详细标注检测模型...")
    model = YOLO(model_path)
    print(f"✓ 模型已加载")
    
    # 开始分析
    print(f"\n[3] 开始分析...")
    print(f"{'='*70}")
    
    results = []
    
    for image_path in test_images:
        print(f"\n分析: {os.path.basename(image_path)}")
        detection = analyze_nickname_detection(image_path, model, output_dir)
        
        if detection:
            results.append({
                'image': os.path.basename(image_path),
                'confidence': detection['confidence'],
                'width': detection['width'],
                'height': detection['height'],
                'area': detection['area']
            })
            
            print(f"  置信度: {detection['confidence']:.3f}")
            print(f"  尺寸: {detection['width']}x{detection['height']}")
            print(f"  面积: {detection['area']}px²")
    
    # 统计分析
    print(f"\n{'='*70}")
    print(f"统计分析")
    print(f"{'='*70}")
    
    if results:
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'])
        
        print(f"\n【按置信度排序】")
        for r in results:
            print(f"  {r['image']}")
            print(f"    置信度: {r['confidence']:.3f}")
            print(f"    尺寸: {r['width']}x{r['height']}")
            print(f"    面积: {r['area']}px²")
        
        # 分析置信度与尺寸的关系
        print(f"\n【置信度与尺寸关系分析】")
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        avg_width = sum(r['width'] for r in results) / len(results)
        avg_height = sum(r['height'] for r in results) / len(results)
        avg_area = sum(r['area'] for r in results) / len(results)
        
        print(f"  平均置信度: {avg_conf:.3f}")
        print(f"  平均宽度: {avg_width:.1f}px")
        print(f"  平均高度: {avg_height:.1f}px")
        print(f"  平均面积: {avg_area:.1f}px²")
        
        # 找出置信度最低和最高的
        lowest = results[0]
        highest = results[-1]
        
        print(f"\n【置信度最低】")
        print(f"  图片: {lowest['image']}")
        print(f"  置信度: {lowest['confidence']:.3f}")
        print(f"  尺寸: {lowest['width']}x{lowest['height']}")
        print(f"  面积: {lowest['area']}px²")
        
        print(f"\n【置信度最高】")
        print(f"  图片: {highest['image']}")
        print(f"  置信度: {highest['confidence']:.3f}")
        print(f"  尺寸: {highest['width']}x{highest['height']}")
        print(f"  面积: {highest['area']}px²")
        
        # 分析差异
        print(f"\n【差异分析】")
        conf_diff = highest['confidence'] - lowest['confidence']
        width_diff = highest['width'] - lowest['width']
        height_diff = highest['height'] - lowest['height']
        area_diff = highest['area'] - lowest['area']
        
        print(f"  置信度差异: {conf_diff:.3f} ({conf_diff/lowest['confidence']*100:.1f}%)")
        print(f"  宽度差异: {width_diff}px ({width_diff/lowest['width']*100:.1f}%)")
        print(f"  高度差异: {height_diff}px ({height_diff/lowest['height']*100:.1f}%)")
        print(f"  面积差异: {area_diff}px² ({area_diff/lowest['area']*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"可视化结果已保存到: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
