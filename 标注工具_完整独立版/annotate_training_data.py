"""在标注工具的训练数据上标注学习器推荐的位置"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from button_position_learner import ButtonPositionLearner
from ocr_region_learner import OCRRegionLearner
from pathlib import Path
import json
import cv2
import random

# 创建输出目录
output_dir = Path("learning_visualization/training_data_annotated")
output_dir.mkdir(parents=True, exist_ok=True)

# 页面类型和对应的标注元素映射
PAGE_TYPE_MAPPING = {
    '首页': {
        'buttons': ['home_checkin_button'],
        'regions': []
    },
    '签到页': {
        'buttons': [],
        'regions': ['checkin_total_times', 'checkin_remaining_times']
    },
    '个人页_已登录': {
        'buttons': [],
        'regions': ['profile_balance', 'profile_points', 'profile_vouchers', 'profile_coupons']
    },
    '钱包页': {
        'buttons': ['wallet_balance_button'],
        'regions': []
    },
    '转账页': {
        'buttons': [],
        'regions': []
    }
}

def draw_button_position(img, button_name, position, stats):
    """在图片上绘制按钮位置"""
    x, y = position
    
    # 绘制推荐位置（红色圆点）
    cv2.circle(img, (x, y), 12, (0, 0, 255), -1)
    cv2.circle(img, (x, y), 18, (0, 0, 255), 3)
    
    # 绘制标准差范围
    if stats and stats['x_stdev'] > 0:
        x_std = max(int(stats['x_stdev'] * 3), 25)
        y_std = max(int(stats['y_stdev'] * 3), 25)
        
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (x - x_std, y - y_std), 
                     (x + x_std, y + y_std), 
                     (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        cv2.rectangle(img, 
                     (x - x_std, y - y_std), 
                     (x + x_std, y + y_std), 
                     (0, 255, 0), 3)
    
    # 添加文字标签
    label = button_name.replace('_', ' ')
    label_pos = f"({x}, {y})"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w1, h1), _ = cv2.getTextSize(label, font, 0.8, 2)
    (w2, h2), _ = cv2.getTextSize(label_pos, font, 0.6, 2)
    
    bg_x = x + 25
    bg_y = y - 55
    bg_w = max(w1, w2) + 25
    bg_h = 60
    
    # 确保标签不超出图片边界
    img_h, img_w = img.shape[:2]
    if bg_x + bg_w > img_w:
        bg_x = x - bg_w - 25
    if bg_y < 0:
        bg_y = y + 25
    
    cv2.rectangle(img, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)
    cv2.rectangle(img, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 255), 3)
    
    cv2.putText(img, label, (bg_x + 12, bg_y + 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, label_pos, (bg_x + 12, bg_y + 50), font, 0.6, (200, 200, 200), 2)

def draw_ocr_region(img, region_name, region, stats):
    """在图片上绘制OCR区域"""
    x, y, w, h = region
    
    # 绘制推荐区域
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 100, 0), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 100, 0), 4)
    
    # 添加文字标签
    label = region_name.replace('_', ' ')
    label_size = f"{w}x{h}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w1, h1), _ = cv2.getTextSize(label, font, 0.7, 2)
    (w2, h2), _ = cv2.getTextSize(label_size, font, 0.6, 2)
    
    bg_w = max(w1, w2) + 25
    bg_h = 55
    bg_x = x
    bg_y = y - bg_h - 8
    
    # 如果标签会超出图片顶部，放到区域下方
    if bg_y < 0:
        bg_y = y + h + 8
    
    cv2.rectangle(img, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)
    cv2.rectangle(img, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (255, 100, 0), 3)
    
    cv2.putText(img, label, (bg_x + 12, bg_y + 28), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img, label_size, (bg_x + 12, bg_y + 48), font, 0.6, (200, 200, 200), 2)

def annotate_image(img_path, button_learner, ocr_learner, page_type):
    """标注单张图片"""
    # 使用numpy读取中文路径的图片
    import numpy as np
    try:
        img_data = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"    ⚠️ 图片解码失败")
            return None
    except Exception as e:
        print(f"    ⚠️ 读取图片失败: {e}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # 获取该页面类型的标注配置
    annotations = PAGE_TYPE_MAPPING.get(page_type, {'buttons': [], 'regions': []})
    
    annotated = False
    
    # 标注按钮
    for button_name in annotations['buttons']:
        best_pos = button_learner.get_best_position(button_name, min_samples=5)
        if best_pos and 0 <= best_pos[0] < img_width and 0 <= best_pos[1] < img_height:
            stats = button_learner.get_statistics(button_name)
            draw_button_position(img, button_name, best_pos, stats)
            annotated = True
    
    # 标注OCR区域
    for region_name in annotations['regions']:
        best_region = ocr_learner.get_best_region(region_name, min_samples=5)
        if best_region:
            x, y, w, h = best_region
            if 0 <= x < img_width and 0 <= y < img_height and x + w <= img_width and y + h <= img_height:
                stats = ocr_learner.get_statistics(region_name)
                draw_ocr_region(img, region_name, best_region, stats)
                annotated = True
    
    return img if annotated else None

def main():
    print("\n在标注工具训练数据上标注学习器推荐的位置")
    print("=" * 70)
    
    # 初始化学习器
    button_learner = ButtonPositionLearner()
    ocr_learner = OCRRegionLearner()
    
    # 训练数据根目录
    training_data_dir = Path("标注工具_完整独立版/training_data")
    
    if not training_data_dir.exists():
        print("⚠️ 找不到训练数据目录")
        return
    
    # 遍历所有页面类型
    total_annotated = 0
    
    for page_type in PAGE_TYPE_MAPPING.keys():
        page_dir = training_data_dir / page_type
        
        if not page_dir.exists():
            print(f"\n⚠️ 跳过: {page_type} (目录不存在)")
            continue
        
        # 获取该页面类型的所有图片
        images = list(page_dir.glob("*.png")) + list(page_dir.glob("*.jpg"))
        
        if not images:
            print(f"\n⚠️ 跳过: {page_type} (没有图片)")
            continue
        
        print(f"\n{'='*70}")
        print(f"处理页面类型: {page_type} ({len(images)} 张图片)")
        print(f"{'='*70}")
        
        # 随机选择最多5张图片进行标注
        sample_images = random.sample(images, min(5, len(images)))
        
        for i, img_path in enumerate(sample_images, 1):
            print(f"\n[{i}/{len(sample_images)}] {img_path.name}")
            
            annotated_img = annotate_image(img_path, button_learner, ocr_learner, page_type)
            
            if annotated_img is not None:
                # 创建页面类型子目录
                page_output_dir = output_dir / page_type
                page_output_dir.mkdir(exist_ok=True)
                
                # 保存标注后的图片 - 使用numpy保存以支持中文路径
                output_path = page_output_dir / f"annotated_{img_path.name}"
                import numpy as np
                is_success, im_buf_arr = cv2.imencode(".png", annotated_img)
                if is_success:
                    im_buf_arr.tofile(str(output_path))
                    print(f"  ✅ 已标注并保存")
                    total_annotated += 1
                else:
                    print(f"  ⚠️ 保存失败")
            else:
                print(f"  ⚠️ 无相关标注元素")
    
    print("\n" + "=" * 70)
    print(f"✅ 完成！共标注 {total_annotated} 张图片")
    print(f"📁 标注图片保存在: {output_dir.absolute()}")
    
    # 打开文件夹
    if total_annotated > 0:
        print("\n正在打开文件夹...")
        import subprocess
        subprocess.run(['explorer', str(output_dir.absolute())])
    
    print("\n图例说明：")
    print("  🔴 红色圆点 + 绿色范围 = 按钮推荐位置及标准差范围")
    print("  🔵 蓝色矩形 = OCR区域推荐位置")
    print("\n标注的页面类型：")
    for page_type, config in PAGE_TYPE_MAPPING.items():
        if config['buttons'] or config['regions']:
            print(f"  - {page_type}: ", end="")
            elements = []
            if config['buttons']:
                elements.extend(config['buttons'])
            if config['regions']:
                elements.extend(config['regions'])
            print(", ".join(elements))
    print("=" * 70)

if __name__ == "__main__":
    main()
