"""在实际截图上标注学习器推荐的位置"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from button_position_learner import ButtonPositionLearner
from ocr_region_learner import OCRRegionLearner
from pathlib import Path
import json
import cv2
import numpy as np
from datetime import datetime

# 创建输出目录
output_dir = Path("learning_visualization")
output_dir.mkdir(exist_ok=True)

def find_latest_screenshots():
    """查找最新的截图"""
    screenshot_dirs = [
        Path("checkin_screenshots"),
        Path("screenshots/checkin"),
        Path("screenshots/exception")
    ]
    
    screenshots = []
    for dir_path in screenshot_dirs:
        if dir_path.exists():
            for img_file in dir_path.glob("**/*.png"):
                screenshots.append(img_file)
    
    # 按修改时间排序，取最新的
    screenshots.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return screenshots[:10] if screenshots else []

def draw_button_position(img, button_name, position, stats):
    """在图片上绘制按钮位置"""
    x, y = position
    
    # 绘制推荐位置（红色圆点）
    cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
    cv2.circle(img, (x, y), 12, (0, 0, 255), 2)
    
    # 绘制标准差范围（半透明矩形）
    if stats:
        x_std = int(stats['x_stdev'] * 2)  # 2倍标准差
        y_std = int(stats['y_stdev'] * 2)
        
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (x - x_std, y - y_std), 
                     (x + x_std, y + y_std), 
                     (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        # 绘制边框
        cv2.rectangle(img, 
                     (x - x_std, y - y_std), 
                     (x + x_std, y + y_std), 
                     (0, 255, 0), 2)
    
    # 添加文字标签
    label = f"{button_name}"
    label_pos = f"({x}, {y})"
    
    # 文字背景
    (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (w2, h2), _ = cv2.getTextSize(label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    cv2.rectangle(img, (x + 15, y - 35), (x + 15 + max(w1, w2) + 10, y + 5), (0, 0, 0), -1)
    
    # 绘制文字
    cv2.putText(img, label, (x + 20, y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, label_pos, (x + 20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_ocr_region(img, region_name, region, stats):
    """在图片上绘制OCR区域"""
    x, y, w, h = region
    
    # 绘制推荐区域（蓝色矩形）
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    
    # 绘制边框
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 添加文字标签
    label = f"{region_name}"
    label_size = f"{w}x{h}"
    
    # 文字背景
    (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    (w2, h2), _ = cv2.getTextSize(label_size, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    
    cv2.rectangle(img, (x, y - 30), (x + max(w1, w2) + 10, y), (0, 0, 0), -1)
    
    # 绘制文字
    cv2.putText(img, label, (x + 5, y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, label_size, (x + 5, y - 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def annotate_screenshot(img_path, button_learner, ocr_learner):
    """在截图上标注学习到的位置"""
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # 获取图片尺寸
    img_height, img_width = img.shape[:2]
    print(f"  图片尺寸: {img_width}x{img_height}")
    
    annotated = False
    
    # 标注按钮位置
    global_file = Path("runtime_data/button_positions/global.json")
    if global_file.exists():
        with open(global_file, 'r', encoding='utf-8') as f:
            button_data = json.load(f)
        
        print(f"  按钮数据: {list(button_data.keys())}")
        
        for button_name in button_data.keys():
            best_pos = button_learner.get_best_position(button_name, min_samples=5)
            if best_pos:
                # 检查坐标是否在图片范围内
                if 0 <= best_pos[0] < img_width and 0 <= best_pos[1] < img_height:
                    stats = button_learner.get_statistics(button_name)
                    draw_button_position(img, button_name, best_pos, stats)
                    print(f"    ✓ 标注按钮: {button_name} at ({best_pos[0]}, {best_pos[1]})")
                    annotated = True
                else:
                    print(f"    ✗ 跳过按钮: {button_name} - 坐标超出范围 ({best_pos[0]}, {best_pos[1]})")
    
    # 标注OCR区域
    global_file = Path("runtime_data/ocr_regions/global.json")
    if global_file.exists():
        with open(global_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        print(f"  OCR区域数据: {list(ocr_data.keys())}")
        
        for region_name in ocr_data.keys():
            best_region = ocr_learner.get_best_region(region_name, min_samples=5)
            if best_region:
                x, y, w, h = best_region
                # 检查区域是否在图片范围内
                if 0 <= x < img_width and 0 <= y < img_height and x + w <= img_width and y + h <= img_height:
                    stats = ocr_learner.get_statistics(region_name)
                    draw_ocr_region(img, region_name, best_region, stats)
                    print(f"    ✓ 标注区域: {region_name} at ({x}, {y}, {w}, {h})")
                    annotated = True
                else:
                    print(f"    ✗ 跳过区域: {region_name} - 坐标超出范围 ({x}, {y}, {w}, {h})")
    
    return img if annotated else None

def main():
    print("\n在实际截图上标注学习器推荐的位置")
    print("=" * 60)
    
    # 初始化学习器
    button_learner = ButtonPositionLearner()
    ocr_learner = OCRRegionLearner()
    
    # 查找截图
    print("\n正在查找截图...")
    screenshots = find_latest_screenshots()
    
    if not screenshots:
        print("⚠️ 没有找到截图文件")
        print("请确保以下目录存在截图：")
        print("  - checkin_screenshots/")
        print("  - screenshots/checkin/")
        print("  - screenshots/exception/")
        return
    
    print(f"找到 {len(screenshots)} 张截图，正在标注...")
    
    # 标注截图
    annotated_count = 0
    for i, img_path in enumerate(screenshots, 1):
        print(f"\n[{i}/{len(screenshots)}] 处理: {img_path.name}")
        
        annotated_img = annotate_screenshot(img_path, button_learner, ocr_learner)
        
        if annotated_img is not None:
            # 保存标注后的图片
            output_path = output_dir / f"annotated_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_img)
            print(f"  ✅ 已保存: {output_path.name}")
            annotated_count += 1
        else:
            print(f"  ⚠️ 跳过（无法标注）")
    
    print("\n" + "=" * 60)
    print(f"✅ 完成！共标注 {annotated_count} 张图片")
    print(f"📁 标注图片保存在: {output_dir.absolute()}")
    
    # 打开文件夹
    if annotated_count > 0:
        print("\n正在打开文件夹...")
        import subprocess
        subprocess.run(['explorer', str(output_dir.absolute())])
    
    print("\n图例说明：")
    print("  🔴 红色圆点 = 按钮推荐位置")
    print("  🟢 绿色矩形 = 按钮位置标准差范围（2倍标准差）")
    print("  🔵 蓝色矩形 = OCR区域推荐位置")
    print("=" * 60)

if __name__ == "__main__":
    main()
