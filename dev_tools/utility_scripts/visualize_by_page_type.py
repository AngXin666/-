"""根据页面类型智能标注学习器推荐的位置"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from button_position_learner import ButtonPositionLearner
from ocr_region_learner import OCRRegionLearner
from pathlib import Path
import json
import cv2
import numpy as np

# 创建输出目录
output_dir = Path("learning_visualization")
output_dir.mkdir(exist_ok=True)

# 定义页面类型和对应的标注元素
PAGE_ANNOTATIONS = {
    'home': {
        'buttons': ['home_checkin_button'],
        'regions': []
    },
    'checkin': {
        'buttons': [],
        'regions': ['checkin_total_times', 'checkin_remaining_times']
    },
    'profile': {
        'buttons': [],
        'regions': ['profile_balance', 'profile_points', 'profile_vouchers', 'profile_coupons']
    },
    'wallet': {
        'buttons': ['wallet_balance_button'],
        'regions': []
    }
}

def detect_page_type(img_path):
    """根据文件路径或内容检测页面类型"""
    path_str = str(img_path).lower()
    
    # 根据路径判断
    if 'checkin' in path_str or '签到' in path_str:
        return 'checkin'
    elif 'profile' in path_str or '个人' in path_str:
        return 'profile'
    elif 'wallet' in path_str or '钱包' in path_str:
        return 'wallet'
    elif 'home' in path_str or '首页' in path_str:
        return 'home'
    
    # 默认返回None，表示标注所有元素
    return None

def draw_button_position(img, button_name, position, stats):
    """在图片上绘制按钮位置"""
    x, y = position
    
    # 绘制推荐位置（红色圆点）
    cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
    cv2.circle(img, (x, y), 15, (0, 0, 255), 3)
    
    # 绘制标准差范围（半透明矩形）
    if stats and stats['x_stdev'] > 0:
        x_std = max(int(stats['x_stdev'] * 3), 20)  # 3倍标准差，最小20像素
        y_std = max(int(stats['y_stdev'] * 3), 20)
        
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (x - x_std, y - y_std), 
                     (x + x_std, y + y_std), 
                     (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        
        # 绘制边框
        cv2.rectangle(img, 
                     (x - x_std, y - y_std), 
                     (x + x_std, y + y_std), 
                     (0, 255, 0), 2)
    
    # 添加文字标签
    label = button_name.replace('_', ' ')
    label_pos = f"({x}, {y})"
    
    # 文字背景
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w1, h1), _ = cv2.getTextSize(label, font, 0.7, 2)
    (w2, h2), _ = cv2.getTextSize(label_pos, font, 0.5, 1)
    
    bg_x = x + 20
    bg_y = y - 45
    bg_w = max(w1, w2) + 20
    bg_h = 50
    
    cv2.rectangle(img, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)
    cv2.rectangle(img, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 255), 2)
    
    # 绘制文字
    cv2.putText(img, label, (bg_x + 10, bg_y + 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img, label_pos, (bg_x + 10, bg_y + 43), font, 0.5, (200, 200, 200), 1)

def draw_ocr_region(img, region_name, region, stats):
    """在图片上绘制OCR区域"""
    x, y, w, h = region
    
    # 绘制推荐区域（蓝色矩形）
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 100, 0), -1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    
    # 绘制边框
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 100, 0), 3)
    
    # 添加文字标签
    label = region_name.replace('_', ' ')
    label_size = f"{w}x{h}"
    
    # 文字背景
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w1, h1), _ = cv2.getTextSize(label, font, 0.6, 2)
    (w2, h2), _ = cv2.getTextSize(label_size, font, 0.5, 1)
    
    bg_w = max(w1, w2) + 20
    bg_h = 45
    bg_x = x
    bg_y = y - bg_h - 5
    
    # 如果标签会超出图片顶部，放到区域下方
    if bg_y < 0:
        bg_y = y + h + 5
    
    cv2.rectangle(img, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)
    cv2.rectangle(img, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (255, 100, 0), 2)
    
    # 绘制文字
    cv2.putText(img, label, (bg_x + 10, bg_y + 22), font, 0.6, (255, 255, 255), 2)
    cv2.putText(img, label_size, (bg_x + 10, bg_y + 38), font, 0.5, (200, 200, 200), 1)

def annotate_screenshot(img_path, button_learner, ocr_learner, page_type=None):
    """在截图上标注学习到的位置"""
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # 获取图片尺寸
    img_height, img_width = img.shape[:2]
    print(f"  图片尺寸: {img_width}x{img_height}")
    
    # 检测页面类型
    if page_type is None:
        page_type = detect_page_type(img_path)
    
    if page_type:
        print(f"  页面类型: {page_type}")
        annotations = PAGE_ANNOTATIONS.get(page_type, {'buttons': [], 'regions': []})
    else:
        print(f"  页面类型: 未知（标注所有元素）")
        annotations = None
    
    annotated = False
    
    # 标注按钮位置
    global_file = Path("runtime_data/button_positions/global.json")
    if global_file.exists():
        with open(global_file, 'r', encoding='utf-8') as f:
            button_data = json.load(f)
        
        for button_name in button_data.keys():
            # 如果指定了页面类型，只标注相关按钮
            if annotations and button_name not in annotations['buttons']:
                continue
            
            best_pos = button_learner.get_best_position(button_name, min_samples=5)
            if best_pos:
                # 检查坐标是否在图片范围内
                if 0 <= best_pos[0] < img_width and 0 <= best_pos[1] < img_height:
                    stats = button_learner.get_statistics(button_name)
                    draw_button_position(img, button_name, best_pos, stats)
                    print(f"    ✓ 标注按钮: {button_name} at ({best_pos[0]}, {best_pos[1]})")
                    annotated = True
    
    # 标注OCR区域
    global_file = Path("runtime_data/ocr_regions/global.json")
    if global_file.exists():
        with open(global_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        for region_name in ocr_data.keys():
            # 如果指定了页面类型，只标注相关区域
            if annotations and region_name not in annotations['regions']:
                continue
            
            best_region = ocr_learner.get_best_region(region_name, min_samples=5)
            if best_region:
                x, y, w, h = best_region
                # 检查区域是否在图片范围内
                if 0 <= x < img_width and 0 <= y < img_height and x + w <= img_width and y + h <= img_height:
                    stats = ocr_learner.get_statistics(region_name)
                    draw_ocr_region(img, region_name, best_region, stats)
                    print(f"    ✓ 标注区域: {region_name} at ({x}, {y}, {w}, {h})")
                    annotated = True
    
    return img if annotated else None

def find_screenshots_by_folder():
    """按文件夹分类查找截图"""
    screenshot_folders = {
        'checkin': [
            Path("checkin_screenshots"),
            Path("screenshots/checkin")
        ],
        'exception': [
            Path("screenshots/exception")
        ]
    }
    
    results = {}
    for folder_type, paths in screenshot_folders.items():
        screenshots = []
        for dir_path in paths:
            if dir_path.exists():
                # 查找最新的截图
                for img_file in sorted(dir_path.glob("**/*.png"), 
                                      key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                    screenshots.append(img_file)
        
        if screenshots:
            results[folder_type] = screenshots
    
    return results

def main():
    print("\n根据页面类型智能标注学习器推荐的位置")
    print("=" * 60)
    
    # 初始化学习器
    button_learner = ButtonPositionLearner()
    ocr_learner = OCRRegionLearner()
    
    # 按文件夹查找截图
    print("\n正在查找截图...")
    screenshots_by_folder = find_screenshots_by_folder()
    
    if not screenshots_by_folder:
        print("⚠️ 没有找到截图文件")
        return
    
    total_annotated = 0
    
    for folder_type, screenshots in screenshots_by_folder.items():
        print(f"\n{'='*60}")
        print(f"处理 {folder_type} 文件夹 ({len(screenshots)} 张图片)")
        print(f"{'='*60}")
        
        # 根据文件夹类型确定页面类型
        page_type = 'checkin' if folder_type == 'checkin' else None
        
        for i, img_path in enumerate(screenshots, 1):
            print(f"\n[{i}/{len(screenshots)}] 处理: {img_path.name}")
            
            annotated_img = annotate_screenshot(img_path, button_learner, ocr_learner, page_type)
            
            if annotated_img is not None:
                # 保存标注后的图片
                output_path = output_dir / f"{folder_type}_annotated_{img_path.name}"
                cv2.imwrite(str(output_path), annotated_img)
                print(f"  ✅ 已保存: {output_path.name}")
                total_annotated += 1
            else:
                print(f"  ⚠️ 跳过（无相关标注）")
    
    print("\n" + "=" * 60)
    print(f"✅ 完成！共标注 {total_annotated} 张图片")
    print(f"📁 标注图片保存在: {output_dir.absolute()}")
    
    # 打开文件夹
    if total_annotated > 0:
        print("\n正在打开文件夹...")
        import subprocess
        subprocess.run(['explorer', str(output_dir.absolute())])
    
    print("\n图例说明：")
    print("  🔴 红色圆点 = 按钮推荐位置")
    print("  🟢 绿色矩形 = 按钮位置标准差范围（3倍标准差）")
    print("  🔵 蓝色矩形 = OCR区域推荐位置")
    print("=" * 60)

if __name__ == "__main__":
    main()
