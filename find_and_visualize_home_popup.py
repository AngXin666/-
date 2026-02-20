"""
查找首页广告弹窗截图并可视化点击位置
"""
import os
from PIL import Image, ImageDraw, ImageFont
import glob

try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_OCR = True
except:
    HAS_OCR = False

def find_home_popup_screenshots():
    """查找可能是首页广告弹窗的截图（使用OCR识别）"""
    screenshot_dirs = [
        'checkin_screenshots/20260211',
        'checkin_screenshots/20260210',
        'checkin_screenshots/20260209',
        'checkin_screenshots/20260208'
    ]
    
    if not HAS_OCR:
        print("⚠️ OCR未安装，无法识别截图内容")
        return []
    
    ocr = RapidOCR()
    candidates = []
    
    for dir_path in screenshot_dirs:
        if not os.path.exists(dir_path):
            continue
            
        png_files = glob.glob(os.path.join(dir_path, '*.png'))
        
        print(f"\n检查目录: {dir_path} ({len(png_files)} 张图片)")
        
        # 检查所有图片
        for i, img_path in enumerate(png_files):
            try:
                img = Image.open(img_path)
                width, height = img.size
                
                # 检查是否是540x960的截图
                if width != 540 or height != 960:
                    continue
                
                # 使用OCR识别文字
                result, _ = ocr(img)
                
                if result:
                    texts = [line[1] for line in result]
                    text_str = " ".join(texts)
                    
                    # 检查是否包含首页广告弹窗的特征文字
                    # 首页广告弹窗通常包含：公告、活动、恭喜、领取、×（关闭按钮）
                    # 但不包含签到相关的文字
                    has_popup_keywords = any(kw in text_str for kw in ["公告", "活动", "恭喜", "领取", "×"])
                    has_checkin_keywords = any(kw in text_str for kw in ["签到", "连续签到", "已签到"])
                    
                    if has_popup_keywords and not has_checkin_keywords:
                        print(f"  ✓ 找到候选: {os.path.basename(img_path)}")
                        print(f"    识别文字: {text_str[:100]}...")
                        candidates.append((img_path, text_str))
                
                # 每10张图片显示进度
                if (i + 1) % 10 == 0:
                    print(f"  进度: {i + 1}/{len(png_files)}")
                    
            except Exception as e:
                continue
    
    return candidates

def visualize_click_position(image_path, output_path='home_popup_click_visualization.png'):
    """在真实截图上可视化点击位置"""
    try:
        # 读取图片
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # 尝试加载中文字体
        try:
            font_large = ImageFont.truetype("msyh.ttc", 20)  # 微软雅黑
            font_small = ImageFont.truetype("msyh.ttc", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 当前点击位置 (270, 150) - 顶部
        click_x, click_y = 270, 150
        
        # 画一个绿色圆圈标记点击位置
        radius = 20
        draw.ellipse(
            [(click_x - radius, click_y - radius), 
             (click_x + radius, click_y + radius)],
            outline='green',
            width=3
        )
        
        # 画十字准星
        cross_size = 30
        draw.line([(click_x - cross_size, click_y), (click_x + cross_size, click_y)], 
                  fill='green', width=2)
        draw.line([(click_x, click_y - cross_size), (click_x, click_y + cross_size)], 
                  fill='green', width=2)
        
        # 标注文字
        draw.text((click_x + 30, click_y - 10), 
                  f'点击位置 ({click_x}, {click_y})', 
                  fill='green', font=font_small)
        
        # 标注底部导航栏区域（危险区域）
        nav_y = 920
        draw.line([(0, nav_y - 50), (540, nav_y - 50)], fill='red', width=2)
        draw.text((10, nav_y - 70), '底部导航栏 (危险区域)', fill='red', font=font_small)
        
        # 标注分类按钮位置
        category_x, category_y = 200, 920
        draw.ellipse(
            [(category_x - 15, category_y - 15), 
             (category_x + 15, category_y + 15)],
            outline='red',
            width=2
        )
        draw.text((category_x + 20, category_y - 10), 
                  f'分类按钮 ({category_x}, {category_y})', 
                  fill='red', font=font_small)
        
        # 保存结果
        img.save(output_path)
        print(f"✓ 可视化结果已保存到: {output_path}")
        print(f"✓ 使用的截图: {image_path}")
        
        # 打开图片
        img.show()
        
        return True
        
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("查找首页广告弹窗截图（使用OCR识别）...")
    print("=" * 60)
    
    candidates = find_home_popup_screenshots()
    
    if not candidates:
        print("\n✗ 未找到首页广告弹窗截图")
        print("提示: 请手动指定一张首页广告弹窗的截图路径")
    else:
        print(f"\n✓ 找到 {len(candidates)} 张首页广告弹窗截图")
        print("\n候选截图:")
        for i, (path, text) in enumerate(candidates[:10], 1):
            print(f"  {i}. {path}")
            print(f"     文字: {text[:80]}...")
        
        # 使用第一张截图进行可视化
        first_image = candidates[0][0]
        print(f"\n使用第一张截图进行可视化: {first_image}")
        visualize_click_position(first_image)
        
        print("\n" + "=" * 60)
        print("说明:")
        print("- 绿色圆圈: 当前点击位置 (270, 150) - 顶部安全区域")
        print("- 红色区域: 底部导航栏 - 危险区域，容易误触")
        print("- 红色圆圈: 分类按钮 (200, 920) - 需要避免误触")
        print("=" * 60)
