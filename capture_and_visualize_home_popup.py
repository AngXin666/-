"""
在指定截图上可视化首页广告弹窗点击位置
"""
import sys
from PIL import Image, ImageDraw, ImageFont

def visualize_home_popup(image_path):
    """在截图上可视化点击位置"""
    
    try:
        # 读取图片
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        print(f"✓ 读取截图成功: {width}x{height}")
        
        # 保存原始截图
        original_path = 'home_popup_original.png'
        img.save(original_path)
        print(f"✓ 原始截图已保存: {original_path}")
        
    except Exception as e:
        print(f"✗ 读取截图失败: {e}")
        return
    
    # 创建可视化版本
    draw = ImageDraw.Draw(img)
    
    # 尝试加载中文字体
    try:
        font_large = ImageFont.truetype("msyh.ttc", 24)  # 微软雅黑
        font_small = ImageFont.truetype("msyh.ttc", 18)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 当前点击位置 (270, 150) - 顶部
    click_x, click_y = 270, 150
    
    # 画一个绿色圆圈标记点击位置
    radius = 25
    draw.ellipse(
        [(click_x - radius, click_y - radius), 
         (click_x + radius, click_y + radius)],
        outline='lime',
        width=4
    )
    
    # 画十字准星
    cross_size = 40
    draw.line([(click_x - cross_size, click_y), (click_x + cross_size, click_y)], 
              fill='lime', width=3)
    draw.line([(click_x, click_y - cross_size), (click_x, click_y + cross_size)], 
              fill='lime', width=3)
    
    # 标注文字（带背景）
    text = f'点击位置 ({click_x}, {click_y})'
    # 绘制文字背景
    text_bbox = draw.textbbox((click_x + 35, click_y - 15), text, font=font_small)
    draw.rectangle(text_bbox, fill='black')
    draw.text((click_x + 35, click_y - 15), text, fill='lime', font=font_small)
    
    # 标注底部导航栏区域（危险区域）
    nav_y = 920
    # 画红色危险区域
    draw.rectangle([(0, nav_y - 50), (width, height)], outline='red', width=3)
    draw.line([(0, nav_y - 50), (width, nav_y - 50)], fill='red', width=3)
    
    # 标注文字
    danger_text = '底部导航栏 (危险区域)'
    danger_bbox = draw.textbbox((10, nav_y - 80), danger_text, font=font_small)
    draw.rectangle(danger_bbox, fill='black')
    draw.text((10, nav_y - 80), danger_text, fill='red', font=font_small)
    
    # 标注分类按钮位置
    category_x, category_y = 200, 920
    draw.ellipse(
        [(category_x - 20, category_y - 20), 
         (category_x + 20, category_y + 20)],
        outline='red',
        width=3
    )
    
    cat_text = f'分类按钮 ({category_x}, {category_y})'
    cat_bbox = draw.textbbox((category_x + 25, category_y - 15), cat_text, font=font_small)
    draw.rectangle(cat_bbox, fill='black')
    draw.text((category_x + 25, category_y - 15), cat_text, fill='red', font=font_small)
    
    # 画一条从旧点击位置到分类按钮的距离线（说明为什么会误触）
    old_click_y = 850
    draw.line([(270, old_click_y), (category_x, category_y)], fill='orange', width=2, dash=(5, 5))
    
    old_text = f'旧位置 (270, {old_click_y})'
    old_bbox = draw.textbbox((280, old_click_y - 20), old_text, font=font_small)
    draw.rectangle(old_bbox, fill='black')
    draw.text((280, old_click_y - 20), old_text, fill='orange', font=font_small)
    
    # 计算距离
    distance = ((270 - category_x)**2 + (old_click_y - category_y)**2)**0.5
    dist_text = f'距离: {distance:.0f}px'
    dist_bbox = draw.textbbox((235, (old_click_y + category_y)//2), dist_text, font=font_small)
    draw.rectangle(dist_bbox, fill='black')
    draw.text((235, (old_click_y + category_y)//2), dist_text, fill='orange', font=font_small)
    
    # 保存可视化结果
    output_path = 'home_popup_click_visualization.png'
    img.save(output_path)
    print(f"✓ 可视化结果已保存: {output_path}")
    
    # 打开图片
    img.show()
    
    print("\n" + "=" * 60)
    print("说明:")
    print(f"- 绿色圆圈: 当前点击位置 ({click_x}, {click_y}) - 顶部安全区域")
    print(f"- 橙色虚线: 旧点击位置 (270, {old_click_y}) 到分类按钮的距离 ({distance:.0f}px)")
    print(f"- 红色区域: 底部导航栏 - 危险区域，容易误触")
    print(f"- 红色圆圈: 分类按钮 ({category_x}, {category_y}) - 需要避免误触")
    print("=" * 60)
    print(f"\n✓ 新位置距离分类按钮: {((click_x - category_x)**2 + (click_y - category_y)**2)**0.5:.0f}px")
    print(f"✓ 旧位置距离分类按钮: {distance:.0f}px")
    print(f"✓ 安全距离提升: {((click_x - category_x)**2 + (click_y - category_y)**2)**0.5 - distance:.0f}px")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python capture_and_visualize_home_popup.py <截图路径>")
        print("示例: python capture_and_visualize_home_popup.py screenshot.png")
        sys.exit(1)
    
    visualize_home_popup(sys.argv[1])
