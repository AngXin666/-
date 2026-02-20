"""
从运行中的设备获取截图并可视化首页广告弹窗点击位置
"""
import asyncio
import sys
import os
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import yaml

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController

async def get_screenshot_and_visualize():
    """从设备获取截图并可视化"""
    
    # 从配置加载模拟器路径
    config_path = 'config.yaml'
    nox_path = None
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            nox_path = config_data.get('nox_path', '')
    
    # 初始化ADB
    adb_path = None
    if nox_path:
        emulator_controller = EmulatorController(nox_path)
        adb_path = emulator_controller.get_adb_path()
        if adb_path:
            print(f"✓ 找到ADB路径: {adb_path}")
    
    adb = ADBBridge(adb_path)
    
    # 获取设备列表
    print("正在查找设备...")
    result = adb._run_adb('devices')
    
    if result.returncode != 0:
        print("✗ 无法获取设备列表")
        return
    
    # 解析设备列表
    devices = []
    for line in result.stdout.strip().split('\n')[1:]:  # 跳过第一行 "List of devices attached"
        if line.strip():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == 'device':
                devices.append(parts[0])
    
    if not devices:
        print("✗ 未找到设备")
        return
    
    device_id = devices[0]
    print(f"✓ 使用设备: {device_id}")
    
    # 截取屏幕
    print("正在截取屏幕...")
    screenshot_data = await adb.screencap(device_id)
    
    if not screenshot_data:
        print("✗ 截图失败")
        return
    
    # 转换为PIL图片
    img = Image.open(BytesIO(screenshot_data)).convert('RGB')
    width, height = img.size
    print(f"✓ 截图成功: {width}x{height}")
    
    # 保存原始截图
    original_path = 'home_popup_original.png'
    img.save(original_path)
    print(f"✓ 原始截图已保存: {original_path}")
    
    # 创建可视化版本
    draw = ImageDraw.Draw(img)
    
    # 尝试加载中文字体
    try:
        font_large = ImageFont.truetype("msyh.ttc", 24)
        font_small = ImageFont.truetype("msyh.ttc", 18)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 当前点击位置 (290, 210) - 首页广告弹窗关闭按钮
    click_x, click_y = 290, 210
    
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
    text_bbox = draw.textbbox((click_x + 35, click_y - 15), text, font=font_small)
    draw.rectangle(text_bbox, fill='black')
    draw.text((click_x + 35, click_y - 15), text, fill='lime', font=font_small)
    
    # 标注底部导航栏区域（危险区域）
    nav_y = 920
    draw.rectangle([(0, nav_y - 50), (width, height)], outline='red', width=3)
    draw.line([(0, nav_y - 50), (width, nav_y - 50)], fill='red', width=3)
    
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
    
    # 画一条从旧点击位置到分类按钮的距离线
    old_click_y = 850
    draw.line([(270, old_click_y), (category_x, category_y)], fill='orange', width=2)
    
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
    new_distance = ((click_x - category_x)**2 + (click_y - category_y)**2)**0.5
    print(f"\n✓ 新位置距离分类按钮: {new_distance:.0f}px")
    print(f"✓ 旧位置距离分类按钮: {distance:.0f}px")
    print(f"✓ 安全距离提升: {new_distance - distance:.0f}px")

if __name__ == '__main__':
    asyncio.run(get_screenshot_and_visualize())
