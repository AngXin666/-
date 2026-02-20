"""
检查首页广告弹窗的点击位置
捕获截图并可视化当前点击位置 (270, 160)
"""
import asyncio
import sys
from pathlib import Path
from io import BytesIO

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("错误：需要安装 Pillow 库")
    sys.exit(1)

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.adb_bridge import ADBBridge
from src.page_detector_integrated import PageDetectorIntegrated
from src.page_detector import PageState


async def main():
    """主函数"""
    print("=" * 60)
    print("首页广告弹窗点击位置检查工具")
    print("=" * 60)
    
    # 初始化ADB
    print("\n[1] 初始化ADB...")
    
    # 从配置加载模拟器路径
    import yaml
    import os
    config_path = 'config.yaml'
    nox_path = None
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            nox_path = config_data.get('nox_path', '')
    
    # 初始化模拟器控制器以获取ADB路径
    adb_path = None
    if nox_path:
        from src.emulator_controller import EmulatorController
        emulator_controller = EmulatorController(nox_path)
        adb_path = emulator_controller.get_adb_path()
        if adb_path:
            print(f"✓ 找到ADB路径: {adb_path}")
        else:
            print("⚠️ 未找到ADB路径")
    else:
        print("⚠️ 未配置模拟器路径")
    
    adb = ADBBridge(adb_path)
    
    # 使用固定设备ID（MuMu模拟器）
    device_id = "127.0.0.1:7555"
    print(f"✓ 使用设备: {device_id}")
    
    # 初始化页面检测器
    print("\n[2] 初始化页面检测器...")
    detector = PageDetectorIntegrated(adb)
    
    # 检测当前页面
    print("\n[3] 检测当前页面...")
    result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
    print(f"当前页面: {result.state.value} (置信度: {result.confidence:.2%})")
    
    # 不管是什么页面，都进行截图和标注
    if result.state != PageState.HOME_NOTICE:
        print(f"\n⚠️ 检测到的页面不是首页广告弹窗")
        print(f"但仍然继续截图和标注，请手动确认...")
    else:
        print(f"✓ 检测到首页广告弹窗")
    
    # 获取截图
    print("\n[4] 获取截图...")
    screenshot_data = await adb.screencap(device_id)
    if not screenshot_data:
        print("❌ 无法获取截图")
        print("请确认：")
        print("1. 模拟器是否正在运行")
        print("2. 设备ID是否正确: 127.0.0.1:16384")
        print("3. ADB连接是否正常")
        return
    
    img = Image.open(BytesIO(screenshot_data))
    print(f"✓ 截图尺寸: {img.size}")
    
    # 在截图上标注点击位置
    print("\n[5] 标注点击位置...")
    draw = ImageDraw.Draw(img)
    
    # 尝试加载中文字体
    try:
        font_large = ImageFont.truetype("msyh.ttc", 28)
        font_small = ImageFont.truetype("msyh.ttc", 20)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 当前点击位置 (370, 290) - 继续下移50像素
    click_x, click_y = 370, 290
    
    # 画一个大红色圆圈标记点击位置
    radius = 30
    draw.ellipse(
        [(click_x - radius, click_y - radius), 
         (click_x + radius, click_y + radius)],
        outline='red',
        width=5
    )
    
    # 画十字准星
    cross_size = 50
    draw.line([(click_x - cross_size, click_y), (click_x + cross_size, click_y)], 
              fill='red', width=4)
    draw.line([(click_x, click_y - cross_size), (click_x, click_y + cross_size)], 
              fill='red', width=4)
    
    # 标注文字（带背景）
    text = f'当前点击位置 ({click_x}, {click_y})'
    text_bbox = draw.textbbox((click_x + 40, click_y - 20), text, font=font_large)
    draw.rectangle(text_bbox, fill='black')
    draw.text((click_x + 40, click_y - 20), text, fill='red', font=font_large)
    
    # 标注底部导航栏区域
    nav_y = 920
    draw.rectangle([(0, nav_y - 50), (img.width, img.height)], outline='yellow', width=3)
    
    danger_text = '底部导航栏 (分类按钮在 200, 920)'
    danger_bbox = draw.textbbox((10, nav_y - 80), danger_text, font=font_small)
    draw.rectangle(danger_bbox, fill='black')
    draw.text((10, nav_y - 80), danger_text, fill='yellow', font=font_small)
    
    # 标注分类按钮位置
    category_x, category_y = 200, 920
    draw.ellipse(
        [(category_x - 25, category_y - 25), 
         (category_x + 25, category_y + 25)],
        outline='yellow',
        width=4
    )
    
    # 保存标注后的截图
    output_path = "popup_click_position_check.png"
    img.save(output_path)
    print(f"✓ 已保存标注截图: {output_path}")
    
    # 显示截图
    print("\n[6] 打开截图...")
    img.show()
    
    print("\n" + "=" * 60)
    print("检查说明：")
    print(f"1. 红色圆圈和十字准星标记了当前点击位置 ({click_x}, {click_y})")
    print("2. 黄色区域是底部导航栏")
    print("3. 黄色圆圈是分类按钮位置 (200, 920)")
    print("4. 请检查红色标记是否在广告弹窗的关闭按钮上")
    print("5. 如果红色标记在广告内容上，说明坐标不正确")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
