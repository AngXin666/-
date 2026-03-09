"""
可视化Navigator默认按钮坐标
显示底部导航栏的所有按钮位置
"""

import asyncio
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.adb_bridge import ADBBridge
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO


async def visualize_navigator_buttons():
    """可视化Navigator的默认按钮坐标"""
    print("=" * 60)
    print("Navigator 默认按钮坐标可视化工具")
    print("=" * 60)
    
    # 1. 检测运行中的模拟器实例
    print("\n[1/4] 检测运行中的模拟器实例...")
    from src.emulator_controller import EmulatorController
    controller = EmulatorController()
    running_instances = await controller.get_running_instances()
    
    if not running_instances:
        print("❌ 未检测到运行中的模拟器实例")
        print("   请先启动模拟器")
        return
    
    # 使用第一个运行中的实例
    instance_id = running_instances[0]
    port = 16384 + instance_id * 32
    device_id = f"127.0.0.1:{port}"
    print(f"✓ 检测到实例 {instance_id}，设备: {device_id}")
    
    # 2. 初始化ADB
    print("\n[2/4] 初始化ADB...")
    adb = ADBBridge(controller._adb_path)
    print(f"✓ ADB路径: {controller._adb_path}")
    
    # 3. 截图
    print("\n[3/4] 截图...")
    screenshot_data = await adb.screencap(device_id)
    if not screenshot_data:
        print("❌ 截图失败")
        return
    
    image = Image.open(BytesIO(screenshot_data))
    print(f"✓ 截图成功: {image.size}")
    
    # 4. 绘制按钮标记
    print("\n[4/4] 绘制按钮标记...")
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Navigator的默认坐标（从src/navigator.py）
    # [2026-03-01] 修正：根据可视化结果调整坐标
    buttons = [
        ("首页", (70, 920), "green"),
        ("分类", (200, 920), "blue"),
        ("我的", (480, 920), "red"),
    ]
    
    print("\n📍 Navigator默认坐标:")
    print("=" * 60)
    
    # 绘制每个按钮
    for name, (x, y), color in buttons:
        # 绘制十字标记
        cross_size = 20
        draw.line([(x - cross_size, y), (x + cross_size, y)], fill=color, width=3)
        draw.line([(x, y - cross_size), (x, y + cross_size)], fill=color, width=3)
        
        # 绘制圆圈
        circle_radius = 30
        draw.ellipse(
            [(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
            outline=color,
            width=3
        )
        
        # 绘制标签（在按钮上方）
        label = f"{name}\n({x}, {y})"
        
        # 计算文本位置（居中对齐）
        bbox = draw.textbbox((0, 0), label, font=small_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x - text_width // 2
        text_y = y - circle_radius - text_height - 10
        
        # 绘制白色背景
        padding = 5
        draw.rectangle(
            [
                (text_x - padding, text_y - padding),
                (text_x + text_width + padding, text_y + text_height + padding)
            ],
            fill="white",
            outline=color,
            width=2
        )
        
        # 绘制文本
        draw.text((text_x, text_y), label, fill=color, font=small_font)
        
        # 打印坐标信息
        print(f"  {name:6s}: ({x:3d}, {y:3d}) - {color}")
    
    print("=" * 60)
    
    # 添加标题
    title = "Navigator 默认按钮坐标 (540x960)"
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (image.width - title_width) // 2
    title_y = 20
    
    # 绘制标题背景
    draw.rectangle(
        [
            (title_x - 10, title_y - 5),
            (title_x + title_width + 10, title_y + 30)
        ],
        fill="white",
        outline="black",
        width=2
    )
    draw.text((title_x, title_y), title, fill="black", font=font)
    
    # 添加说明文字
    info_text = "绿色=首页, 蓝色=分类, 红色=我的"
    info_bbox = draw.textbbox((0, 0), info_text, font=small_font)
    info_width = info_bbox[2] - info_bbox[0]
    info_x = (image.width - info_width) // 2
    info_y = 60
    
    draw.rectangle(
        [
            (info_x - 5, info_y - 3),
            (info_x + info_width + 5, info_y + 20)
        ],
        fill="white",
        outline="black",
        width=1
    )
    draw.text((info_x, info_y), info_text, fill="black", font=small_font)
    
    # 保存图片
    output_path = "navigator_buttons_visualization.png"
    image.save(output_path)
    print(f"\n✓ 可视化图片已保存: {output_path}")
    
    # 打开图片
    import os
    os.startfile(output_path)
    print(f"✓ 已自动打开图片")
    
    print("\n" + "=" * 60)
    print("可视化完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(visualize_navigator_buttons())
