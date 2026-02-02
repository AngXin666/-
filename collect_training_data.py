"""
训练数据收集工具 - 自动截图并分类保存
"""
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.adb_bridge import ADBBridge


# 页面类别
PAGE_CLASSES = [
    '首页',
    '个人页_已登录',
    '个人页_未登录',
    '签到页',
    '登录页',
    '广告页',
    '弹窗',
    '加载页',
    '其他'
]


async def collect_screenshots():
    """收集训练数据"""
    print("=" * 60)
    print("训练数据收集工具")
    print("=" * 60)
    
    # 创建数据集目录
    data_dir = Path("training_data")
    data_dir.mkdir(exist_ok=True)
    
    for page_class in PAGE_CLASSES:
        class_dir = data_dir / page_class
        class_dir.mkdir(exist_ok=True)
    
    print(f"\n数据集目录: {data_dir.absolute()}")
    print("\n页面类别:")
    for i, page_class in enumerate(PAGE_CLASSES, 1):
        print(f"  {i}. {page_class}")
    
    # 初始化 ADB
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path=adb_path)
    device_id = "127.0.0.1:16384"
    
    print(f"\n设备: {device_id}")
    print("\n" + "=" * 60)
    print("使用说明:")
    print("  1. 在模拟器中切换到要截图的页面")
    print("  2. 输入页面类别编号（1-9）")
    print("  3. 按回车自动截图并保存")
    print("  4. 输入 'q' 退出")
    print("=" * 60)
    
    screenshot_count = {page_class: 0 for page_class in PAGE_CLASSES}
    
    while True:
        print(f"\n当前统计:")
        for page_class in PAGE_CLASSES:
            count = screenshot_count[page_class]
            print(f"  {page_class}: {count} 张")
        
        # 获取用户输入
        user_input = input("\n请输入页面类别编号（1-9）或 'q' 退出: ").strip()
        
        if user_input.lower() == 'q':
            break
        
        try:
            class_idx = int(user_input) - 1
            if class_idx < 0 or class_idx >= len(PAGE_CLASSES):
                print("❌ 无效的类别编号")
                continue
        except ValueError:
            print("❌ 请输入数字")
            continue
        
        page_class = PAGE_CLASSES[class_idx]
        
        # 截图
        print(f"正在截图: {page_class}...")
        screenshot_data = await adb.screencap(device_id)
        
        if not screenshot_data:
            print("❌ 截图失败")
            continue
        
        # 保存截图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{page_class}_{timestamp}.png"
        filepath = data_dir / page_class / filename
        
        with open(filepath, 'wb') as f:
            f.write(screenshot_data)
        
        screenshot_count[page_class] += 1
        print(f"✓ 已保存: {filepath}")
    
    print("\n" + "=" * 60)
    print("数据收集完成！")
    print("=" * 60)
    print(f"\n总计:")
    total = 0
    for page_class in PAGE_CLASSES:
        count = screenshot_count[page_class]
        total += count
        print(f"  {page_class}: {count} 张")
    print(f"\n总共: {total} 张")
    
    print(f"\n数据集目录: {data_dir.absolute()}")
    print("\n下一步:")
    print("  1. 确保每个类别至少有 20-50 张截图")
    print("  2. 运行训练脚本: python train_page_classifier.py")


if __name__ == "__main__":
    asyncio.run(collect_screenshots())
