"""
可视化个人页标注
Visualize Profile Annotations
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 类别映射 (更新后的)
CLASS_NAMES = {
    12: "首页",
    13: "我的",
    15: "抵扣券数字",
    16: "优惠券数字",
    21: "余额数字",
    22: "积分数字",
    23: "昵称文字",
    24: "ID文字"
}

# 颜色映射
COLORS = {
    12: (255, 165, 0),   # 橙色 - 首页
    13: (0, 255, 0),     # 绿色 - 我的
    15: (0, 0, 255),     # 蓝色 - 抵扣券数字
    16: (255, 0, 255),   # 紫色 - 优惠券数字
    21: (255, 255, 0),   # 黄色 - 余额数字
    22: (0, 255, 255),   # 青色 - 积分数字
    23: (255, 192, 203), # 粉色 - 昵称文字
    24: (255, 0, 0)      # 红色 - ID文字 (重点标注)
}

def visualize_annotation(image_path, txt_path, output_path):
    """可视化单个标注"""
    
    # 读取图片
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # 读取标注
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    img_width, img_height = image.size
    
    # 绘制每个标注框
    for line in lines:
        if not line.strip():
            continue
        
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # 转换为像素坐标
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # 获取颜色和类别名称
        color = COLORS.get(class_id, (255, 255, 255))
        class_name = CLASS_NAMES.get(class_id, f"未知_{class_id}")
        
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制类别标签
        label = f"{class_id}: {class_name}"
        
        # 绘制标签背景
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 获取文本边界框
        bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)
    
    # 保存结果
    image.save(output_path)
    print(f"✓ 已保存: {output_path}")


def main():
    """主函数"""
    
    print("=" * 70)
    print("可视化个人页标注")
    print("=" * 70)
    
    # 源目录
    source_dir = Path("training_data/新已登陆页")
    output_dir = Path("debug_annotations")
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有标注文件
    txt_files = list(source_dir.glob("*.txt"))
    
    # 只可视化前5个文件
    print(f"\n找到 {len(txt_files)} 个标注文件，可视化前5个...")
    
    for i, txt_file in enumerate(txt_files[:5], 1):
        # 找到对应的图片
        img_file = txt_file.with_suffix('.png')
        if not img_file.exists():
            img_file = txt_file.with_suffix('.jpg')
        
        if not img_file.exists():
            print(f"  ⚠️  未找到图片: {txt_file.stem}")
            continue
        
        # 输出路径
        output_path = output_dir / f"annotated_{i}_{img_file.name}"
        
        print(f"\n[{i}] {img_file.name}")
        
        # 读取并显示标注内容
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"  标注数量: {len(lines)}")
        for line in lines:
            if line.strip():
                class_id = int(line.split()[0])
                class_name = CLASS_NAMES.get(class_id, f"未知_{class_id}")
                print(f"    - 类别 {class_id}: {class_name}")
        
        # 可视化
        visualize_annotation(img_file, txt_file, output_path)
    
    print(f"\n{'='*70}")
    print(f"可视化完成！")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
