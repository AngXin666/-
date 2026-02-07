"""
恢复原始标注图

从 training_data/个人页_已登录 重新筛选出余额积分和头像首页的原始标注图
"""

import json
import shutil
from pathlib import Path
from collections import Counter

def filter_and_restore_balance_data():
    """筛选并恢复余额积分数据"""
    print("\n" + "=" * 80)
    print("恢复余额积分原始标注图")
    print("=" * 80)
    
    # 读取原始标注
    source_dir = Path("training_data/个人页_已登录")
    ann_file = source_dir / "annotations.json"
    
    if not ann_file.exists():
        print("⚠ annotations.json 不存在，无法恢复")
        return 0
    
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 目标类别
    target_classes = {'余额数字', '积分数字', '抵扣劵数字', '优惠劵数字'}
    
    # 筛选数据
    filtered = {}
    for img_path, anns in data.items():
        if anns:
            # 获取这张图片的所有类别
            img_classes = {ann.get('label') or ann.get('class') for ann in anns}
            
            # 如果包含目标类别
            if img_classes & target_classes:
                # 只保留目标类别的标注
                new_anns = [ann for ann in anns if (ann.get('label') or ann.get('class')) in target_classes]
                if new_anns:
                    filtered[img_path] = new_anns
    
    print(f"找到包含目标元素的图片: {len(filtered)} 张")
    
    # 创建输出目录
    output_dir = Path("training_data_completed/个人页_已登录_余额积分")
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    
    # 删除旧的增强数据
    if output_images_dir.exists():
        shutil.rmtree(output_images_dir)
    if output_labels_dir.exists():
        shutil.rmtree(output_labels_dir)
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制图片和创建 YOLO 格式标签
    copied_count = 0
    for img_path, anns in filtered.items():
        src = Path(img_path)
        if src.exists():
            # 复制图片
            dst = output_images_dir / src.name
            shutil.copy2(src, dst)
            
            # 创建 YOLO 格式标签（如果有对应的 .txt 文件）
            txt_file = src.with_suffix('.txt')
            if txt_file.exists():
                label_dst = output_labels_dir / txt_file.name
                shutil.copy2(txt_file, label_dst)
            
            copied_count += 1
    
    # 保存 annotations.json
    output_ann = output_dir / "annotations.json"
    new_filtered = {}
    for img_path, anns in filtered.items():
        new_path = str(output_images_dir / Path(img_path).name)
        new_filtered[new_path] = anns
    
    with open(output_ann, 'w', encoding='utf-8') as f:
        json.dump(new_filtered, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 恢复完成: {copied_count} 张图片")
    return copied_count

def filter_and_restore_avatar_homepage_data():
    """筛选并恢复头像首页数据"""
    print("\n" + "=" * 80)
    print("恢复头像首页原始标注图")
    print("=" * 80)
    
    # 读取原始标注
    source_dir = Path("training_data/个人页_已登录")
    ann_file = source_dir / "annotations.json"
    
    if not ann_file.exists():
        print("⚠ annotations.json 不存在，无法恢复")
        return 0
    
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 目标类别
    target_classes = {'头像', '首页按钮'}
    
    # 筛选数据
    filtered = {}
    for img_path, anns in data.items():
        if anns:
            # 获取这张图片的所有类别
            img_classes = {ann.get('label') or ann.get('class') for ann in anns}
            
            # 如果包含目标类别
            if img_classes & target_classes:
                # 只保留目标类别的标注
                new_anns = [ann for ann in anns if (ann.get('label') or ann.get('class')) in target_classes]
                if new_anns:
                    filtered[img_path] = new_anns
    
    print(f"找到包含目标元素的图片: {len(filtered)} 张")
    
    # 创建输出目录
    output_dir = Path("training_data_completed/个人页_已登录_头像首页")
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    
    # 删除旧的增强数据
    if output_images_dir.exists():
        shutil.rmtree(output_images_dir)
    if output_labels_dir.exists():
        shutil.rmtree(output_labels_dir)
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制图片和创建 YOLO 格式标签
    copied_count = 0
    for img_path, anns in filtered.items():
        src = Path(img_path)
        if src.exists():
            # 复制图片
            dst = output_images_dir / src.name
            shutil.copy2(src, dst)
            
            # 创建 YOLO 格式标签（如果有对应的 .txt 文件）
            txt_file = src.with_suffix('.txt')
            if txt_file.exists():
                label_dst = output_labels_dir / txt_file.name
                shutil.copy2(txt_file, label_dst)
            
            copied_count += 1
    
    # 保存 annotations.json
    output_ann = output_dir / "annotations.json"
    new_filtered = {}
    for img_path, anns in filtered.items():
        new_path = str(output_images_dir / Path(img_path).name)
        new_filtered[new_path] = anns
    
    with open(output_ann, 'w', encoding='utf-8') as f:
        json.dump(new_filtered, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 恢复完成: {copied_count} 张图片")
    return copied_count

def main():
    """主函数"""
    print("=" * 80)
    print("恢复原始标注图")
    print("=" * 80)
    
    total_restored = 0
    
    # 恢复余额积分数据
    count1 = filter_and_restore_balance_data()
    total_restored += count1
    
    # 恢复头像首页数据
    count2 = filter_and_restore_avatar_homepage_data()
    total_restored += count2
    
    # 总结
    print("\n" + "=" * 80)
    print("恢复完成！")
    print("=" * 80)
    print(f"总计恢复: {total_restored} 张原始标注图")
    print("\n说明:")
    print("  - 启动页服务弹窗: 原始数据已丢失（training_data 中只有 1 个文件）")
    print("  - 登录异常: 原始数据已丢失（合并数据集的原始文件已被删除）")
    print("  - 这些数据只能从 YOLO 数据集中的增强数据恢复")

if __name__ == '__main__':
    main()
