"""
准备个人页详细标注数据集（按钮+文字+数字）
Prepare Profile Detailed Annotation Dataset (Buttons + Text + Numbers)
"""

import os
import shutil
from pathlib import Path

# 源数据目录
SOURCE_DIR = "training_data/新已登陆页"

# 目标YOLO数据集目录
TARGET_DIR = "yolo_dataset/profile_detailed"

# 类别映射（根据用户提供的正确映射）
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

def prepare_dataset():
    """准备YOLO数据集"""
    
    print("=" * 70)
    print("准备个人页详细标注数据集")
    print("=" * 70)
    
    # 创建目录结构
    print("\n[1] 创建目录结构...")
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            path = Path(TARGET_DIR) / split / subdir
            path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ 目录结构已创建: {TARGET_DIR}")
    
    # 获取所有标注文件
    print("\n[2] 扫描标注文件...")
    source_path = Path(SOURCE_DIR)
    txt_files = list(source_path.glob("*.txt"))
    
    print(f"✓ 找到 {len(txt_files)} 个标注文件")
    
    if len(txt_files) == 0:
        print("❌ 未找到标注文件")
        return
    
    # 检查第一个标注文件的类别
    print("\n[3] 检查标注类别...")
    
    class_ids = set()
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.strip():
                try:
                    class_id = int(line.split()[0])
                    class_ids.add(class_id)
                except (ValueError, IndexError):
                    pass
    
    print(f"✓ 检测到的类别ID: {sorted(class_ids)}")
    
    # 显示类别映射
    print("\n[4] 类别映射:")
    for class_id in sorted(class_ids):
        class_name = CLASS_NAMES.get(class_id, f"未知类别_{class_id}")
        print(f"  {class_id} -> {class_name}")
    
    # 重新映射类别ID（固定映射，不依赖检测到的类别）
    print("\n[5] 重新映射类别ID...")
    # 固定映射：按照CLASS_NAMES的顺序
    sorted_class_ids = sorted(CLASS_NAMES.keys())
    class_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_class_ids)}
    
    print("  映射关系:")
    for old_id, new_id in class_id_mapping.items():
        class_name = CLASS_NAMES.get(old_id, f"未知类别_{old_id}")
        print(f"    {old_id} -> {new_id} ({class_name})")
    
    # 分割数据集（80% train, 20% val）
    print("\n[6] 分割数据集...")
    import random
    random.seed(42)
    
    txt_files_list = list(txt_files)
    random.shuffle(txt_files_list)
    
    split_idx = int(len(txt_files_list) * 0.8)
    train_files = txt_files_list[:split_idx]
    val_files = txt_files_list[split_idx:]
    
    print(f"✓ 训练集: {len(train_files)} 张")
    print(f"✓ 验证集: {len(val_files)} 张")
    
    # 复制文件并重新映射类别ID
    print("\n[7] 复制文件并重新映射类别ID...")
    
    def copy_files(file_list, split):
        """复制文件到目标目录"""
        for txt_file in file_list:
            # 找到对应的图片文件
            img_file = txt_file.with_suffix('.png')
            if not img_file.exists():
                img_file = txt_file.with_suffix('.jpg')
            
            if not img_file.exists():
                print(f"  ⚠️  未找到图片: {txt_file.stem}")
                continue
            
            # 复制图片
            target_img = Path(TARGET_DIR) / split / 'images' / img_file.name
            shutil.copy2(img_file, target_img)
            
            # 读取标注并重新映射类别ID
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 重新映射类别ID
            new_lines = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    old_class_id = int(parts[0])
                    new_class_id = class_id_mapping[old_class_id]
                    new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
            
            # 写入新的标注文件
            target_txt = Path(TARGET_DIR) / split / 'labels' / txt_file.name
            with open(target_txt, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    
    print(f"✓ 文件复制完成")
    
    # 创建data.yaml配置文件
    print("\n[8] 创建data.yaml配置文件...")
    
    # 构建类别名称列表（按固定顺序）
    sorted_class_ids = sorted(CLASS_NAMES.keys())
    class_names_list = []
    for old_id in sorted_class_ids:
        class_name = CLASS_NAMES.get(old_id, f"未知类别_{old_id}")
        class_names_list.append(class_name)
    
    yaml_content = f"""# 个人页详细标注数据集配置
# Profile Detailed Annotation Dataset Configuration

path: {os.path.abspath(TARGET_DIR)}
train: train/images
val: val/images

# 类别数量
nc: {len(class_names_list)}

# 类别名称
names: {class_names_list}
"""
    
    yaml_path = Path(TARGET_DIR) / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✓ 配置文件已创建: {yaml_path}")
    
    # 统计信息
    print("\n" + "=" * 70)
    print("数据集准备完成！")
    print("=" * 70)
    print(f"\n数据集位置: {TARGET_DIR}")
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")
    print(f"类别数量: {len(class_names_list)}")
    print(f"\n类别列表:")
    for i, name in enumerate(class_names_list):
        print(f"  {i}: {name}")
    print("\n下一步: 运行数据增强脚本增加训练数据")
    print("=" * 70)


if __name__ == '__main__':
    prepare_dataset()
