"""
转换个人页标注文件的类别ID
Convert Profile Page Label Class IDs

从原始标注的类别ID (15, 16, 21, 22) 转换为新数据集的类别ID (0-3)
"""

from pathlib import Path


def convert_labels():
    """转换标注文件的类别ID"""
    
    # 类别ID映射
    # 原始ID -> 新ID
    class_mapping = {
        21: 0,  # 余额数字
        22: 1,  # 积分数字
        15: 2,  # 抵扣劵数字
        16: 3,  # 优惠劵数字
        23: 4,  # 昵称文本
        24: 5,  # 用户ID
    }
    
    print("=" * 70)
    print("转换个人页标注文件的类别ID")
    print("=" * 70)
    
    print(f"\n[类别映射]")
    class_names = {
        21: "余额数字",
        22: "积分数字",
        15: "抵扣劵数字",
        16: "优惠劵数字",
        23: "昵称文本",
        24: "用户ID",
    }
    for old_id, new_id in class_mapping.items():
        print(f"  {old_id} ({class_names[old_id]}) -> {new_id}")
    
    # 标注文件目录
    label_dir = Path("yolo_dataset/profile_numbers/labels/train")
    
    if not label_dir.exists():
        print(f"\n❌ 标注目录不存在: {label_dir}")
        return
    
    # 获取所有标注文件
    label_files = list(label_dir.glob("*.txt"))
    
    if not label_files:
        print(f"\n❌ 没有找到标注文件")
        return
    
    print(f"\n[处理文件]")
    print(f"  找到 {len(label_files)} 个标注文件")
    
    converted_count = 0
    
    for label_file in label_files:
        # 读取原始标注
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 转换类别ID
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_class_id = int(parts[0])
                
                if old_class_id in class_mapping:
                    new_class_id = class_mapping[old_class_id]
                    parts[0] = str(new_class_id)
                    new_lines.append(' '.join(parts) + '\n')
                else:
                    print(f"  ⚠ 未知类别ID: {old_class_id} in {label_file.name}")
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # 写回文件
        with open(label_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        converted_count += 1
    
    print(f"\n✅ 转换完成！")
    print(f"  处理文件数: {converted_count}")
    print(f"  保存位置: {label_dir}")


if __name__ == '__main__':
    convert_labels()
