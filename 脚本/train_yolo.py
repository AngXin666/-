"""
统一YOLO训练脚本 - 简化版
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("YOLO训练脚本")
    print("=" * 60)
    print("\n请选择模型类型:")
    print("  1. 阶段1模型（核心按钮）")
    print("  2. 个人页区域检测")
    print("  3. 个人页数字识别")
    print("  4. 个人页详细标注")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-4): ").strip()
    
    if choice == '1':
        from train_yolo_stage1 import train_model
        train_model()
    elif choice == '2':
        from train_profile_regions_yolo import train_profile_regions_detector
        train_profile_regions_detector()
    elif choice == '3':
        from train_profile_numbers_yolo import train_profile_numbers_detector
        train_profile_numbers_detector()
    elif choice == '4':
        from train_profile_detailed import train_model
        train_model()
    elif choice == '0':
        print("退出")
    else:
        print("无效选项")

if __name__ == '__main__':
    main()
