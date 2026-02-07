"""
统一数据增强脚本 - 简化版
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("数据增强脚本")
    print("=" * 60)
    print("\n请选择增强类型:")
    print("  1. 4类数据增强")
    print("  2. 页面分类器数据增强")
    print("  3. 个人页详细标注数据增强")
    print("  4. 个人页区域数据增强")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-4): ").strip()
    
    if choice == '1':
        from augment_4class_data import augment_dataset
        augment_dataset()
    elif choice == '2':
        from augment_page_classifier_updated import augment_dataset
        augment_dataset()
    elif choice == '3':
        from augment_profile_detailed_fixed import augment_dataset
        augment_dataset()
    elif choice == '4':
        from augment_profile_regions import augment_dataset
        augment_dataset()
    elif choice == '0':
        print("退出")
    else:
        print("无效选项")

if __name__ == '__main__':
    main()
