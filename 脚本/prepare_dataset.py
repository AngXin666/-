"""
统一数据准备脚本 - 简化版
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("数据准备脚本")
    print("=" * 60)
    print("\n请选择数据集类型:")
    print("  1. 页面分类器数据")
    print("  2. 个人页详细标注数据")
    print("  3. 个人页区域数据")
    print("  4. 个人页数字数据")
    print("  5. 签到弹窗数据")
    print("  6. 完整分类器数据")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-6): ").strip()
    
    if choice == '1':
        from prepare_page_classifier_data import prepare_dataset
        prepare_dataset()
    elif choice == '2':
        from prepare_profile_detailed_data import prepare_dataset
        prepare_dataset()
    elif choice == '3':
        from prepare_profile_region_data import prepare_dataset
        prepare_dataset()
    elif choice == '4':
        from prepare_profile_numbers_dataset import prepare_dataset
        prepare_dataset()
    elif choice == '5':
        from prepare_checkin_popup_dataset import prepare_dataset
        prepare_dataset()
    elif choice == '6':
        from prepare_full_classifier_dataset import prepare_dataset
        prepare_dataset()
    elif choice == '0':
        print("退出")
    else:
        print("无效选项")

if __name__ == '__main__':
    main()
