"""
统一分类器训练脚本 - 简化版
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("分类器训练脚本")
    print("=" * 60)
    print("\n请选择分类器类型:")
    print("  1. Keras版本")
    print("  2. PyTorch版本")
    print("  3. 二分类版本")
    print("  4. 4类分类器")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-4): ").strip()
    
    if choice == '1':
        from train_page_classifier import main as train_keras
        train_keras()
    elif choice == '2':
        from train_page_classifier_pytorch import main as train_pytorch
        train_pytorch()
    elif choice == '3':
        from train_page_classifier_binary_v2 import main as train_binary
        train_binary()
    elif choice == '4':
        from train_4class_classifier import main as train_4class
        train_4class()
    elif choice == '0':
        print("退出")
    else:
        print("无效选项")

if __name__ == '__main__':
    main()
