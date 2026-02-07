"""
统一分类器训练脚本
Unified Classifier Training Script

用法:
    python train_classifier.py --type keras
    python train_classifier.py --type pytorch
    python train_classifier.py --type binary
    python train_classifier.py --type 4class
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='统一分类器训练脚本')
    parser.add_argument('--type', type=str, required=True,
                        choices=['keras', 'pytorch', 'binary', '4class'],
                        help='分类器类型')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    
    args = parser.parse_args()
    
    # 根据类型调用对应的训练函数
    if args.type == 'keras':
        from train_page_classifier import main as train_keras
        train_keras()
    elif args.type == 'pytorch':
        from train_page_classifier_pytorch import main as train_pytorch
        train_pytorch()
    elif args.type == 'binary':
        from train_page_classifier_binary_v2 import main as train_binary
        train_binary()
    elif args.type == '4class':
        from train_4class_classifier import main as train_4class
        train_4class()

if __name__ == '__main__':
    main()
