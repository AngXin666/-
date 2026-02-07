"""
统一YOLO训练脚本
Unified YOLO Training Script

用法:
    python train_yolo.py --type stage1
    python train_yolo.py --type profile_regions
    python train_yolo.py --type profile_numbers
    python train_yolo.py --type profile_detailed
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='统一YOLO训练脚本')
    parser.add_argument('--type', type=str, required=True,
                        choices=['stage1', 'profile_regions', 'profile_numbers', 'profile_detailed'],
                        help='YOLO模型类型')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    # 根据类型调用对应的训练函数
    if args.type == 'stage1':
        from train_yolo_stage1 import train_model
        train_model(epochs=args.epochs, batch=args.batch)
    elif args.type == 'profile_regions':
        from train_profile_regions_yolo import train_profile_regions_detector
        train_profile_regions_detector()
    elif args.type == 'profile_numbers':
        from train_profile_numbers_yolo import train_profile_numbers_detector
        train_profile_numbers_detector()
    elif args.type == 'profile_detailed':
        from train_profile_detailed import train_model
        train_model()

if __name__ == '__main__':
    main()
