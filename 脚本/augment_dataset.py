"""
统一数据增强脚本
Unified Data Augmentation Script

用法:
    python augment_dataset.py --type 4class
    python augment_dataset.py --type page_classifier
    python augment_dataset.py --type profile_detailed
    python augment_dataset.py --type profile_regions
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='统一数据增强脚本')
    parser.add_argument('--type', type=str, required=True,
                        choices=['4class', 'page_classifier', 'profile_detailed', 'profile_regions'],
                        help='数据增强类型')
    
    args = parser.parse_args()
    
    # 根据类型调用对应的增强函数
    if args.type == '4class':
        from augment_4class_data import augment_dataset
        augment_dataset()
    elif args.type == 'page_classifier':
        from augment_page_classifier_updated import augment_dataset
        augment_dataset()
    elif args.type == 'profile_detailed':
        from augment_profile_detailed_fixed import augment_dataset
        augment_dataset()
    elif args.type == 'profile_regions':
        from augment_profile_regions import augment_dataset
        augment_dataset()

if __name__ == '__main__':
    main()
