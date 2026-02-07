"""
统一数据准备脚本
Unified Dataset Preparation Script

用法:
    python prepare_dataset.py --type page_classifier
    python prepare_dataset.py --type profile_detailed
    python prepare_dataset.py --type profile_regions
    python prepare_dataset.py --type profile_numbers
    python prepare_dataset.py --type checkin_popup
    python prepare_dataset.py --type full_classifier
"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='统一数据准备脚本')
    parser.add_argument('--type', type=str, required=True,
                        choices=['page_classifier', 'profile_detailed', 'profile_regions', 
                                'profile_numbers', 'checkin_popup', 'full_classifier'],
                        help='数据集类型')
    
    args = parser.parse_args()
    
    # 根据类型调用对应的准备函数
    if args.type == 'page_classifier':
        from prepare_page_classifier_data import prepare_dataset
        prepare_dataset()
    elif args.type == 'profile_detailed':
        from prepare_profile_detailed_data import prepare_dataset
        prepare_dataset()
    elif args.type == 'profile_regions':
        from prepare_profile_region_data import prepare_dataset
        prepare_dataset()
    elif args.type == 'profile_numbers':
        from prepare_profile_numbers_dataset import prepare_dataset
        prepare_dataset()
    elif args.type == 'checkin_popup':
        from prepare_checkin_popup_dataset import prepare_dataset
        prepare_dataset()
    elif args.type == 'full_classifier':
        from prepare_full_classifier_dataset import prepare_dataset
        prepare_dataset()

if __name__ == '__main__':
    main()
