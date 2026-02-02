"""
生成模型版本文件工具

用法：
    python tools/generate_model_version.py --version 1.0.0
    python tools/generate_model_version.py --version 1.0.1 --description "更新登录检测模型"
    python tools/generate_model_version.py --version 1.0.1 --models-dir models/
"""

import sys
import argparse
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_updater import ModelVersionManager


def main():
    parser = argparse.ArgumentParser(description='生成模型版本文件')
    parser.add_argument('--version', type=str, default='1.0.0', help='版本号')
    parser.add_argument('--description', type=str, default='', help='版本描述')
    parser.add_argument('--models-dir', type=str, default='models', help='模型目录')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ 模型目录不存在: {models_dir}")
        return 1
    
    print("=" * 60)
    print("生成模型版本文件")
    print("=" * 60)
    print(f"模型目录: {models_dir.absolute()}")
    print(f"版本号: {args.version}")
    if args.description:
        print(f"描述: {args.description}")
    print()
    
    manager = ModelVersionManager(models_dir)
    manager.generate_version_file(args.version, args.description)
    
    print()
    print("=" * 60)
    print("完成！")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
