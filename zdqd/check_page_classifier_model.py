"""
检查页面分类器模型
Check Page Classifier Model
"""

import torch
import json
from pathlib import Path

def check_model(model_path='page_classifier_pytorch_best.pth'):
    """检查模型包含的类别"""
    print(f'检查模型: {model_path}')
    print('='*60)
    
    if not Path(model_path).exists():
        print(f'❌ 模型文件不存在: {model_path}')
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'classes' in checkpoint:
                classes = checkpoint['classes']
                print(f'✓ 模型包含类别信息')
                print(f'总类别数: {len(classes)}')
                print(f'\n类别列表:')
                for i, cls in enumerate(classes):
                    marker = '  ← 个人页广告' if cls == '个人页广告' else ''
                    print(f'  {i:2d}: {cls}{marker}')
                
                # 检查是否包含个人页广告
                if '个人页广告' in classes:
                    idx = classes.index('个人页广告')
                    print(f'\n✓ 模型包含【个人页广告】类别')
                    print(f'   索引: {idx}')
                else:
                    print(f'\n❌ 模型不包含【个人页广告】类别')
                    print(f'\n建议: 需要重新训练模型，包含个人页广告类别')
                
                return classes
            else:
                print('⚠️ 模型checkpoint中没有classes信息')
                print('   这可能是旧版本的模型')
                return None
        else:
            print('⚠️ 模型是state_dict格式，没有类别信息')
            return None
            
    except Exception as e:
        print(f'❌ 加载模型失败: {e}')
        import traceback
        traceback.print_exc()
        return None

def check_training_data():
    """检查训练数据集"""
    print(f'\n检查训练数据集')
    print('='*60)
    
    dataset_path = Path('page_classifier_dataset')
    if not dataset_path.exists():
        print(f'❌ 训练数据集不存在: {dataset_path}')
        return
    
    # 列出所有类别文件夹
    class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    class_folders.sort()
    
    print(f'训练数据集包含 {len(class_folders)} 个类别:')
    
    for folder in class_folders:
        # 统计图片数量
        images = list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
        marker = '  ← 个人页广告' if folder.name == '个人页广告' else ''
        print(f'  {folder.name}: {len(images)} 张图片{marker}')
    
    # 检查个人页广告
    ad_folder = dataset_path / '个人页广告'
    if ad_folder.exists():
        images = list(ad_folder.glob('*.png')) + list(ad_folder.glob('*.jpg'))
        print(f'\n✓ 训练数据包含【个人页广告】类别')
        print(f'   图片数量: {len(images)}')
        
        if len(images) < 30:
            print(f'   ⚠️ 图片数量较少，建议至少50张')
    else:
        print(f'\n❌ 训练数据不包含【个人页广告】类别')

if __name__ == '__main__':
    # 检查模型
    classes = check_model()
    
    # 检查训练数据
    check_training_data()
    
    print(f'\n总结')
    print('='*60)
    
    if classes and '个人页广告' in classes:
        print('✓ 模型已包含个人页广告类别')
        print('  如果检测不到，可能是以下原因：')
        print('  1. 模型准确率不够（需要更多训练数据）')
        print('  2. 图片预处理问题（分辨率、归一化）')
        print('  3. 置信度阈值设置过高')
    else:
        print('❌ 模型不包含个人页广告类别')
        print('  需要重新训练模型！')
        print('  步骤：')
        print('  1. 确保 page_classifier_dataset/个人页广告/ 有足够图片')
        print('  2. 运行: python train_page_classifier_pytorch.py')
