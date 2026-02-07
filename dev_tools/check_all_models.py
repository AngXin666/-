"""检查所有模型信息"""
import torch
from pathlib import Path

models_to_check = [
    "models/page_classifier_pytorch_best.pth",
    "标注工具_完整独立版/模型导出/page_classifier_20260207_190233/page_classifier_pytorch_best.pth",
    "标注工具_完整独立版/模型导出/page_classifier_20260207_203733/page_classifier_pytorch_best.pth",
]

for model_path_str in models_to_check:
    model_path = Path(model_path_str)
    
    print(f"\n{'='*80}")
    print(f"模型: {model_path}")
    print(f"{'='*80}")
    
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            print(f"  ✓ Epoch: {checkpoint.get('epoch', '未知')}")
            print(f"  ✓ 验证准确率: {checkpoint.get('val_acc', '未知'):.2f}%")
            print(f"  ✓ 验证损失: {checkpoint.get('val_loss', '未知')}")
            
            # 检查文件大小
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ 文件大小: {size_mb:.2f} MB")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    else:
        print(f"  ❌ 文件不存在")
