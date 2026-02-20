"""
诊断并修复模型加载问题
"""
import os
import sys

print("=" * 60)
print("诊断模型加载问题")
print("=" * 60)

# 1. 检查PyTorch
print("\n[1] 检查PyTorch...")
try:
    import torch
    print(f"✓ PyTorch已安装: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
except ImportError as e:
    print(f"✗ PyTorch未安装: {e}")
    sys.exit(1)

# 2. 检查模型文件
print("\n[2] 检查模型文件...")
model_path = "models/page_classifier_pytorch_best.pth"
if not os.path.exists(model_path):
    print(f"✗ 模型文件不存在: {model_path}")
    sys.exit(1)

file_size = os.path.getsize(model_path) / 1024 / 1024
print(f"✓ 模型文件存在: {model_path}")
print(f"  文件大小: {file_size:.2f} MB")

# 3. 尝试加载模型
print("\n[3] 尝试加载模型...")
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✓ 模型加载成功")
    
    if isinstance(checkpoint, dict):
        print(f"  模型类型: dict")
        print(f"  包含的键: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            print(f"  state_dict大小: {len(checkpoint['model_state_dict'])} 个参数")
    else:
        print(f"  模型类型: state_dict")
        print(f"  参数数量: {len(checkpoint)} 个")
        
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n[修复建议]")
    print("1. 模型文件可能损坏，尝试使用备份:")
    backup_path = "models/page_classifier_pytorch_best.pth.backup_epoch4"
    if os.path.exists(backup_path):
        print(f"   发现备份文件: {backup_path}")
        print(f"   运行以下命令恢复:")
        print(f"   copy {backup_path} {model_path}")
    else:
        print(f"   未找到备份文件")
    
    print("\n2. 或者PyTorch版本不兼容，需要重新训练模型")
    sys.exit(1)

# 4. 检查类别文件
print("\n[4] 检查类别文件...")
classes_path = "models/page_classes.json"
if not os.path.exists(classes_path):
    print(f"✗ 类别文件不存在: {classes_path}")
    sys.exit(1)

import json
with open(classes_path, 'r', encoding='utf-8') as f:
    classes = json.load(f)
print(f"✓ 类别文件加载成功")
print(f"  类别数量: {len(classes)}")

# 5. 测试完整加载流程
print("\n[5] 测试完整加载流程...")
try:
    from torchvision import models
    import torch.nn as nn
    
    class PageClassifier(nn.Module):
        def __init__(self, num_classes):
            super(PageClassifier, self).__init__()
            self.mobilenet = models.mobilenet_v2(weights=None)
            in_features = self.mobilenet.classifier[1].in_features
            self.mobilenet.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            return self.mobilenet(x)
    
    num_classes = len(classes)
    model = PageClassifier(num_classes)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ 模型架构创建成功")
    print(f"  参数总数: {sum(p.numel() for p in model.parameters())}")
    
except Exception as e:
    print(f"✗ 模型架构创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有检查通过，模型可以正常加载")
print("=" * 60)
