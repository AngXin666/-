"""
验号专用页面分类器训练脚本
支持GPU加速、混合精度训练、详细进度日志

验号专用模型用于账号验证流程中的页面识别，包括：
- 个人页广告, 个人页已登陆, 个人页未登陆
- 首页, 地址页, 数据中心, 首页广告
- 设置, 登陆页

架构与其他专用模型保持一致：MobileNetV3Large + 960→1280→num_classes
"""
import os
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import shutil
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# [2026-03-10] 验号专用模型的类别
VALIDATION_CLASSES = [
    "个人页广告",
    "个人页已登陆", 
    "个人页未登陆",
    "首页",
    "地址页",
    "数据中心",
    "首页广告",
    "设置",
    "登陆页"
]


class ValidationPageDataset(Dataset):
    """验号专用页面分类器数据集"""
    
    def __init__(self, data_dir, classes, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = classes
        
        print(f"\n📋 验号专用模型类别:")
        for i, class_name in enumerate(self.classes, 1):
            print(f"  {i}. {class_name}")
        
        # 扫描所有类别目录
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"  ⚠️  警告: 类别目录不存在: {class_name}")
                continue
            
            class_idx = self.classes.index(class_name)
            
            # 扫描该类别的所有图片
            img_count = 0
            for img_path in class_dir.glob("*.png"):
                self.samples.append((str(img_path), class_idx))
                img_count += 1
            
            print(f"  ✓ {class_name}: {img_count} 张图片")
        
        print(f"\n  • 总计: {len(self.samples)} 张图片")
        print(f"  • 类别数: {len(self.classes)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ValidationPageClassifier(nn.Module):
    """验号专用页面分类器模型 - 使用MobileNetV3"""
    
    def __init__(self, num_classes):
        super(ValidationPageClassifier, self).__init__()
        
        # [2026-03-10] 使用MobileNetV3-Large作为骨干网络（与其他专用模型一致）
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        
        # [2026-03-10] 修复原因：正确替换MobileNetV3的分类器
        # MobileNetV3的classifier是一个Sequential，包含多个层
        # 我们需要替换最后的Linear层，保持与其他专用模型完全一致的架构
        in_features = self.mobilenet.classifier[0].in_features  # 960
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}分{seconds % 60:.0f}秒"
    else:
        return f"{seconds // 3600:.0f}小时{(seconds % 3600) // 60:.0f}分"


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, scaler=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        if scaler is not None and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 显示进度
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            progress = (batch_idx + 1) / len(train_loader) * 100
            
            # 估算剩余时间
            elapsed = time.time() - start_time
            if batch_idx > 0:
                time_per_batch = elapsed / (batch_idx + 1)
                remaining_batches = len(train_loader) - (batch_idx + 1)
                eta = time_per_batch * remaining_batches
                eta_str = format_time(eta)
            else:
                eta_str = "计算中..."
            
            bar_length = 30
            filled = int(bar_length * (batch_idx + 1) / len(train_loader))
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r  Epoch [{epoch}/{total_epochs}] "
                  f"[{bar}] {progress:.1f}% "
                  f"Loss: {avg_loss:.4f} Acc: {accuracy:.2f}% "
                  f"ETA: {eta_str}", end='', flush=True)
    
    print()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    epoch_time = time.time() - start_time
    
    return epoch_loss, epoch_acc, epoch_time


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main():
    """主训练函数"""
    print("\n" + "=" * 80)
    print("🎯 验号专用页面分类器训练")
    print("=" * 80)
    
    # [2026-03-10] 不固定随机种子，保持训练随机性，提高模型泛化能力
    print(f"\n🎲 随机种子: 未固定 (保持随机性)")
    
    # 配置
    script_dir = Path(__file__).parent
    training_data_dir = Path("标注工具_完整独立版/training_data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    IMG_SIZE = (224, 224)
    VAL_INTERVAL = 5  # 每5轮验证一次
    
    print(f"\n📁 数据目录: {training_data_dir}")
    print(f"💾 模型目录: {models_dir}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    if device.type == 'cuda':
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # [2026-03-10] 启用benchmark模式以提高训练速度
        torch.backends.cudnn.benchmark = True
    
    # 训练参数
    print(f"\n⚙️  训练参数:")
    print(f"  • Batch Size: {BATCH_SIZE}")
    print(f"  • Epochs: {EPOCHS}")
    print(f"  • Learning Rate: {LEARNING_RATE}")
    print(f"  • Image Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print(f"\n📦 加载数据集...")
    dataset = ValidationPageDataset(training_data_dir, VALIDATION_CLASSES, transform=transform)
    
    if len(dataset) == 0:
        print("\n❌ 错误: 没有找到训练数据")
        return
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # [2026-03-10] 不固定生成器种子，保持数据划分的随机性
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"\n📊 数据集划分:")
    print(f"  • 训练集: {len(train_dataset)} 张")
    print(f"  • 验证集: {len(val_dataset)} 张")
    
    # 创建数据加载器
    # [2026-03-10] 不固定数据加载顺序，保持随机性
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=8, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True if device.type == 'cuda' else False)
    
    # 创建模型
    print(f"\n🏗️  创建模型...")
    model = ValidationPageClassifier(num_classes=len(dataset.classes))
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print(f"  • 混合精度训练(AMP): 已启用 ⚡")
    
    # 训练
    print("\n" + "=" * 80)
    print("🚀 开始训练...")
    print("=" * 80)
    
    best_val_acc = 0.0
    best_model_path = models_dir / "page_classifier_validation_best.pth"
    training_start_time = time.time()
    
    # 保存验证历史
    last_val_loss = 0.0
    last_val_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, EPOCHS, scaler
        )
        
        # 只在指定间隔或最后一轮进行验证
        should_validate = (epoch % VAL_INTERVAL == 0) or (epoch == EPOCHS)
        
        if should_validate:
            # 验证
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            # 更新历史
            last_val_loss = val_loss
            last_val_acc = val_acc
        else:
            # 使用上次的验证结果
            val_loss = last_val_loss
            val_acc = last_val_acc
        
        # 显示结果
        print(f"  📈 训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        if should_validate:
            print(f"  📉 验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        else:
            print(f"  📉 验证 - 跳过 (上次: Acc {val_acc:.2f}%)")
        
        # 保存最佳模型（只在验证时更新）
        if should_validate and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': dataset.classes
            }, best_model_path)
            print(f"  ✨ 新的最佳模型! 验证准确率: {val_acc:.2f}%")
        
        print()
    
    # 训练完成
    total_time = time.time() - training_start_time
    
    print("=" * 80)
    print("✅ 训练完成!")
    print("=" * 80)
    print(f"\n📊 训练统计:")
    print(f"  • 总耗时: {format_time(total_time)}")
    print(f"  • 最佳验证准确率: {best_val_acc:.2f}%")
    
    # 保存类别列表
    classes_path = models_dir / "page_classes_validation.json"
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(dataset.classes, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 已保存:")
    print(f"  • 模型: {best_model_path}")
    print(f"  • 类别: {classes_path}")
    print(f"\n⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消训练")
    except Exception as e:
        print(f"\n\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()