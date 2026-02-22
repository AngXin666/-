"""
页面分类器训练脚本 - PyTorch版本
支持GPU加速、混合精度训练、详细进度日志
"""
import os
import sys
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class PageClassifierDataset(Dataset):
    """页面分类器数据集"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # 扫描所有类别目录
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in self.classes:
                self.classes.append(class_name)
            
            class_idx = self.classes.index(class_name)
            
            # 扫描该类别的所有图片
            for img_path in class_dir.glob("*.png"):
                self.samples.append((str(img_path), class_idx))
        
        print(f"  • 加载了 {len(self.samples)} 张图片")
        print(f"  • {len(self.classes)} 个类别: {', '.join(self.classes[:5])}{'...' if len(self.classes) > 5 else ''}")
    
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


class PageClassifier(nn.Module):
    """页面分类器模型 - 使用MobileNetV2"""
    
    def __init__(self, num_classes):
        super(PageClassifier, self).__init__()
        
        # 使用MobileNetV2作为骨干网络
        self.mobilenet = models.mobilenet_v2(weights=None)
        
        # 替换分类器
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


def check_gpu_status():
    """检查GPU状态并给出优化建议"""
    if not torch.cuda.is_available():
        print("\n⚠️  警告: 未检测到CUDA支持的GPU")
        print("  建议:")
        print("  1. 检查是否安装了CUDA版本的PyTorch")
        print("  2. 运行: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("  3. 确认显卡驱动已正确安装")
        return False
    
    print(f"\n✓ GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"  • CUDA版本: {torch.version.cuda}")
    print(f"  • 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 测试GPU性能
    try:
        # 创建测试张量
        test_tensor = torch.randn(1000, 1000).cuda()
        start = time.time()
        for _ in range(100):
            _ = test_tensor @ test_tensor
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  • GPU性能测试: {elapsed:.3f}秒 (100次矩阵乘法)")
        
        if elapsed > 1.0:
            print(f"  ⚠️  GPU性能较低，可能是:")
            print(f"     - 使用的是集成显卡")
            print(f"     - GPU驱动未正确安装")
            print(f"     - GPU被其他程序占用")
    except Exception as e:
        print(f"  ⚠️  GPU测试失败: {e}")
        return False
    
    return True


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}分{seconds % 60:.0f}秒"
    else:
        return f"{seconds // 3600:.0f}小时{(seconds % 3600) // 60:.0f}分"


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, scaler=None):
    """训练一个epoch - 支持混合精度训练"""
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
            # CPU训练或不使用混合精度
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 显示进度 - 减少打印频率提升速度
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
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
    
    print()  # 换行
    
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


def clean_augmented_images(training_data_dir):
    """清理增强的图片"""
    print("\n🧹 清理增强图片...")
    
    deleted_count = 0
    for class_dir in training_data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        # 删除增强图片
        for img_path in class_dir.glob("*_aug_*.png"):
            img_path.unlink()
            deleted_count += 1
    
    print(f"  ✓ 已删除 {deleted_count} 张增强图片")


def main():
    """主训练函数"""
    print("\n" + "=" * 80)
    print("🎯 页面分类器训练 (PyTorch)")
    print("=" * 80)
    
    # 配置
    script_dir = Path(__file__).parent.parent
    training_data_dir = script_dir / "training_data"
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # 训练参数
    BATCH_SIZE = 256  # 保持256以确保训练质量
    EPOCHS = 30
    LEARNING_RATE = 0.001
    IMG_SIZE = (224, 224)  # 保持224以确保准确率
    VAL_INTERVAL = 5  # 每5轮验证一次
    
    print(f"\n📁 数据目录: {training_data_dir}")
    print(f"💾 模型目录: {models_dir}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查GPU状态
    gpu_available = check_gpu_status()
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    if device.type == 'cuda':
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  • CUDA版本: {torch.version.cuda}")
        print(f"  • cuDNN启用: {torch.backends.cudnn.enabled}")
        # 启用cuDNN自动优化
        torch.backends.cudnn.benchmark = True
    else:
        print(f"  ⚠️  警告: 未检测到GPU，将使用CPU训练（速度会很慢）")
    
    # 训练参数
    print(f"\n⚙️  训练参数:")
    print(f"  • Batch Size: {BATCH_SIZE}")
    print(f"  • Epochs: {EPOCHS}")
    print(f"  • Learning Rate: {LEARNING_RATE}")
    print(f"  • Image Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"  • 验证间隔: 每{VAL_INTERVAL}轮验证一次")
    print(f"  • 数据加载: 8线程 (训练) / 4线程 (验证) + persistent_workers")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print(f"\n📦 加载数据集...")
    dataset = PageClassifierDataset(training_data_dir, transform=transform)
    
    if len(dataset) == 0:
        print("\n❌ 错误: 没有找到训练数据")
        return
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 数据集划分:")
    print(f"  • 训练集: {len(train_dataset)} 张")
    print(f"  • 验证集: {len(val_dataset)} 张")
    print(f"  • 类别数: {len(dataset.classes)}")
    
    # [2026-02-22] 修改原因：提升数据加载速度，使用10线程训练/2线程验证
    # 创建数据加载器 - 使用多线程加速
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=10, pin_memory=True if device.type == 'cuda' else False,
                             persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=2, pin_memory=True if device.type == 'cuda' else False,
                           persistent_workers=True)
    
    # 创建模型
    print(f"\n🏗️  创建模型...")
    model = PageClassifier(num_classes=len(dataset.classes))
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 创建混合精度训练的GradScaler（仅GPU）
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print(f"  • 混合精度训练(AMP): 已启用 ⚡")
    
    # 训练
    print("\n" + "=" * 80)
    print("🚀 开始训练...")
    print("=" * 80)
    
    best_val_acc = 0.0
    best_model_path = models_dir / "page_classifier_pytorch_best.pth"
    training_start_time = time.time()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, EPOCHS, scaler
        )
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 只在指定间隔或最后一轮进行验证
        should_validate = (epoch % VAL_INTERVAL == 0) or (epoch == EPOCHS)
        
        if should_validate:
            # 验证
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 记录验证历史
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        else:
            # 不验证时使用上一次的验证结果
            val_loss = history['val_loss'][-1] if history['val_loss'] else 0.0
            val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
        
        # 计算时间
        epoch_time = time.time() - epoch_start_time
        elapsed_total = time.time() - training_start_time
        avg_epoch_time = elapsed_total / epoch
        remaining_epochs = EPOCHS - epoch
        eta_total = avg_epoch_time * remaining_epochs
        
        # 显示结果
        print(f"  📈 训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        if should_validate:
            print(f"  📉 验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        else:
            print(f"  📉 验证 - 跳过 (上次: Acc {val_acc:.2f}%)")
        print(f"  ⏱️  耗时: {format_time(epoch_time)} | "
              f"总耗时: {format_time(elapsed_total)} | "
              f"预计剩余: {format_time(eta_total)}")
        
        # 保存最佳模型（只在验证时更新）
        if should_validate and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
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
    print(f"  • 平均每轮: {format_time(total_time / EPOCHS)}")
    print(f"  • 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  • 最终训练准确率: {history['train_acc'][-1]:.2f}%")
    print(f"  • 最终验证准确率: {history['val_acc'][-1]:.2f}%")
    
    # 保存类别列表
    classes_path = models_dir / "page_classes.json"
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(dataset.classes, f, ensure_ascii=False, indent=2)
    
    # 生成模型版本文件
    version_path = models_dir / "model_version.json"
    model_size_mb = best_model_path.stat().st_size / (1024 * 1024)
    classes_size_mb = classes_path.stat().st_size / (1024 * 1024)
    
    version_info = {
        "version": "1.0.0",
        "update_date": datetime.now().strftime("%Y-%m-%d"),
        "description": f"页面分类器训练完成 - {len(dataset.classes)}个类别",
        "models": {
            "page_classifier": {
                "version": "1.0.0",
                "file": "page_classifier_pytorch_best.pth",
                "size_mb": round(model_size_mb, 2),
                "description": "页面分类器（PyTorch）"
            },
            "page_classes": {
                "version": "1.0.0",
                "file": "page_classes.json",
                "size_mb": round(classes_size_mb, 2),
                "description": "页面类别映射"
            }
        }
    }
    
    with open(version_path, 'w', encoding='utf-8') as f:
        json.dump(version_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 已保存:")
    print(f"  • 模型: {best_model_path}")
    print(f"  • 类别: {classes_path}")
    print(f"  • 版本: {version_path}")
    
    # 不自动清理增强图片，留待验证后手动清理
    print(f"\n💡 提示: 增强图片已保留，可用于验证模型")
    print(f"💡 提示: 验证完成后可手动删除增强图片")
    
    print(f"\n⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    print("\n💡 提示: 模型已保存到 models/ 目录")
    print("💡 提示: 可以使用该模型进行页面分类识别")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消训练")
    except Exception as e:
        print(f"\n\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
