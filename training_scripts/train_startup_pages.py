"""
启动页面分类器训练脚本
训练类别：启动页协议弹窗、广告页、加载页、首页公告、首页异常代码弹窗、首页、登录页、模拟器桌面
支持GPU加速、混合精度训练、详细进度日志、自动数据增强
"""
import os
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import shutil
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageEnhance, ImageFilter

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============ 配置区域 ============
# 指定要训练的类别（启动流程需要的页面）
# [2026-03-02] 修复原因：修正类别名称并添加缺失的类别
SELECTED_CLASSES = [
    "启动页协议弹窗",
    "广告页",
    "加载页",
    "首页公告",
    "首页异常代码弹窗",
    "首页",
    "登录页",
    "模拟器桌面"
]

# 数据增强配置
AUGMENT_THRESHOLD = 100  # 低于此数量的类别需要增强
AUGMENT_MULTIPLIER = 5   # 增强倍数
# ==================================


def augment_image(image):
    """对图片进行随机增强"""
    # 随机选择增强方式
    aug_type = random.choice(['brightness', 'contrast', 'blur', 'rotate'])
    
    if aug_type == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    elif aug_type == 'contrast':
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    elif aug_type == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    elif aug_type == 'rotate':
        angle = random.uniform(-5, 5)
        return image.rotate(angle, fillcolor=(255, 255, 255))
    
    return image


def augment_category_data(category_dir, category_name, threshold=100, multiplier=5):
    """增强指定类别的数据
    
    Args:
        category_dir: 类别目录路径
        category_name: 类别名称
        threshold: 数量阈值，低于此值需要增强
        multiplier: 增强倍数
    
    Returns:
        tuple: (原始图片数量, 增强图片数量)
    """
    # 获取原始图片（不包括已增强的）
    original_images = [f for f in category_dir.glob("*.png") if "_aug_" not in f.name]
    original_count = len(original_images)
    
    # 判断是否需要增强
    if original_count >= threshold:
        print(f"  • {category_name}: {original_count} 张 (≥{threshold}，无需增强)")
        return original_count, 0
    
    # 需要增强
    augment_count = original_count * multiplier
    print(f"  • {category_name}: {original_count} 张 (<{threshold}，增强{multiplier}倍)")
    print(f"    生成 {augment_count} 张增强图片...", end='', flush=True)
    
    # 生成增强图片
    generated = 0
    for i in range(augment_count):
        # 随机选择一张原始图片
        source_img_path = random.choice(original_images)
        
        # 加载图片
        img = Image.open(source_img_path)
        
        # 增强
        aug_img = augment_image(img)
        
        # 保存
        aug_filename = f"{source_img_path.stem}_aug_{i+1}.png"
        aug_path = category_dir / aug_filename
        aug_img.save(aug_path)
        
        generated += 1
    
    print(f" 完成")
    return original_count, generated


class SelectivePageDataset(Dataset):
    """选择性页面分类器数据集 - 只加载指定类别"""
    
    def __init__(self, data_dir, selected_classes, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = selected_classes  # 使用指定的类别列表
        
        print(f"\n📋 选择训练的类别:")
        for i, class_name in enumerate(self.classes, 1):
            print(f"  {i}. {class_name}")
        
        # 只扫描选定的类别目录
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


class PageClassifier(nn.Module):
    """页面分类器模型 - 使用MobileNetV3"""
    
    def __init__(self, num_classes):
        super(PageClassifier, self).__init__()
        
        # 使用MobileNetV3-Large作为骨干网络
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        
        # 替换MobileNetV3的分类器
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
    print("🎯 启动页面分类器训练")
    print("=" * 80)
    
    # 配置
    script_dir = Path(__file__).parent.parent
    training_data_dir = script_dir / "标注工具_完整独立版" / "training_data"
    models_dir = script_dir / "标注工具_完整独立版" / "models"
    models_dir.mkdir(exist_ok=True)
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    IMG_SIZE = (224, 224)
    
    print(f"\n📁 数据目录: {training_data_dir}")
    print(f"💾 模型目录: {models_dir}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    if device.type == 'cuda':
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    # 训练参数
    print(f"\n⚙️  训练参数:")
    print(f"  • Batch Size: {BATCH_SIZE}")
    print(f"  • Epochs: {EPOCHS}")
    print(f"  • Learning Rate: {LEARNING_RATE}")
    print(f"  • Image Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    
    # 数据增强
    print(f"\n🔄 数据增强:")
    print(f"  • 阈值: {AUGMENT_THRESHOLD} 张")
    print(f"  • 倍数: {AUGMENT_MULTIPLIER}x")
    print(f"\n正在检查并增强数据...")
    
    total_original = 0
    total_augmented = 0
    
    for class_name in SELECTED_CLASSES:
        class_dir = training_data_dir / class_name
        
        if not class_dir.exists():
            print(f"  ⚠️  {class_name}: 目录不存在，跳过")
            continue
        
        original_count, augmented_count = augment_category_data(
            class_dir, class_name, AUGMENT_THRESHOLD, AUGMENT_MULTIPLIER
        )
        total_original += original_count
        total_augmented += augmented_count
    
    print(f"\n📊 数据增强统计:")
    print(f"  • 原始图片: {total_original} 张")
    print(f"  • 增强图片: {total_augmented} 张")
    print(f"  • 总计: {total_original + total_augmented} 张")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集（只加载选定的类别）
    print(f"\n📦 加载数据集...")
    dataset = SelectivePageDataset(training_data_dir, SELECTED_CLASSES, transform=transform)
    
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
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=16, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True if device.type == 'cuda' else False)
    
    # 创建模型
    print(f"\n🏗️  创建模型...")
    model = PageClassifier(num_classes=len(dataset.classes))
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
    best_model_path = models_dir / "page_classifier_startup_best.pth"
    training_start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, EPOCHS, scaler
        )
        
        # 每5轮验证一次，或最后一轮
        if epoch % 5 == 0 or epoch == EPOCHS:
            # 验证
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            # 显示结果
            print(f"  📈 训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  📉 验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
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
        else:
            # 只显示训练结果
            print(f"  📈 训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
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
    classes_path = models_dir / "page_classes_startup.json"
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
