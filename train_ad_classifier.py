"""
单独训练广告页分类器（二分类：广告页 vs 非广告页）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
import random

class AdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # 加载广告页图片（标签=1）
        ad_dir = self.root_dir / "广告页"
        if ad_dir.exists():
            for img_path in ad_dir.glob("*.png"):
                self.samples.append((img_path, 1))
        
        # 加载非广告页图片（标签=0）
        for category_dir in self.root_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name == "广告页":
                continue
            
            for img_path in category_dir.glob("*.png"):
                self.samples.append((img_path, 0))
        
        # 打乱数据
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_ad_classifier():
    print("\n" + "=" * 80)
    print("🎯 广告页二分类器训练")
    print("=" * 80)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    
    if torch.cuda.is_available():
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print("\n📦 加载数据集...")
    training_data_dir = Path("标注工具_完整独立版/training_data")
    dataset = AdDataset(training_data_dir, transform=transform)
    
    # 统计数据
    ad_count = sum(1 for _, label in dataset.samples if label == 1)
    non_ad_count = len(dataset) - ad_count
    
    print(f"  • 广告页: {ad_count}张")
    print(f"  • 非广告页: {non_ad_count}张")
    print(f"  • 总计: {len(dataset)}张")
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 数据集划分:")
    print(f"  • 训练集: {train_size}张")
    print(f"  • 验证集: {val_size}张")
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, persistent_workers=True)
    
    # 创建模型（二分类）
    print("\n🏗️  创建模型...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)  # 二分类
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    print("  • 混合精度训练(AMP): 已启用 ⚡")
    
    # 训练参数
    epochs = 30
    best_acc = 0.0
    
    print("\n" + "=" * 80)
    print("🚀 开始训练...")
    print("=" * 80)
    
    start_time = datetime.now()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # 验证阶段（每5轮验证一次）
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            print(f"  Epoch [{epoch+1}/{epochs}] - 训练 Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | 验证 Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "models/ad_classifier_best.pth")
                print(f"  ✨ 新的最佳模型! 验证准确率: {val_acc:.2f}%")
        else:
            print(f"  Epoch [{epoch+1}/{epochs}] - 训练 Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        scheduler.step()
    
    # 完成统计
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("✅ 训练完成!")
    print("=" * 80)
    print(f"\n📊 训练统计:")
    print(f"  • 总耗时: {duration/60:.1f}分钟")
    print(f"  • 最佳验证准确率: {best_acc:.2f}%")
    print(f"\n💾 已保存:")
    print(f"  • 模型: models/ad_classifier_best.pth")
    print(f"\n⏰ 完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    try:
        train_ad_classifier()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消操作")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
