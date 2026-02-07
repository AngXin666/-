"""
训练4类页面分类器（首页、签到页、温馨提示、签到弹窗）

用法：
    python train_4class_classifier.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import random


class PageDataset(Dataset):
    """页面数据集"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # 扫描所有类别文件夹
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.classes.append(class_name)
                class_idx = len(self.classes) - 1
                
                # 扫描该类别下的所有图片
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), class_idx))
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), class_idx))
        
        print(f"  找到 {len(self.classes)} 个类别: {self.classes}")
        print(f"  总图片数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_model():
    """训练4类页面分类器"""
    print("=" * 60)
    print("训练4类页面分类器")
    print("=" * 60)
    
    # 配置
    dataset_dir = "page_classifier_dataset_4classes_augmented"  # 使用增强后的数据集
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print(f"\n📦 加载数据集...")
    full_dataset = PageDataset(dataset_dir, transform=train_transform)
    
    # 划分训练集和验证集 (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 为验证集设置不同的transform
    val_dataset.dataset.transform = val_transform
    
    print(f"  训练集: {train_size} 张")
    print(f"  验证集: {val_size} 张")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Windows使用0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    print(f"\n🏗️  创建模型...")
    num_classes = len(full_dataset.classes)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    print(f"  模型: ResNet18")
    print(f"  类别数: {num_classes}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练
    print(f"\n🚀 开始训练...")
    print(f"  轮数: {num_epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
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
        
        # 更新学习率
        scheduler.step()
        
        # 打印进度
        print(f"  Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss/len(val_loader):.4f} "
              f"Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': full_dataset.classes,
            }, 'page_classifier_4classes_best.pth')
            print(f"    ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
    
    print(f"\n✅ 训练完成!")
    print(f"  最佳验证准确率: {best_acc:.2f}%")
    print(f"  模型已保存: page_classifier_4classes_best.pth")
    
    # 保存类别映射
    import json
    with open('page_classes_4.json', 'w', encoding='utf-8') as f:
        json.dump({
            'classes': full_dataset.classes,
            'num_classes': len(full_dataset.classes)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"  类别映射已保存: page_classes_4.json")


if __name__ == "__main__":
    train_model()
