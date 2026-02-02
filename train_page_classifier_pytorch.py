"""
页面分类器训练脚本 - 使用PyTorch和GPU加速
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class PageDataset(Dataset):
    """页面图片数据集"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
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
        
        # 使用预训练的MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # 冻结特征提取层
        for param in self.mobilenet.features.parameters():
            param.requires_grad = False
        
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


def load_dataset(data_dir, img_size=(224, 224)):
    """加载数据集"""
    print("加载数据集...")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
    
    # 获取所有类别
    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    print(f"\n检测到 {len(classes)} 个类别:")
    for i, cls in enumerate(classes, 1):
        print(f"  {i}. {cls}")
    
    # 加载所有图片路径和标签
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = data_path / class_name
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        print(f"  加载 {class_name}: {len(image_files)} 张")
        
        for img_path in image_files:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"\n总计: {len(image_paths)} 张图片")
    
    return image_paths, labels, classes


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_path='page_classifier_pytorch.pth'):
    """训练模型"""
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc='训练')
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='验证')
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f'✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)')
        
        # 学习率调整
        scheduler.step(val_loss)
    
    return history


def main():
    """主函数"""
    print("=" * 60)
    print("页面分类器训练 (PyTorch + GPU)")
    print("=" * 60)
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 参数设置
    data_dir = "page_classifier_dataset_updated"  # 使用更新后的数据集
    img_size = (224, 224)
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    
    # 1. 加载数据集
    print("\n" + "=" * 60)
    print("1. 加载数据集")
    print("=" * 60)
    image_paths, labels, classes = load_dataset(data_dir, img_size)
    
    # 2. 划分训练集和验证集
    print("\n" + "=" * 60)
    print("2. 划分数据集")
    print("=" * 60)
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=123, stratify=labels
    )
    print(f"训练集: {len(X_train)} 张")
    print(f"验证集: {len(X_val)} 张")
    
    # 3. 数据变换
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. 创建数据加载器
    train_dataset = PageDataset(X_train, y_train, transform=train_transform)
    val_dataset = PageDataset(X_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 5. 创建模型
    print("\n" + "=" * 60)
    print("3. 创建模型")
    print("=" * 60)
    model = PageClassifier(num_classes=len(classes))
    model = model.to(device)
    print("✓ 模型已创建并移至GPU")
    
    # 6. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 7. 训练模型
    print("\n" + "=" * 60)
    print("4. 开始训练")
    print("=" * 60)
    print(f"训练参数:")
    print(f"  - 训练轮数: {num_epochs} epochs")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 优化器: Adam")
    print(f"  - 学习率调度: ReduceLROnPlateau")
    
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs, device, save_path='page_classifier_pytorch_best.pth'
    )
    
    # 8. 保存最终模型和类别列表
    print("\n" + "=" * 60)
    print("5. 保存模型")
    print("=" * 60)
    torch.save(model.state_dict(), 'page_classifier_pytorch.pth')
    print("✓ 最终模型已保存: page_classifier_pytorch.pth")
    
    with open('page_classes.json', 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print("✓ 类别列表已保存: page_classes.json")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n模型文件: page_classifier_pytorch.pth")
    print(f"最佳模型: page_classifier_pytorch_best.pth")
    print(f"类别列表: page_classes.json")
    print(f"类别数量: {len(classes)}")


if __name__ == "__main__":
    main()
