"""
训练页面分类器（排除广告页）
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

class PageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # 扫描所有类别（排除广告页）
        for category_dir in sorted(self.root_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            
            # 跳过广告页
            if category_dir.name == "广告页":
                continue
            
            # 跳过错误标注目录
            if category_dir.name.startswith("_标注错误"):
                continue
            
            class_idx = len(self.classes)
            self.classes.append(category_dir.name)
            
            # 加载图片
            for img_path in category_dir.glob("*.png"):
                self.samples.append((img_path, class_idx))
        
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


def train_page_classifier():
    print("\n" + "=" * 80)
    print("🎯 页面分类器训练（排除广告页）")
    print("=" * 80)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    
    if torch.cuda.is_available():
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  • CUDA版本: {torch.version.cuda}")
        print(f"  • cuDNN启用: {torch.backends.cudnn.enabled}")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print("\n📦 加载数据集...")
    training_data_dir = Path("标注工具_完整独立版/training_data")
    dataset = PageDataset(training_data_dir, transform=transform)
    
    print(f"  • 加载了 {len(dataset)} 张图片")
    print(f"  • {len(dataset.classes)} 个类别: {', '.join(dataset.classes[:5])}...")
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 数据集划分:")
    print(f"  • 训练集: {train_size}张")
    print(f"  • 验证集: {val_size}张")
    print(f"  • 类别数: {len(dataset.classes)}")
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, persistent_workers=True)
    
    # 创建模型
    print("\n🏗️  创建模型...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(dataset.classes))
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=False)
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')
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
            
            with torch.amp.autocast('cuda'):
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
                torch.save(model.state_dict(), "models/page_classifier_without_ad_best.pth")
                
                # 保存类别列表
                with open("models/page_classes_without_ad.json", "w", encoding="utf-8") as f:
                    json.dump(dataset.classes, f, ensure_ascii=False, indent=2)
                
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
    print(f"  • 平均每轮: {duration/epochs:.0f}秒")
    print(f"  • 最佳验证准确率: {best_acc:.2f}%")
    print(f"\n💾 已保存:")
    print(f"  • 模型: models/page_classifier_without_ad_best.pth")
    print(f"  • 类别: models/page_classes_without_ad.json")
    print(f"\n⏰ 完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    try:
        train_page_classifier()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消操作")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
