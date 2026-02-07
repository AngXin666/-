"""
训练页面分类器 - 二分类模式
为每个页面类型训练一个独立的二分类模型（是/否该页面）
"""
import os
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import shutil
import random

# 配置GPU
print("=" * 60)
print("检查GPU配置")
print("=" * 60)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ 找到 {len(gpus)} 个GPU设备")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    except RuntimeError as e:
        print(f"✗ GPU配置失败: {e}")
else:
    print("✗ 未找到GPU设备，将使用CPU训练（速度较慢）")
print("=" * 60)

def load_images_for_binary_classification(dataset_dir, target_class):
    """
    加载二分类数据
    
    Args:
        dataset_dir: 数据集目录
        target_class: 目标类别名称
        
    Returns:
        images: 图片数组
        labels: 标签数组（1=目标类别，0=其他类别）
        class_counts: 类别统计
    """
    print(f"\n加载二分类数据: {target_class}")
    print("=" * 60)
    
    dataset_path = Path(dataset_dir)
    images = []
    labels = []
    
    positive_count = 0  # 正样本数量
    negative_count = 0  # 负样本数量
    
    # 遍历所有类别文件夹
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        is_target = (class_name == target_class)
        label = 1 if is_target else 0
        
        # 获取该类别的所有图片
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        
        print(f"  {'✓' if is_target else ' '} {class_name}: {len(image_files)} 张图片")
        
        for img_path in image_files:
            try:
                # 读取并预处理图片
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(label)
                
                if is_target:
                    positive_count += 1
                else:
                    negative_count += 1
                    
            except Exception as e:
                print(f"  ✗ 加载图片失败 {img_path}: {e}")
    
    print(f"\n数据统计:")
    print(f"  正样本（{target_class}）: {positive_count} 张")
    print(f"  负样本（其他类别）: {negative_count} 张")
    print(f"  总计: {positive_count + negative_count} 张")
    
    return np.array(images), np.array(labels), {
        'positive': positive_count,
        'negative': negative_count,
        'total': positive_count + negative_count
    }

def create_binary_model(input_shape=(224, 224, 3)):
    """
    创建二分类模型（轻量版，CPU训练更快）
    
    Args:
        input_shape: 输入图片尺寸
        
    Returns:
        model: Keras模型
    """
    # 使用函数式API，更兼容
    from tensorflow.keras import Input, Model
    
    inputs = Input(shape=input_shape)
    
    # 卷积层1 - 减少通道数
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # 卷积层2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # 卷积层3
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # 全连接层 - 减少神经元数量
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # 输出层（二分类使用sigmoid）
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_binary_classifier(dataset_dir, target_class, output_dir="binary_models"):
    """
    训练单个二分类器
    
    Args:
        dataset_dir: 数据集目录
        target_class: 目标类别名称
        output_dir: 模型输出目录
    """
    print("\n" + "=" * 60)
    print(f"开始训练二分类器: {target_class}")
    print("=" * 60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 加载数据
    images, labels, class_counts = load_images_for_binary_classification(
        dataset_dir, target_class
    )
    
    if len(images) == 0:
        print(f"✗ 没有找到数据，跳过训练")
        return None
    
    # 检查正样本数量
    if class_counts['positive'] < 5:
        print(f"✗ 正样本数量太少（{class_counts['positive']}张），建议至少5张")
        return None
    
    # 划分训练集和验证集（手动实现分层划分）
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 分别处理正负样本
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    
    # 打乱索引
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)
    
    # 计算验证集大小（20%）
    val_size_positive = max(1, int(len(positive_indices) * 0.2))
    val_size_negative = max(1, int(len(negative_indices) * 0.2))
    
    # 划分索引
    val_indices = np.concatenate([
        positive_indices[:val_size_positive],
        negative_indices[:val_size_negative]
    ])
    train_indices = np.concatenate([
        positive_indices[val_size_positive:],
        negative_indices[val_size_negative:]
    ])
    
    # 打乱训练集和验证集索引
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # 创建训练集和验证集
    X_train = images[train_indices]
    y_train = labels[train_indices]
    X_val = images[val_indices]
    y_val = labels[val_indices]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 张")
    print(f"  验证集: {len(X_val)} 张")
    
    # 创建模型
    print(f"\n创建模型...")
    model = create_binary_model()
    
    # 编译模型（二分类使用binary_crossentropy）
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # 打印模型结构
    print("\n模型结构:")
    model.summary()
    
    # 设置回调函数
    callbacks = [
        # 早停
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # 保存最佳模型
        keras.callbacks.ModelCheckpoint(
            str(output_path / f"{target_class}_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 训练模型
    print(f"\n开始训练...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估模型
    print("\n" + "=" * 60)
    print("训练完成，评估模型...")
    print("=" * 60)
    
    train_loss, train_acc, train_precision, train_recall = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n训练集性能:")
    print(f"  准确率: {train_acc:.4f}")
    print(f"  精确率: {train_precision:.4f}")
    print(f"  召回率: {train_recall:.4f}")
    
    print(f"\n验证集性能:")
    print(f"  准确率: {val_acc:.4f}")
    print(f"  精确率: {val_precision:.4f}")
    print(f"  召回率: {val_recall:.4f}")
    
    # 保存最终模型
    model_path = output_path / f"{target_class}.h5"
    model.save(model_path)
    print(f"\n✓ 模型已保存: {model_path}")
    
    # 保存训练信息
    info = {
        'class_name': target_class,
        'positive_samples': int(class_counts['positive']),
        'negative_samples': int(class_counts['negative']),
        'total_samples': int(class_counts['total']),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'train_precision': float(train_precision),
        'val_precision': float(val_precision),
        'train_recall': float(train_recall),
        'val_recall': float(val_recall),
        'model_path': str(model_path)
    }
    
    info_path = output_path / f"{target_class}_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 训练信息已保存: {info_path}")
    
    return info

def train_all_binary_classifiers(dataset_dir="page_classifier_dataset_augmented", output_dir="binary_models"):
    """
    为所有类别训练二分类器
    
    Args:
        dataset_dir: 数据集目录
        output_dir: 模型输出目录
    """
    print("=" * 60)
    print("开始训练所有二分类器")
    print("=" * 60)
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"✗ 数据集目录不存在: {dataset_dir}")
        return
    
    # 获取所有类别
    classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    classes.sort()
    
    print(f"\n找到 {len(classes)} 个类别:")
    for i, class_name in enumerate(classes, 1):
        print(f"  {i}. {class_name}")
    
    # 训练每个类别的二分类器
    results = []
    for i, class_name in enumerate(classes, 1):
        print(f"\n{'='*60}")
        print(f"进度: {i}/{len(classes)}")
        print(f"{'='*60}")
        
        info = train_binary_classifier(dataset_dir, class_name, output_dir)
        if info:
            results.append(info)
    
    # 保存总体结果
    print("\n" + "=" * 60)
    print("所有模型训练完成")
    print("=" * 60)
    
    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_classes': len(classes),
            'trained_models': len(results),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 训练总结已保存: {summary_path}")
    
    # 打印总结
    print("\n训练总结:")
    print(f"  总类别数: {len(classes)}")
    print(f"  成功训练: {len(results)}")
    print(f"  失败/跳过: {len(classes) - len(results)}")
    
    if results:
        print("\n各模型性能:")
        print(f"{'类别':<20} {'验证准确率':>10} {'精确率':>10} {'召回率':>10}")
        print("-" * 60)
        for info in results:
            print(f"{info['class_name']:<20} {info['val_accuracy']:>9.2%} {info['val_precision']:>9.2%} {info['val_recall']:>9.2%}")

def main():
    """主函数"""
    train_all_binary_classifiers()

if __name__ == '__main__':
    main()
