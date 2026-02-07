"""
页面分类器训练脚本 - 使用深度学习识别页面
"""
import os
import sys
import numpy as np
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    print("❌ 未安装 TensorFlow，请运行: pip install tensorflow")
    HAS_TF = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    print("❌ 未安装 PIL，请运行: pip install pillow")
    HAS_PIL = False

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    print("❌ 未安装 scikit-learn，请运行: pip install scikit-learn")
    HAS_SKLEARN = False


class PageClassifier:
    """页面分类器 - 使用卷积神经网络"""
    
    def __init__(self, img_size=(224, 224), data_dir="training_data"):
        """初始化分类器
        
        Args:
            img_size: 输入图片大小,默认224x224
            data_dir: 数据集目录
        """
        self.img_size = img_size
        self.data_dir = data_dir
        
        # 从数据集目录自动读取类别
        self.PAGE_CLASSES = self._load_classes()
        self.num_classes = len(self.PAGE_CLASSES)
        self.model = None
    
    def _load_classes(self):
        """从数据集目录加载类别列表"""
        data_path = Path(self.data_dir)
        if not data_path.exists():
            print(f"⚠️ 数据集目录不存在: {self.data_dir}")
            return []
        
        # 读取所有子目录作为类别
        classes = []
        for item in sorted(data_path.iterdir()):
            if item.is_dir():
                # 统计该类别的图片数量
                png_count = len(list(item.glob("*.png")))
                if png_count > 0:
                    classes.append(item.name)
                    print(f"  - {item.name}: {png_count} 张")
        
        return classes
    
    def build_model(self):
        """构建CNN模型（使用MobileNetV2作为基础）"""
        if not HAS_TF:
            raise ImportError("需要安装 TensorFlow")
        
        # 使用预训练的MobileNetV2作为特征提取器
        base_model = keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # 冻结基础模型的权重
        base_model.trainable = False
        
        # 构建完整模型
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # 预处理（不使用数据增强）
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # 特征提取
        x = base_model(x, training=False)
        
        # 分类头
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # 编译模型
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',  # 修改为 sparse
            metrics=['accuracy']
        )
        
        print("✓ 模型构建完成")
        self.model.summary()
    
    def prepare_dataset(self, data_dir: str):
        """准备训练数据集（手动加载，避免编码问题）
        
        Args:
            data_dir: 数据集根目录
        """
        if not HAS_TF or not HAS_PIL:
            raise ImportError("需要安装 TensorFlow 和 PIL")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
        
        print("手动加载数据集...")
        
        # 手动加载所有图片
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.PAGE_CLASSES):
            class_dir = data_path / class_name
            if not class_dir.exists():
                continue
            
            image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            print(f"  加载 {class_name}: {len(image_files)} 张")
            
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"    跳过损坏图片: {img_path.name}")
        
        # 转换为numpy数组
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"\n总计加载: {len(images)} 张图片")
        
        # 划分训练集和验证集
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=0.2, random_state=123, stratify=labels
            )
        except Exception as e:
            print(f"⚠️ 使用sklearn划分数据集失败: {e}")
            print("使用手动划分...")
            # 手动划分数据集
            indices = np.arange(len(images))
            np.random.seed(123)
            np.random.shuffle(indices)
            split_idx = int(len(images) * 0.8)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            X_train, y_train = images[train_indices], labels[train_indices]
            X_val, y_val = images[val_indices], labels[val_indices]
        
        print(f"训练集: {len(X_train)} 张")
        print(f"验证集: {len(X_val)} 张")
        
        # 创建TensorFlow数据集
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def train(self, train_ds, val_ds, epochs=20):
        """训练模型
        
        Args:
            train_ds: 训练数据集
            val_ds: 验证数据集
            epochs: 训练轮数
        """
        if self.model is None:
            raise ValueError("请先调用 build_model() 构建模型")
        
        # 回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                'page_classifier_best.keras',
                monitor='val_accuracy',
                save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3
            )
        ]
        
        # 训练
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def save_model(self, filepath='page_classifier.keras'):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未构建")
        
        self.model.save(filepath)
        print(f"✓ 模型已保存到: {filepath}")
    
    def load_model(self, filepath='page_classifier.keras'):
        """加载模型"""
        if not HAS_TF:
            raise ImportError("需要安装 TensorFlow")
        
        # 加载模型
        self.model = keras.models.load_model(filepath)
        print(f"✓ 模型已加载: {filepath}")
    
    def predict(self, image_path: str):
        """预测单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            (类别名称, 置信度)
        """
        if self.model is None:
            raise ValueError("模型未加载")
        
        if not HAS_PIL:
            raise ImportError("需要安装 PIL")
        
        # 加载并预处理图片
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, 0)  # 添加batch维度
        
        # 预测
        predictions = self.model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return self.PAGE_CLASSES[class_idx], confidence


def main():
    """主函数 - 训练流程"""
    print("=" * 60)
    print("页面分类器训练")
    print("=" * 60)
    
    if not HAS_TF or not HAS_PIL:
        print("\n请先安装依赖:")
        print("  pip install tensorflow pillow")
        return
    
    # 检查GPU可用性
    print("\nGPU 检查:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ 检测到 {len(gpus)} 个GPU:")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        # 设置GPU内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️ 未检测到GPU，将使用CPU训练（速度较慢）")
    
    # 1. 创建分类器
    data_dir = "page_classifier_dataset"  # 使用原始数据集（不增强）
    classifier = PageClassifier(img_size=(224, 224), data_dir=data_dir)
    
    # 检查是否有类别
    if not classifier.PAGE_CLASSES:
        print(f"\n❌ 数据集目录为空或不存在: {data_dir}")
        print("\n请先收集训练数据!")
        return
    
    print(f"\n检测到 {len(classifier.PAGE_CLASSES)} 个类别:")
    for i, cls in enumerate(classifier.PAGE_CLASSES, 1):
        print(f"  {i}. {cls}")
    
    # 2. 构建模型
    print("\n1. 构建模型...")
    classifier.build_model()
    
    # 3. 准备数据集
    print("\n2. 准备数据集...")
    
    if not os.path.exists(data_dir):
        print(f"\n❌ 数据集目录不存在: {data_dir}")
        return
    
    try:
        train_ds, val_ds = classifier.prepare_dataset(data_dir)
        print("✓ 数据集准备完成")
    except Exception as e:
        print(f"❌ 数据集准备失败: {e}")
        return
    
    # 4. 训练模型
    print("\n3. 开始训练...")
    print("训练参数:")
    print("  - 训练轮数: 30 epochs")
    print("  - 早停机制: patience=5")
    print("  - 学习率衰减: patience=3")
    print("  - 数据增强: 禁用（使用原始数据）")
    try:
        history = classifier.train(train_ds, val_ds, epochs=30)
        print("✓ 训练完成")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return
    
    # 5. 保存模型
    print("\n4. 保存模型...")
    classifier.save_model('page_classifier.keras')
    
    # 6. 保存类别列表
    print("\n5. 保存类别列表...")
    import json
    with open('page_classes.json', 'w', encoding='utf-8') as f:
        json.dump(classifier.PAGE_CLASSES, f, ensure_ascii=False, indent=2)
    print("✓ 类别列表已保存到: page_classes.json")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n模型文件: page_classifier.keras")
    print(f"最佳模型: page_classifier_best.keras")
    print(f"类别列表: page_classes.json")
    print(f"类别数量: {len(classifier.PAGE_CLASSES)}")
    print("\n使用方法:")
    print("  from train_page_classifier import PageClassifier")
    print("  classifier = PageClassifier()")
    print("  classifier.load_model('page_classifier.keras')")
    print("  page_type, confidence = classifier.predict('screenshot.png')")


if __name__ == "__main__":
    main()
