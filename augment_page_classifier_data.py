"""
页面分类器数据增强脚本
- 对原始数据进行增强，生成更多训练样本
- 增强方法：翻转、旋转、亮度调整、对比度调整、噪声
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random


class ImageAugmentor:
    """图像增强器"""
    
    def __init__(self):
        """初始化增强器"""
        pass
    
    def horizontal_flip(self, image):
        """水平翻转"""
        return cv2.flip(image, 1)
    
    def vertical_flip(self, image):
        """垂直翻转"""
        return cv2.flip(image, 0)
    
    def rotate(self, image, angle):
        """旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def adjust_brightness(self, image, factor):
        """调整亮度
        
        Args:
            image: 输入图像
            factor: 亮度因子（0.5-1.5）
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image, factor):
        """调整对比度
        
        Args:
            image: 输入图像
            factor: 对比度因子（0.5-1.5）
        """
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    def add_noise(self, image, noise_level=10):
        """添加高斯噪声
        
        Args:
            image: 输入图像
            noise_level: 噪声强度（0-50）
        """
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    def augment_image(self, image, augmentation_type):
        """应用指定的增强方法
        
        Args:
            image: 输入图像
            augmentation_type: 增强类型
        """
        if augmentation_type == 'hflip':
            return self.horizontal_flip(image)
        elif augmentation_type == 'vflip':
            return self.vertical_flip(image)
        elif augmentation_type == 'rotate_5':
            return self.rotate(image, 5)
        elif augmentation_type == 'rotate_-5':
            return self.rotate(image, -5)
        elif augmentation_type == 'rotate_10':
            return self.rotate(image, 10)
        elif augmentation_type == 'rotate_-10':
            return self.rotate(image, -10)
        elif augmentation_type == 'bright_0.8':
            return self.adjust_brightness(image, 0.8)
        elif augmentation_type == 'bright_1.2':
            return self.adjust_brightness(image, 1.2)
        elif augmentation_type == 'contrast_0.8':
            return self.adjust_contrast(image, 0.8)
        elif augmentation_type == 'contrast_1.2':
            return self.adjust_contrast(image, 1.2)
        elif augmentation_type == 'noise':
            return self.add_noise(image, 10)
        else:
            return image


def augment_dataset(source_dir, target_dir, augment_factor=2):
    """增强数据集
    
    Args:
        source_dir: 源数据集目录
        target_dir: 目标数据集目录
        augment_factor: 增强倍数（每张图片生成多少张增强图）
    """
    print("=" * 60)
    print("页面分类器数据增强")
    print("=" * 60)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"\n❌ 源目录不存在: {source_dir}")
        return
    
    # 删除旧的目标目录
    if target_path.exists():
        print(f"\n删除旧的增强数据目录: {target_dir}")
        import shutil
        shutil.rmtree(target_path)
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"创建增强数据目录: {target_dir}")
    
    # 增强方法列表
    augmentation_types = [
        'hflip',           # 水平翻转
        'rotate_5',        # 旋转5度
        'rotate_-5',       # 旋转-5度
        'bright_0.8',      # 降低亮度
        'bright_1.2',      # 提高亮度
        'contrast_0.8',    # 降低对比度
        'contrast_1.2',    # 提高对比度
        'noise',           # 添加噪声
    ]
    
    augmentor = ImageAugmentor()
    
    total_original = 0
    total_augmented = 0
    class_stats = {}
    
    # 遍历所有类别
    for class_dir in sorted(source_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\n处理类别: {class_name}")
        
        # 创建目标类别目录
        target_class_dir = target_path / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片
        images = list(class_dir.glob('*.png'))
        if len(images) == 0:
            print(f"  跳过空文件夹")
            continue
        
        total_original += len(images)
        
        # 复制原始图片
        for img_path in images:
            target_file = target_class_dir / img_path.name
            import shutil
            shutil.copy2(img_path, target_file)
        
        # 生成增强图片
        augmented_count = 0
        for idx, img_path in enumerate(tqdm(images, desc=f"  增强 {class_name}")):
            # 读取图片 - 使用 cv2.imdecode 避免中文路径问题
            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                img_array = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    print(f"    警告: 无法读取图片 {img_path.name}")
                    continue
            except Exception as e:
                print(f"    错误: 读取图片失败 {img_path.name}: {e}")
                continue
            
            # 随机选择增强方法
            selected_augmentations = random.sample(
                augmentation_types, 
                min(augment_factor, len(augmentation_types))
            )
            
            # 应用增强
            for i, aug_type in enumerate(selected_augmentations):
                augmented = augmentor.augment_image(image, aug_type)
                
                # 保存增强图片 - 使用简短的文件名
                aug_filename = f"aug_{idx:04d}_{i}_{aug_type}.png"
                aug_path = target_class_dir / aug_filename
                
                # 使用 cv2.imencode 避免中文路径问题
                try:
                    success, encoded_img = cv2.imencode('.png', augmented)
                    if success:
                        with open(aug_path, 'wb') as f:
                            f.write(encoded_img.tobytes())
                        augmented_count += 1
                    else:
                        print(f"    警告: 编码图片失败 {aug_filename}")
                except Exception as e:
                    print(f"    错误: 保存图片失败 {aug_filename}: {e}")
        
        total_augmented += augmented_count
        class_stats[class_name] = {
            'original': len(images),
            'augmented': augmented_count,
            'total': len(images) + augmented_count
        }
        
        print(f"  ✓ 原始: {len(images)} 张，增强: {augmented_count} 张，总计: {len(images) + augmented_count} 张")
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("数据增强完成")
    print("=" * 60)
    print(f"\n原始图片: {total_original} 张")
    print(f"增强图片: {total_augmented} 张")
    print(f"总计图片: {total_original + total_augmented} 张")
    print(f"增强倍数: {total_augmented / total_original:.1f}x")
    
    print("\n各类别统计:")
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        print(f"  {class_name}: {stats['original']} → {stats['total']} 张")
    
    print(f"\n增强数据集目录: {target_dir}")
    print("\n下一步:")
    print("  修改 train_page_classifier.py 中的 data_dir 为增强数据集目录")
    print("  python train_page_classifier.py")


if __name__ == '__main__':
    # 配置参数
    source_dir = 'page_classifier_dataset'  # 原始数据集
    target_dir = 'page_classifier_dataset_augmented'  # 增强后的数据集
    augment_factor = 5  # 每张图片生成5张增强图（增加增强倍数）
    
    augment_dataset(source_dir, target_dir, augment_factor)
