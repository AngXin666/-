"""
YOLO训练管理器 - 统一的训练、准备、测试脚本

⚠️  重要说明：
    这是一键训练脚本，用于所有页面元素的YOLO模型训练
    文件名：yolo_train_manager.py
    ❌ 禁止删除此文件！后期训练新模型时需要使用
    
功能：
  1. prepare  - 准备数据集（自动数据增强、划分训练集/验证集）
  2. train    - 训练模型（GPU加速、自动早停）
  3. test     - 测试模型（全量测试+随机抽样10张截图）
  4. cleanup  - 训练后整理（保存原始图、删除增强数据、注册模型）
  5. all      - 完整流程（准备→训练→测试→整理）

用法：
  python yolo_train_manager.py prepare 分类页    # 准备数据集
  python yolo_train_manager.py train 分类页 --epochs 30      # 训练模型（指定轮数）
  python yolo_train_manager.py test 分类页       # 测试模型
  python yolo_train_manager.py cleanup 分类页    # 训练后整理
  python yolo_train_manager.py all 分类页        # 完整流程

已完成训练的模型：
  - 分类页、搜索页、积分页、文章页、钱包页
  - 个人页广告、首页异常代码弹窗
  - 其他20+个页面元素检测模型
"""
import json
import shutil
import random
import glob
import cv2
import os
import subprocess
import platform
from pathlib import Path
from PIL import Image, ImageEnhance
import yaml
import torch
from ultralytics import YOLO
from datetime import datetime
import argparse


class YOLOTrainManager:
    """YOLO训练管理器"""
    
    def __init__(self, page_type):
        self.page_type = page_type
        self.source_dir = Path(f"training_data/{page_type}")
        self.dataset_dir = Path(f"yolo_dataset_{page_type}")
        self.model_name = f"{page_type}_detector"
        
    def check_annotations(self):
        """检查标注数据"""
        annotation_file = self.source_dir / "annotations.json"
        if not annotation_file.exists():
            print(f"❌ 找不到标注文件: {annotation_file}")
            return None
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        total = len(annotations)
        annotated = sum(1 for v in annotations.values() if v)
        unannotated = total - annotated
        
        # 获取所有类别
        classes = set()
        for boxes in annotations.values():
            if boxes:
                for box in boxes:
                    classes.add(box['class'])
        
        print(f"\n📊 标注数据统计:")
        print(f"  总图片数: {total}")
        print(f"  已标注: {annotated}")
        print(f"  未标注: {unannotated}")
        print(f"  类别: {sorted(classes)}")
        
        return {
            'total': total,
            'annotated': annotated,
            'unannotated': unannotated,
            'classes': sorted(classes),
            'annotations': annotations
        }
    
    def augment_image(self, image_path, output_dir, base_name, annotations, augment_factor=15):
        """数据增强"""
        img = Image.open(image_path)
        augmented_data = []
        
        # 1. 原图
        original_path = output_dir / f"{base_name}_original.png"
        img.save(original_path)
        augmented_data.append((str(original_path), annotations))
        
        # 根据增强倍数生成不同的增强
        if augment_factor >= 5:
            # 亮度调整
            for i, factor in enumerate([0.7, 0.85, 1.15, 1.3], 1):
                enhancer = ImageEnhance.Brightness(img)
                bright_img = enhancer.enhance(factor)
                path = output_dir / f"{base_name}_bright_{i}.png"
                bright_img.save(path)
                augmented_data.append((str(path), annotations))
        
        if augment_factor >= 10:
            # 对比度调整
            for i, factor in enumerate([0.6, 0.8, 1.2, 1.4], 1):
                enhancer = ImageEnhance.Contrast(img)
                contrast_img = enhancer.enhance(factor)
                path = output_dir / f"{base_name}_contrast_{i}.png"
                contrast_img.save(path)
                augmented_data.append((str(path), annotations))
        
        if augment_factor >= 15:
            # 色彩和锐度
            for i, factor in enumerate([0.7, 1.15, 1.3], 1):
                enhancer = ImageEnhance.Color(img)
                color_img = enhancer.enhance(factor)
                path = output_dir / f"{base_name}_color_{i}.png"
                color_img.save(path)
                augmented_data.append((str(path), annotations))
            
            for i, factor in enumerate([0.5, 1.3], 1):
                enhancer = ImageEnhance.Sharpness(img)
                sharp_img = enhancer.enhance(factor)
                path = output_dir / f"{base_name}_sharp_{i}.png"
                sharp_img.save(path)
                augmented_data.append((str(path), annotations))
        
        # 只返回需要的数量
        return augmented_data[:augment_factor]
    
    def prepare_dataset(self, augment_factor=None):
        """准备YOLO数据集"""
        print(f"\n{'='*60}")
        print(f"准备 {self.page_type} 数据集")
        print(f"{'='*60}\n")
        
        # 检查标注
        data_info = self.check_annotations()
        if not data_info:
            return False
        
        annotations = data_info['annotations']
        annotated_count = data_info['annotated']
        
        # 自动选择增强倍数
        if augment_factor is None:
            if annotated_count < 20:
                augment_factor = 20
            elif annotated_count < 50:
                augment_factor = 15
            elif annotated_count < 100:
                augment_factor = 10
            else:
                augment_factor = 5
        
        print(f"\n📦 数据增强配置:")
        print(f"  原始图片: {annotated_count}张")
        print(f"  增强倍数: {augment_factor}x")
        print(f"  预计生成: {annotated_count * augment_factor}张")
        
        # 创建临时增强目录
        temp_dir = Path(f"training_data/{self.page_type}_temp_augmented")
        temp_dir.mkdir(exist_ok=True)
        
        # 数据增强
        print(f"\n🎨 开始数据增强...")
        augmented_annotations = {}
        for image_path_str, anns in annotations.items():
            if not anns:
                continue
            
            image_path = Path(image_path_str)
            if not image_path.exists():
                continue
            
            base_name = image_path.stem
            augmented_list = self.augment_image(image_path, temp_dir, base_name, anns, augment_factor)
            
            for aug_path, aug_anns in augmented_list:
                augmented_annotations[aug_path] = aug_anns
        
        print(f"  ✓ 生成了 {len(augmented_annotations)} 张增强图片")
        
        # 创建YOLO数据集目录
        self.dataset_dir.mkdir(exist_ok=True)
        for split in ['train', 'val']:
            (self.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # 划分训练集和验证集
        all_images = list(augmented_annotations.keys())
        random.seed(42)
        random.shuffle(all_images)
        
        split_idx = int(len(all_images) * 0.8)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        print(f"\n📂 划分数据集:")
        print(f"  训练集: {len(train_images)}张")
        print(f"  验证集: {len(val_images)}张")
        
        # 创建类别映射
        classes = data_info['classes']
        class_to_id = {cls: idx for idx, cls in enumerate(classes)}
        
        # 转换为YOLO格式
        print(f"\n🔄 转换为YOLO格式...")
        for split, images in [('train', train_images), ('val', val_images)]:
            for img_path in images:
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                # 复制图片
                img_name = Path(img_path).name
                shutil.copy2(img_path, self.dataset_dir / "images" / split / img_name)
                
                # 创建标签文件
                label_path = self.dataset_dir / "labels" / split / Path(img_name).with_suffix(".txt")
                with open(label_path, 'w') as f:
                    for box in augmented_annotations[img_path]:
                        class_id = class_to_id[box['class']]
                        
                        # 处理两种标注格式
                        if 'x' in box and 'width' in box:
                            # 格式1: x, y, width, height
                            x_center = (box['x'] + box['width'] / 2) / img_width
                            y_center = (box['y'] + box['height'] / 2) / img_height
                            width = box['width'] / img_width
                            height = box['height'] / img_height
                        elif 'x1' in box and 'x2' in box:
                            # 格式2: x1, y1, x2, y2
                            x_center = ((box['x1'] + box['x2']) / 2) / img_width
                            y_center = ((box['y1'] + box['y2']) / 2) / img_height
                            width = (box['x2'] - box['x1']) / img_width
                            height = (box['y2'] - box['y1']) / img_height
                        else:
                            print(f"  ⚠ 未知的标注格式: {box}")
                            continue
                        
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # 创建dataset.yaml
        dataset_yaml = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {idx: cls for idx, cls in enumerate(classes)},
            'nc': len(classes)
        }
        
        yaml_path = self.dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_yaml, f, allow_unicode=True)
        
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            print(f"  ⚠ 无法删除临时目录（文件被占用），请手动删除: {temp_dir}")
        except Exception as e:
            print(f"  ⚠ 清理临时目录时出错: {e}")
        
        print(f"\n✅ 数据集准备完成!")
        print(f"  位置: {self.dataset_dir}")
        print(f"  配置: {yaml_path}")
        
        return True
    
    def train_model(self, epochs=50):
        """训练模型"""
        print(f"\n{'='*60}")
        print(f"训练 {self.page_type} 模型")
        print(f"{'='*60}\n")
        
        # 检查数据集
        yaml_path = self.dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            print(f"❌ 数据集不存在，请先运行 prepare")
            return False
        
        # 检查GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  设备信息:")
        print(f"  使用设备: {device}")
        if device == 'cuda':
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 训练配置
        print(f"\n⚙️  训练配置:")
        print(f"  模型: YOLOv8n")
        print(f"  轮数: {epochs}")
        print(f"  批次: 16")
        print(f"  图片大小: 640")
        print(f"  早停: patience=50")
        print(f"  Workers: 0 (Windows)")
        
        # 加载模型
        model = YOLO('yolov8n.pt')
        
        # 开始训练
        print(f"\n🚀 开始训练...")
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=640,
            batch=16,
            device=0 if device == 'cuda' else 'cpu',
            workers=0,  # Windows系统使用0
            cache=True,
            amp=True,
            patience=50,
            save=True,
            project='runs/detect/yolo_runs',
            name=self.model_name,
            verbose=True
        )
        
        print(f"\n✅ 训练完成!")
        return True
    
    def test_model(self):
        """测试模型"""
        print(f"\n{'='*60}")
        print(f"测试 {self.page_type} 模型")
        print(f"{'='*60}\n")
        
        # 查找最新的模型
        model_pattern = f"runs/detect/**/yolo_runs/{self.model_name}*/weights/best.pt"
        model_files = glob.glob(model_pattern, recursive=True)
        
        if not model_files:
            print(f"❌ 找不到训练好的模型")
            return None
        
        # 使用最新的模型
        model_path = sorted(model_files, key=lambda x: Path(x).stat().st_mtime)[-1]
        print(f"📦 加载模型: {model_path}")
        
        model = YOLO(model_path)
        
        # 测试数据集
        yaml_path = self.dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            print(f"❌ 数据集不存在")
            return None
        
        print(f"\n🧪 开始测试...")
        results = model.val(data=str(yaml_path))
        
        # 统计数据集信息
        train_images = list((self.dataset_dir / "images" / "train").glob("*.png"))
        train_images += list((self.dataset_dir / "images" / "train").glob("*.jpg"))
        val_images = list((self.dataset_dir / "images" / "val").glob("*.png"))
        val_images += list((self.dataset_dir / "images" / "val").glob("*.jpg"))
        all_images = train_images + val_images
        
        print(f"\n📊 测试结果:")
        print(f"  mAP50: {results.box.map50:.3f}")
        print(f"  mAP50-95: {results.box.map:.3f}")
        print(f"  Precision: {results.box.mp:.3f}")
        print(f"  Recall: {results.box.mr:.3f}")
        print(f"\n📦 数据集统计:")
        print(f"  训练集: {len(train_images)}张")
        print(f"  验证集: {len(val_images)}张")
        print(f"  总计: {len(all_images)}张（包括增强图）")
        
        # 读取类别信息
        with open(yaml_path, 'r', encoding='utf-8') as f:
            dataset_info = yaml.safe_load(f)
        classes = list(dataset_info['names'].values())
        num_classes = len(classes)
        
        # 测试所有图片并统计检测结果
        print(f"\n🔍 测试所有图片的检测情况...")
        all_detection_stats = []
        fully_detected_count = 0
        
        for i, img_path in enumerate(all_images, 1):
            if i % 100 == 0:
                print(f"  进度: {i}/{len(all_images)}...")
            
            # 预测
            pred_results = model.predict(str(img_path), conf=0.25, save=False, verbose=False)
            
            # 统计检测结果
            detections = pred_results[0].boxes
            detected_classes = {}
            for cls_id in detections.cls:
                cls_name = classes[int(cls_id)]
                detected_classes[cls_name] = detected_classes.get(cls_name, 0) + 1
            
            # 检查是否所有类别都被检测到
            missing_classes = [cls for cls in classes if cls not in detected_classes]
            
            all_detection_stats.append({
                'image': img_path.name,
                'detected': detected_classes,
                'missing': missing_classes,
                'total_detections': len(detections)
            })
            
            if not missing_classes:
                fully_detected_count += 1
        
        # 统计总体检测情况
        total_images = len(all_detection_stats)
        partially_detected = total_images - fully_detected_count
        
        print(f"\n  📊 全部图片检测统计:")
        print(f"    测试图片: {total_images}张")
        print(f"    全部检测到: {fully_detected_count}张 ({fully_detected_count/total_images*100:.1f}%)")
        if partially_detected > 0:
            print(f"    有遗漏: {partially_detected}张 ({partially_detected/total_images*100:.1f}%)")
        
        # 随机抽取10张图片进行可视化测试
        print(f"\n📸 随机抽取10张图片生成可视化截图...")
        sample_count = min(10, len(all_images))
        sample_images = random.sample(all_images, sample_count)
        
        # 创建测试结果目录
        test_results_dir = Path(f"test_results/{self.page_type}_test_samples")
        
        # 删除旧的截图
        if test_results_dir.exists():
            print(f"  🗑️  删除旧的测试截图...")
            shutil.rmtree(test_results_dir)
        
        # 创建新目录
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # 对抽取的10张图片生成可视化截图
        print(f"  📷 生成可视化截图...")
        sample_detection_stats = []
        for i, img_path in enumerate(sample_images, 1):
            # 预测
            pred_results = model.predict(str(img_path), conf=0.25, save=False, verbose=False)
            
            # 统计检测结果
            detections = pred_results[0].boxes
            detected_classes = {}
            for cls_id in detections.cls:
                cls_name = classes[int(cls_id)]
                detected_classes[cls_name] = detected_classes.get(cls_name, 0) + 1
            
            # 检查是否所有类别都被检测到
            missing_classes = [cls for cls in classes if cls not in detected_classes]
            
            sample_detection_stats.append({
                'image': img_path.name,
                'detected': detected_classes,
                'missing': missing_classes,
                'total_detections': len(detections)
            })
            
            # 绘制结果
            annotated = pred_results[0].plot()
            
            # 保存
            output_file = test_results_dir / f"test_{i:02d}_{img_path.name}"
            cv2.imwrite(str(output_file), annotated)
            
            # 打印检测结果
            status = "✓" if not missing_classes else "⚠"
            print(f"    {status} 已保存测试截图 {i}/{sample_count}: {output_file.name}")
            print(f"       检测到: {dict(detected_classes)}")
            if missing_classes:
                print(f"       遗漏: {missing_classes}")
        
        # 统计抽样截图的检测情况
        sample_fully_detected = sum(1 for stat in sample_detection_stats if not stat['missing'])
        sample_partially_detected = sample_count - sample_fully_detected
        
        print(f"\n  📊 抽样截图检测统计:")
        print(f"    测试图片: {sample_count}张")
        print(f"    全部检测到: {sample_fully_detected}张 ({sample_fully_detected/sample_count*100:.1f}%)")
        if sample_partially_detected > 0:
            print(f"    有遗漏: {sample_partially_detected}张 ({sample_partially_detected/sample_count*100:.1f}%)")
        
        print(f"\n  ✅ 测试截图已保存到: {test_results_dir}")
        
        # 打开截图文件夹
        print(f"\n  📂 打开截图文件夹...")
        try:
            if platform.system() == 'Windows':
                os.startfile(str(test_results_dir.absolute()))
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(test_results_dir.absolute())])
            else:  # Linux
                subprocess.run(['xdg-open', str(test_results_dir.absolute())])
            print(f"  ✓ 已打开文件夹")
        except Exception as e:
            print(f"  ⚠ 无法自动打开文件夹: {e}")
            print(f"  请手动打开: {test_results_dir.absolute()}")
        
        print(f"\n✅ 测试完成!")
        
        # 返回测试结果字典
        return {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'train_count': len(train_images),
            'val_count': len(val_images),
            'total_count': len(all_images),
            'test_samples_dir': str(test_results_dir),
            'all_images_stats': {
                'total_images': total_images,
                'fully_detected': fully_detected_count,
                'partially_detected': partially_detected,
                'detection_rate': fully_detected_count / total_images * 100
            },
            'sample_images_stats': {
                'total_images': sample_count,
                'fully_detected': sample_fully_detected,
                'partially_detected': sample_partially_detected,
                'detection_rate': sample_fully_detected / sample_count * 100
            }
        }
    
    def cleanup_after_training(self):
        """训练后整理：移动原始标注图、删除增强数据、注册模型"""
        print(f"\n{'='*60}")
        print(f"训练后整理 - {self.page_type}")
        print(f"{'='*60}\n")
        
        cleanup_results = {
            'original_saved': False,
            'dataset_deleted': False,
            'model_registered': False,
            'report_generated': False
        }
        
        # 1. 移动原始标注图到 原始标注图/ 目录
        print(f"📦 [1/4] 保存原始标注图...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_dir = Path(f"原始标注图/{self.page_type}_{timestamp}")
        original_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制原始图片和标签
        annotation_file = self.source_dir / "annotations.json"
        copied_count = 0
        
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # 创建子目录
            (original_dir / "images").mkdir(exist_ok=True)
            (original_dir / "labels").mkdir(exist_ok=True)
            
            # 获取类别映射
            classes = sorted(set(box['class'] for boxes in annotations.values() if boxes for box in boxes))
            class_to_id = {cls: idx for idx, cls in enumerate(classes)}
            
            # 复制原始图片和生成YOLO标签
            for img_path_str, boxes in annotations.items():
                if not boxes:
                    continue
                
                img_path = Path(img_path_str)
                if not img_path.exists():
                    continue
                
                # 复制图片
                shutil.copy2(img_path, original_dir / "images" / img_path.name)
                
                # 生成YOLO标签
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                label_path = original_dir / "labels" / img_path.with_suffix(".txt").name
                with open(label_path, 'w') as f:
                    for box in boxes:
                        class_id = class_to_id[box['class']]
                        
                        # 处理两种标注格式
                        if 'x' in box and 'width' in box:
                            # 格式1: x, y, width, height
                            x_center = (box['x'] + box['width'] / 2) / img_width
                            y_center = (box['y'] + box['height'] / 2) / img_height
                            width = box['width'] / img_width
                            height = box['height'] / img_height
                        elif 'x1' in box and 'x2' in box:
                            # 格式2: x1, y1, x2, y2
                            x_center = ((box['x1'] + box['x2']) / 2) / img_width
                            y_center = ((box['y1'] + box['y2']) / 2) / img_height
                            width = (box['x2'] - box['x1']) / img_width
                            height = (box['y2'] - box['y1']) / img_height
                        else:
                            continue
                        
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                
                copied_count += 1
            
            # 复制annotations.json
            shutil.copy2(annotation_file, original_dir / "annotations.json")
            
            print(f"  ✓ 已保存 {copied_count} 张原始标注图到: {original_dir}")
            cleanup_results['original_saved'] = True
        else:
            print(f"  ⚠ 未找到标注文件，跳过")
        
        # 2. 删除YOLO数据集（增强数据）
        print(f"\n🗑️  [2/4] 删除YOLO数据集（增强数据）...")
        if self.dataset_dir.exists():
            try:
                shutil.rmtree(self.dataset_dir)
                print(f"  ✓ 已删除: {self.dataset_dir}")
                cleanup_results['dataset_deleted'] = True
            except Exception as e:
                print(f"  ✗ 删除失败: {e}")
        else:
            print(f"  ⚠ 数据集不存在，跳过")
            cleanup_results['dataset_deleted'] = True  # 不存在也算成功
        
        # 3. 注册模型到 yolo_model_registry.json
        print(f"\n📝 [3/4] 注册模型...")
        registry_file = Path("yolo_model_registry.json")
        
        # 读取现有注册表
        if registry_file.exists():
            with open(registry_file, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        else:
            registry = {
                "models": {},
                "usage": {
                    "description": "YOLO模型注册表，记录所有训练完成的模型信息",
                    "how_to_use": {
                        "load_model": "from ultralytics import YOLO; model = YOLO(registry['models']['homepage']['model_path'])",
                        "get_classes": "classes = registry['models']['homepage']['classes']",
                        "check_performance": "performance = registry['models']['homepage']['performance']"
                    }
                },
                "version": "1.0"
            }
        
        # 查找最新的模型
        model_pattern = f"runs/detect/**/yolo_runs/{self.model_name}*/weights/best.pt"
        model_files = glob.glob(model_pattern, recursive=True)
        
        if model_files:
            model_path = sorted(model_files, key=lambda x: Path(x).stat().st_mtime)[-1]
            
            # 从原始标注中获取类别
            if annotation_file.exists():
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                classes = sorted(set(box['class'] for boxes in annotations.values() if boxes for box in boxes))
                original_count = sum(1 for v in annotations.values() if v)
            else:
                classes = []
                original_count = 0
            
            # 测试模型获取性能指标
            model = YOLO(model_path)
            
            # 尝试从已存在的数据集测试，如果不存在则跳过性能测试
            yaml_path_for_test = self.dataset_dir / "dataset.yaml"
            
            if yaml_path_for_test.exists():
                try:
                    results = model.val(data=str(yaml_path_for_test))
                    performance = {
                        "mAP50": float(results.box.map50),
                        "mAP50-95": float(results.box.map),
                        "precision": float(results.box.mp),
                        "recall": float(results.box.mr)
                    }
                except Exception as e:
                    print(f"  ⚠ 无法测试模型性能: {e}")
                    performance = {
                        "mAP50": 0.0,
                        "mAP50-95": 0.0,
                        "precision": 0.0,
                        "recall": 0.0
                    }
            else:
                print(f"  ⚠ 数据集不存在，跳过性能测试")
                performance = {
                    "mAP50": 0.0,
                    "mAP50-95": 0.0,
                    "precision": 0.0,
                    "recall": 0.0
                }
            
            # 添加到注册表
            model_key = self.page_type.lower().replace(" ", "_")
            registry["models"][model_key] = {
                "name": f"{self.page_type}检测模型",
                "page_type": self.page_type,
                "model_path": model_path,
                "classes": classes,
                "num_classes": len(classes),
                "performance": performance,
                "training_date": datetime.now().strftime("%Y-%m-%d"),
                "dataset_size": {
                    "original": original_count,
                    "augmented": 0,  # 已删除
                    "train": 0,
                    "val": 0
                },
                "original_data_path": str(original_dir),
                "notes": f"使用yolo_train_manager.py训练，默认50轮"
            }
            
            # 更新last_updated
            registry["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            
            # 保存注册表
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 已注册模型: {model_key}")
            print(f"  模型路径: {model_path}")
            print(f"  类别: {classes}")
            print(f"  性能: mAP50={performance['mAP50']:.3f}, Precision={performance['precision']:.3f}, Recall={performance['recall']:.3f}")
            cleanup_results['model_registered'] = True
        else:
            print(f"  ✗ 未找到训练好的模型，跳过注册")
        
        # 4. 生成整理报告
        print(f"\n📄 [4/4] 生成整理报告...")
        report_path = Path(f"training_reports/{self.page_type}_{timestamp}.txt")
        report_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"{self.page_type} 训练完成报告\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("1. 原始标注图\n")
                f.write(f"   位置: {original_dir}\n")
                f.write(f"   数量: {copied_count if 'copied_count' in locals() else 0} 张\n\n")
                
                f.write("2. 模型信息\n")
                if model_files:
                    f.write(f"   路径: {model_path}\n")
                    f.write(f"   类别: {', '.join(classes)}\n")
                    f.write(f"   性能:\n")
                    f.write(f"     mAP50: {performance['mAP50']:.3f}\n")
                    f.write(f"     mAP50-95: {performance['mAP50-95']:.3f}\n")
                    f.write(f"     Precision: {performance['precision']:.3f}\n")
                    f.write(f"     Recall: {performance['recall']:.3f}\n\n")
                
                f.write("3. 数据清理\n")
                f.write(f"   ✓ 已删除YOLO数据集: {self.dataset_dir}\n")
                f.write(f"   ✓ 已保存原始标注图: {original_dir}\n")
                f.write(f"   ✓ 已注册模型到: yolo_model_registry.json\n\n")
                
                f.write("=" * 60 + "\n")
            
            print(f"  ✓ 报告已保存: {report_path}")
            cleanup_results['report_generated'] = True
        except Exception as e:
            print(f"  ✗ 生成报告失败: {e}")
        
        # 验证整理是否完成
        print(f"\n{'='*60}")
        print(f"🔍 验证整理结果")
        print(f"{'='*60}\n")
        
        all_success = True
        
        # 验证1: 原始标注图是否保存
        if cleanup_results['original_saved']:
            images_count = len(list((original_dir / "images").glob("*")))
            labels_count = len(list((original_dir / "labels").glob("*.txt")))
            if images_count > 0 and labels_count > 0:
                print(f"  ✓ 原始标注图已保存: {images_count}张图片, {labels_count}个标签")
            else:
                print(f"  ✗ 原始标注图保存不完整: {images_count}张图片, {labels_count}个标签")
                all_success = False
        else:
            print(f"  ✗ 原始标注图未保存")
            all_success = False
        
        # 验证2: YOLO数据集是否删除
        if cleanup_results['dataset_deleted']:
            if not self.dataset_dir.exists():
                print(f"  ✓ YOLO数据集已删除")
            else:
                print(f"  ✗ YOLO数据集仍然存在: {self.dataset_dir}")
                all_success = False
        else:
            print(f"  ✗ YOLO数据集未删除")
            all_success = False
        
        # 验证3: 模型是否注册
        if cleanup_results['model_registered']:
            if registry_file.exists():
                with open(registry_file, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                model_key = self.page_type.lower().replace(" ", "_")
                if model_key in registry.get("models", {}):
                    print(f"  ✓ 模型已注册: {model_key}")
                else:
                    print(f"  ✗ 模型未在注册表中找到: {model_key}")
                    all_success = False
            else:
                print(f"  ✗ 模型注册表不存在")
                all_success = False
        else:
            print(f"  ✗ 模型未注册")
            all_success = False
        
        # 验证4: 报告是否生成
        if cleanup_results['report_generated']:
            if report_path.exists():
                print(f"  ✓ 整理报告已生成")
            else:
                print(f"  ✗ 整理报告不存在")
                all_success = False
        else:
            print(f"  ✗ 整理报告未生成")
            all_success = False
        
        print(f"\n{'='*60}")
        if all_success:
            print(f"✅ 整理完成！所有操作均成功")
        else:
            print(f"⚠️  整理完成，但部分操作失败，请检查上述错误")
        print(f"{'='*60}")
        print(f"原始标注图: {original_dir}")
        print(f"模型注册表: yolo_model_registry.json")
        print(f"整理报告: {report_path}")
        
        return all_success


def main():
    parser = argparse.ArgumentParser(description='YOLO训练管理器')
    parser.add_argument('action', choices=['prepare', 'train', 'test', 'cleanup', 'all'], 
                       help='操作: prepare(准备数据), train(训练), test(测试), cleanup(整理), all(全部)')
    parser.add_argument('page_type', help='页面类型，如"分类页"、"登录页"')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数(默认50)')
    parser.add_argument('--augment', type=int, default=None, help='数据增强倍数(默认自动)')
    
    args = parser.parse_args()
    
    manager = YOLOTrainManager(args.page_type)
    
    if args.action == 'prepare':
        manager.prepare_dataset(args.augment)
    elif args.action == 'train':
        manager.train_model(args.epochs)
    elif args.action == 'test':
        manager.test_model()
    elif args.action == 'cleanup':
        manager.cleanup_after_training()
    elif args.action == 'all':
        if manager.prepare_dataset(args.augment):
            if manager.train_model(args.epochs):
                test_results = manager.test_model()
                if test_results:
                    # 汇报测试结果，等待用户批准
                    print(f"\n{'='*60}")
                    print(f"⚠️  等待用户批准")
                    print(f"{'='*60}")
                    print(f"\n测试已完成，性能指标如下：")
                    print(f"  mAP50: {test_results['mAP50']:.3f}")
                    print(f"  mAP50-95: {test_results['mAP50-95']:.3f}")
                    print(f"  Precision: {test_results['precision']:.3f}")
                    print(f"  Recall: {test_results['recall']:.3f}")
                    print(f"\n测试了所有图片（包括增强图）：")
                    print(f"  训练集: {test_results['train_count']}张")
                    print(f"  验证集: {test_results['val_count']}张")
                    print(f"  总计: {test_results['total_count']}张")
                    print(f"\n所有图片检测结果：")
                    print(f"  测试图片: {test_results['all_images_stats']['total_images']}张")
                    print(f"  全部检测到: {test_results['all_images_stats']['fully_detected']}张 ({test_results['all_images_stats']['detection_rate']:.1f}%)")
                    if test_results['all_images_stats']['partially_detected'] > 0:
                        print(f"  有遗漏: {test_results['all_images_stats']['partially_detected']}张 ({100-test_results['all_images_stats']['detection_rate']:.1f}%)")
                    print(f"\n随机抽取的10张测试截图检测结果：")
                    print(f"  测试图片: {test_results['sample_images_stats']['total_images']}张")
                    print(f"  全部检测到: {test_results['sample_images_stats']['fully_detected']}张 ({test_results['sample_images_stats']['detection_rate']:.1f}%)")
                    if test_results['sample_images_stats']['partially_detected'] > 0:
                        print(f"  有遗漏: {test_results['sample_images_stats']['partially_detected']}张")
                    print(f"  位置: {test_results['test_samples_dir']}")
                    print(f"  请查看截图以判断模型是否正确匹配")
                    print(f"\n如需执行最终整理（删除YOLO数据集），请运行：")
                    print(f"  python yolo_train_manager.py cleanup {args.page_type}")
                    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
