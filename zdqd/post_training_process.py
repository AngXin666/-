"""
训练后处理脚本
按照编程规则执行：
1. 测试所有训练图片
2. 随机抽取10张图片进行可视化验证
3. 生成训练报告
4. 移动数据到完成目录
"""
import glob
import random
import shutil
import cv2
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def test_all_images(model, dataset_path):
    """测试数据集中的所有图片"""
    print("正在测试所有训练图片...")
    
    # 获取所有图片
    images = []
    for split in ['train', 'val']:
        img_dir = Path(dataset_path) / "images" / split
        if img_dir.exists():
            images.extend(list(img_dir.glob("*.png")))
            images.extend(list(img_dir.glob("*.jpg")))
    
    print(f"找到 {len(images)} 张图片")
    
    results = []
    for i, img_path in enumerate(images, 1):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(images)}")
        
        try:
            result = model.predict(str(img_path), conf=0.01, verbose=False)
            results.append({
                'path': str(img_path),
                'detections': len(result[0].boxes),
                'confidence': result[0].boxes.conf.tolist() if len(result[0].boxes) > 0 else []
            })
        except Exception as e:
            print(f"  警告: 测试 {img_path.name} 失败: {e}")
            results.append({
                'path': str(img_path),
                'detections': 0,
                'confidence': [],
                'error': str(e)
            })
    
    print(f"✅ 测试完成，共 {len(results)} 张图片")
    return results

def random_sample_screenshots(dataset_path, output_dir, num_samples=10):
    """随机抽取10张图片进行可视化验证"""
    print(f"\n随机抽取 {num_samples} 张图片...")
    
    # 获取所有图片
    images = []
    for split in ['train', 'val']:
        img_dir = Path(dataset_path) / "images" / split
        if img_dir.exists():
            images.extend(list(img_dir.glob("*.png")))
            images.extend(list(img_dir.glob("*.jpg")))
    
    # 随机抽取
    samples = random.sample(images, min(num_samples, len(images)))
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 复制样本图片
    for i, img_path in enumerate(samples, 1):
        dest = output_path / f"sample_{i}_{img_path.name}"
        shutil.copy2(img_path, dest)
        print(f"  样本 {i}: {img_path.name}")
    
    print(f"✅ 已保存 {len(samples)} 张样本图片到: {output_path}")
    return samples

def visualize_test_results(model, sample_images, output_dir):
    """对抽样图片进行预测并保存可视化结果"""
    print(f"\n生成可视化测试结果...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, img_path in enumerate(sample_images, 1):
        try:
            # 预测
            results = model.predict(str(img_path), conf=0.01, save=False, verbose=False)
            
            # 绘制结果
            annotated = results[0].plot()
            
            # 保存
            output_file = output_path / f"result_{i}_{img_path.name}"
            cv2.imwrite(str(output_file), annotated)
            print(f"  结果 {i}: {output_file.name}")
        except Exception as e:
            print(f"  警告: 处理 {img_path.name} 失败: {e}")
    
    print(f"✅ 可视化结果已保存到: {output_path}")

def generate_training_report(model, test_results, output_dir, model_name):
    """生成训练报告"""
    print(f"\n生成训练报告...")
    
    report_path = Path(output_dir) / "training_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{model_name} 训练完成报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 基本信息
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型名称: {model_name}\n")
        if hasattr(model, 'ckpt_path'):
            f.write(f"模型路径: {model.ckpt_path}\n")
        f.write("\n")
        
        # 测试统计
        total_images = len(test_results)
        detected_images = sum(1 for r in test_results if r['detections'] > 0)
        error_images = sum(1 for r in test_results if 'error' in r)
        
        # 计算平均置信度
        all_confidences = []
        for r in test_results:
            if r['confidence']:
                all_confidences.extend(r['confidence'])
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        f.write("测试统计:\n")
        f.write(f"  总图片数: {total_images}\n")
        f.write(f"  检测到目标的图片数: {detected_images}\n")
        f.write(f"  检测率: {detected_images/total_images*100:.2f}%\n")
        f.write(f"  平均置信度: {avg_confidence:.4f}\n")
        if error_images > 0:
            f.write(f"  测试失败的图片数: {error_images}\n")
        f.write("\n")
        
        # 详细结果（显示前30个）
        f.write("详细测试结果 (前30个):\n")
        for i, result in enumerate(test_results[:30], 1):
            path = Path(result['path'])
            f.write(f"  {i}. {path.name}\n")
            f.write(f"     检测数: {result['detections']}\n")
            if result['confidence']:
                conf_str = ', '.join([f"{c:.3f}" for c in result['confidence'][:5]])
                if len(result['confidence']) > 5:
                    conf_str += f" ... (共{len(result['confidence'])}个)"
                f.write(f"     置信度: [{conf_str}]\n")
            if 'error' in result:
                f.write(f"     错误: {result['error']}\n")
        
        if len(test_results) > 30:
            f.write(f"  ... 还有 {len(test_results) - 30} 个结果\n")
    
    print(f"✅ 训练报告已保存: {report_path}")
    return report_path

def setup_completed_directory(base_dir="training_data_completed"):
    """创建已完成训练数据的目录结构"""
    completed_dir = Path(base_dir)
    completed_dir.mkdir(exist_ok=True)
    return completed_dir

def move_completed_data(source_dataset, completed_dir, model_name):
    """移动已完成训练的数据到统一文件夹"""
    print(f"\n移动训练数据到完成目录...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = Path(completed_dir) / f"{model_name}_{timestamp}"
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dataset)
    
    # 移动图片
    images_src = source_path / "images"
    if images_src.exists():
        images_dst = target_dir / "images"
        shutil.copytree(images_src, images_dst)
        print(f"  ✅ 已复制图片: {images_src} -> {images_dst}")
    
    # 移动标签
    labels_src = source_path / "labels"
    if labels_src.exists():
        labels_dst = target_dir / "labels"
        shutil.copytree(labels_src, labels_dst)
        print(f"  ✅ 已复制标签: {labels_src} -> {labels_dst}")
    
    # 复制配置文件
    yaml_file = source_path / "dataset.yaml"
    if yaml_file.exists():
        shutil.copy2(yaml_file, target_dir / "dataset.yaml")
        print(f"  ✅ 已复制配置: dataset.yaml")
    
    print(f"✅ 数据已移动到: {target_dir}")
    return target_dir

def post_training_workflow(model_path, dataset_path, model_name):
    """训练后完整工作流程"""
    print("=" * 60)
    print("开始训练后处理流程")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n[1/6] 加载训练好的模型...")
    model = YOLO(model_path)
    print(f"✅ 模型已加载: {model_path}")
    
    # 2. 测试所有图片
    print("\n[2/6] 测试所有训练图片...")
    test_results = test_all_images(model, dataset_path)
    
    # 3. 随机抽样10张图片
    print("\n[3/6] 随机抽取10张图片...")
    samples_dir = f"test_results/{model_name}_samples"
    sample_images = random_sample_screenshots(dataset_path, samples_dir, num_samples=10)
    
    # 4. 可视化测试结果
    print("\n[4/6] 生成可视化测试结果...")
    results_dir = f"test_results/{model_name}_results"
    visualize_test_results(model, sample_images, results_dir)
    
    # 5. 生成训练报告
    print("\n[5/6] 生成训练报告...")
    report_path = generate_training_report(model, test_results, results_dir, model_name)
    
    # 6. 移动数据到完成目录
    print("\n[6/6] 移动训练数据到完成目录...")
    completed_dir = setup_completed_directory()
    target_dir = move_completed_data(dataset_path, completed_dir, model_name)
    
    # 复制测试结果到完成目录
    print("\n复制测试结果到完成目录...")
    shutil.copytree(samples_dir, target_dir / "samples")
    print(f"  ✅ 已复制样本: {samples_dir} -> {target_dir / 'samples'}")
    
    shutil.copytree(results_dir, target_dir / "test_results")
    print(f"  ✅ 已复制测试结果: {results_dir} -> {target_dir / 'test_results'}")
    
    print("\n" + "=" * 60)
    print("训练后处理完成！")
    print("=" * 60)
    print(f"数据已移动到: {target_dir}")
    print(f"测试样本: {target_dir / 'samples'}")
    print(f"测试结果: {target_dir / 'test_results'}")
    print(f"训练报告: {target_dir / 'test_results' / 'training_report.txt'}")
    print("=" * 60)
    
    return target_dir

if __name__ == '__main__':
    # 配置参数
    model_path = 'yolo_runs/transfer_detector/weights/best.pt'
    dataset_path = 'yolo_dataset_transfer'
    model_name = 'transfer_detector'
    
    # 执行训练后处理
    post_training_workflow(model_path, dataset_path, model_name)
