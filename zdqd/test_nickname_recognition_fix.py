"""
测试昵称识别修复效果
Test Nickname Recognition Fix

本测试脚本用于验证昵称识别修复的效果，包括：
1. 准确率测试 - 使用真实截图测试昵称识别准确率
2. 性能测试 - 测量昵称提取的执行时间
3. 修复前后对比 - 对比修复前后的识别效果（如果可用）

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import asyncio
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("❌ PIL未安装")
    sys.exit(1)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("❌ YOLO未安装")
    sys.exit(1)

# 导入项目模块
try:
    from src.ocr_image_processor import enhance_for_ocr
    from src.ocr_thread_pool import get_ocr_pool
    from src.profile_reader import ProfileReader
    HAS_OCR = True
except ImportError as e:
    HAS_OCR = False
    print(f"❌ 项目模块导入失败: {e}")
    sys.exit(1)


class NicknameRecognitionTester:
    """昵称识别测试器"""
    
    def __init__(self, model_path: str, test_data_dir: str):
        """初始化测试器
        
        Args:
            model_path: YOLO模型路径
            test_data_dir: 测试数据目录
        """
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.model = None
        self.ocr_pool = None
        self.test_images = []
        
    def initialize(self):
        """初始化模型和OCR"""
        print("\n[1] 初始化测试环境...")
        
        # 加载YOLO模型
        if not os.path.exists(self.model_path):
            print(f"❌ 模型文件不存在: {self.model_path}")
            return False
        
        print(f"  加载YOLO模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("  ✓ YOLO模型已加载")
        
        # 初始化OCR线程池
        print("  初始化OCR线程池...")
        self.ocr_pool = get_ocr_pool()
        print("  ✓ OCR线程池已初始化")
        
        # 查找测试图片
        print(f"  查找测试图片: {self.test_data_dir}")
        if not os.path.exists(self.test_data_dir):
            print(f"  ❌ 测试数据目录不存在: {self.test_data_dir}")
            return False
        
        self.test_images = [
            os.path.join(self.test_data_dir, f)
            for f in os.listdir(self.test_data_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not self.test_images:
            print(f"  ❌ 未找到测试图片")
            return False
        
        print(f"  ✓ 找到 {len(self.test_images)} 张测试图片")
        return True
    
    async def extract_nickname_from_image(
        self, 
        image_path: str,
        use_fix: bool = True
    ) -> Dict:
        """从图片中提取昵称
        
        Args:
            image_path: 图片路径
            use_fix: 是否使用修复后的逻辑
            
        Returns:
            dict: 包含识别结果的字典
                - success: bool, 是否成功
                - nickname: str, 识别的昵称
                - confidence: float, YOLO检测置信度
                - ocr_texts: List[str], OCR识别的所有文本
                - execution_time: float, 执行时间（秒）
                - error: str, 错误信息（如果有）
        """
        start_time = time.time()
        result = {
            'success': False,
            'nickname': None,
            'confidence': 0.0,
            'ocr_texts': [],
            'execution_time': 0.0,
            'error': None
        }
        
        try:
            # 加载图片
            image = Image.open(image_path)
            
            # YOLO检测昵称区域
            yolo_results = self.model.predict(image, conf=0.25, verbose=False)
            
            # 查找昵称文字区域
            nickname_found = False
            for r in yolo_results:
                boxes = r.boxes
                
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = r.names[cls]
                    conf = float(box.conf[0])
                    
                    # 只处理昵称文字区域
                    if class_name == '昵称文字':
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        result['confidence'] = conf
                        
                        # 裁剪昵称区域
                        region = image.crop((int(x1), int(y1), int(x2), int(y2)))
                        
                        # OCR识别
                        enhanced_image = enhance_for_ocr(region)
                        ocr_result = await self.ocr_pool.recognize(
                            enhanced_image, 
                            timeout=5.0
                        )
                        
                        if ocr_result and ocr_result.texts:
                            result['ocr_texts'] = ocr_result.texts
                            
                            if use_fix:
                                # 使用修复后的逻辑（通过ProfileReader）
                                from unittest.mock import Mock
                                
                                # 创建临时ProfileReader实例
                                mock_adb = Mock()
                                
                                # 临时修改__init__以避免完整初始化
                                original_init = ProfileReader.__init__
                                
                                def mock_init(self, adb, yolo_detector=None):
                                    self.adb = adb
                                    self._ocr_pool = None
                                    self._cache = None
                                    self._integrated_detector = None
                                    self._yolo_detector = None
                                    self._silent_log = None
                                
                                ProfileReader.__init__ = mock_init
                                reader = ProfileReader(adb=mock_adb)
                                ProfileReader.__init__ = original_init
                                
                                # 使用修复后的提取逻辑
                                nickname = reader._extract_nickname_from_texts(
                                    ocr_result.texts,
                                    ocr_result=ocr_result,
                                    detection_bbox=(int(x1), int(y1), int(x2), int(y2))
                                )
                            else:
                                # 使用简单逻辑（修复前）
                                nickname = self._extract_nickname_simple(
                                    ocr_result.texts
                                )
                            
                            if nickname:
                                result['success'] = True
                                result['nickname'] = nickname
                                nickname_found = True
                                break
                
                if nickname_found:
                    break
            
            if not nickname_found:
                result['error'] = "未检测到昵称区域或OCR识别失败"
        
        except Exception as e:
            result['error'] = str(e)
        
        finally:
            result['execution_time'] = time.time() - start_time
        
        return result
    
    def _extract_nickname_simple(self, texts: List[str]) -> Optional[str]:
        """简单的昵称提取逻辑（修复前）
        
        Args:
            texts: OCR识别的文本列表
            
        Returns:
            str: 提取的昵称，如果没有找到则返回None
        """
        if not texts:
            return None
        
        # 简单逻辑：返回第一个非空文本
        for text in texts:
            text = text.strip()
            if text:
                return text
        
        return None
    
    async def run_accuracy_test(
        self, 
        use_fix: bool = True,
        max_samples: Optional[int] = None
    ) -> Dict:
        """运行准确率测试
        
        Args:
            use_fix: 是否使用修复后的逻辑
            max_samples: 最大测试样本数（None表示全部测试）
            
        Returns:
            dict: 测试结果统计
        """
        print(f"\n[2] 运行准确率测试 ({'修复后' if use_fix else '修复前'})...")
        
        test_images = self.test_images[:max_samples] if max_samples else self.test_images
        
        results = []
        success_count = 0
        fail_count = 0
        
        print(f"  测试样本数: {len(test_images)}")
        print(f"  {'='*70}")
        
        for i, image_path in enumerate(test_images, 1):
            result = await self.extract_nickname_from_image(image_path, use_fix)
            results.append({
                'image': os.path.basename(image_path),
                'result': result
            })
            
            if result['success']:
                success_count += 1
                status = "✓"
            else:
                fail_count += 1
                status = "✗"
            
            # 显示前10个和每10个
            if i <= 10 or i % 10 == 0:
                print(f"  [{i}/{len(test_images)}] {status} {os.path.basename(image_path)}")
                print(f"    昵称: {result['nickname']}")
                print(f"    YOLO置信度: {result['confidence']:.3f}")
                print(f"    执行时间: {result['execution_time']:.3f}s")
                if result['error']:
                    print(f"    错误: {result['error']}")
        
        # 统计结果
        accuracy = success_count / len(test_images) if test_images else 0.0
        
        return {
            'total': len(test_images),
            'success': success_count,
            'fail': fail_count,
            'accuracy': accuracy,
            'results': results
        }
    
    async def run_performance_test(
        self, 
        iterations: int = 100
    ) -> Dict:
        """运行性能测试
        
        Args:
            iterations: 测试迭代次数
            
        Returns:
            dict: 性能测试结果
        """
        print(f"\n[3] 运行性能测试...")
        print(f"  迭代次数: {iterations}")
        
        if not self.test_images:
            print("  ❌ 没有测试图片")
            return {}
        
        # 使用第一张图片进行性能测试
        test_image = self.test_images[0]
        print(f"  测试图片: {os.path.basename(test_image)}")
        
        execution_times = []
        
        for i in range(iterations):
            result = await self.extract_nickname_from_image(test_image, use_fix=True)
            execution_times.append(result['execution_time'])
            
            if (i + 1) % 20 == 0:
                print(f"  进度: {i + 1}/{iterations}")
        
        # 统计性能指标
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        return {
            'iterations': iterations,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'times': execution_times
        }
    
    async def run_comparison_test(
        self,
        max_samples: Optional[int] = 20
    ) -> Dict:
        """运行修复前后对比测试
        
        Args:
            max_samples: 最大测试样本数
            
        Returns:
            dict: 对比测试结果
        """
        print(f"\n[4] 运行修复前后对比测试...")
        
        # 测试修复前
        print("\n  测试修复前逻辑...")
        before_results = await self.run_accuracy_test(use_fix=False, max_samples=max_samples)
        
        # 测试修复后
        print("\n  测试修复后逻辑...")
        after_results = await self.run_accuracy_test(use_fix=True, max_samples=max_samples)
        
        return {
            'before': before_results,
            'after': after_results
        }
    
    def print_summary(self, accuracy_results: Dict, performance_results: Dict = None):
        """打印测试总结
        
        Args:
            accuracy_results: 准确率测试结果
            performance_results: 性能测试结果（可选）
        """
        print(f"\n{'='*70}")
        print("测试总结")
        print(f"{'='*70}")
        
        # 准确率统计
        print(f"\n【准确率统计】")
        print(f"  总样本数: {accuracy_results['total']}")
        print(f"  成功: {accuracy_results['success']} ({accuracy_results['accuracy']:.1%})")
        print(f"  失败: {accuracy_results['fail']} ({accuracy_results['fail']/accuracy_results['total']:.1%})")
        
        # 昵称分布
        nickname_counts = defaultdict(int)
        for item in accuracy_results['results']:
            if item['result']['success']:
                nickname = item['result']['nickname']
                nickname_counts[nickname] += 1
        
        if nickname_counts:
            print(f"\n【昵称分布】")
            for nickname, count in sorted(nickname_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {nickname}: {count}次")
        
        # YOLO置信度统计
        confidences = [
            item['result']['confidence'] 
            for item in accuracy_results['results'] 
            if item['result']['success']
        ]
        
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            
            print(f"\n【YOLO置信度统计】")
            print(f"  平均: {avg_conf:.3f}")
            print(f"  最低: {min_conf:.3f}")
            print(f"  最高: {max_conf:.3f}")
        
        # 性能统计
        if performance_results:
            print(f"\n【性能统计】")
            print(f"  迭代次数: {performance_results['iterations']}")
            print(f"  平均执行时间: {performance_results['avg_time']*1000:.2f}ms")
            print(f"  最快: {performance_results['min_time']*1000:.2f}ms")
            print(f"  最慢: {performance_results['max_time']*1000:.2f}ms")
        
        # 失败案例
        failed_cases = [
            item for item in accuracy_results['results']
            if not item['result']['success']
        ]
        
        if failed_cases:
            print(f"\n【失败案例】（前10个）")
            for item in failed_cases[:10]:
                print(f"  {item['image']}")
                if item['result']['error']:
                    print(f"    错误: {item['result']['error']}")
                if item['result']['ocr_texts']:
                    print(f"    OCR文本: {item['result']['ocr_texts']}")
        
        print(f"\n{'='*70}")
    
    def print_comparison_summary(self, comparison_results: Dict):
        """打印对比测试总结
        
        Args:
            comparison_results: 对比测试结果
        """
        print(f"\n{'='*70}")
        print("修复前后对比总结")
        print(f"{'='*70}")
        
        before = comparison_results['before']
        after = comparison_results['after']
        
        print(f"\n【准确率对比】")
        print(f"  修复前: {before['accuracy']:.1%} ({before['success']}/{before['total']})")
        print(f"  修复后: {after['accuracy']:.1%} ({after['success']}/{after['total']})")
        
        improvement = after['accuracy'] - before['accuracy']
        print(f"  改进: {improvement:+.1%}")
        
        # 对比具体案例
        print(f"\n【案例对比】")
        for i, (before_item, after_item) in enumerate(zip(before['results'], after['results'])):
            before_result = before_item['result']
            after_result = after_item['result']
            
            # 只显示结果不同的案例
            if before_result['nickname'] != after_result['nickname']:
                print(f"\n  案例 {i+1}: {before_item['image']}")
                print(f"    修复前: {before_result['nickname']} ({'成功' if before_result['success'] else '失败'})")
                print(f"    修复后: {after_result['nickname']} ({'成功' if after_result['success'] else '失败'})")
                if after_result['ocr_texts']:
                    print(f"    OCR文本: {after_result['ocr_texts']}")
        
        print(f"\n{'='*70}")


async def main():
    """主函数"""
    print("=" * 70)
    print("昵称识别修复测试")
    print("=" * 70)
    
    # 配置
    model_path = "runs/detect/runs/detect/profile_detailed_detector/weights/best.pt"
    test_data_dir = "training_data/新已登陆页"
    
    # 创建测试器
    tester = NicknameRecognitionTester(model_path, test_data_dir)
    
    # 初始化
    if not tester.initialize():
        print("\n❌ 初始化失败")
        return
    
    # 运行准确率测试
    accuracy_results = await tester.run_accuracy_test(use_fix=True, max_samples=None)
    
    # 运行性能测试（可选，较耗时）
    # performance_results = await tester.run_performance_test(iterations=100)
    performance_results = None
    
    # 打印总结
    tester.print_summary(accuracy_results, performance_results)
    
    # 运行对比测试（可选）
    print("\n" + "=" * 70)
    user_input = input("是否运行修复前后对比测试？(y/n): ")
    if user_input.lower() == 'y':
        comparison_results = await tester.run_comparison_test(max_samples=20)
        tester.print_comparison_summary(comparison_results)
    
    print("\n✅ 测试完成！")


if __name__ == '__main__':
    asyncio.run(main())
