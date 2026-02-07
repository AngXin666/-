"""
页面元素检测示例
结合页面分类和YOLO目标检测
"""
from ultralytics import YOLO
from page_element_mapping import get_element_classes_for_page, get_element_names_for_page
import cv2


class PageElementDetector:
    """页面元素检测器"""
    
    def __init__(self, page_classifier_path, yolo_model_path):
        """
        初始化检测器
        
        Args:
            page_classifier_path: 页面分类模型路径
            yolo_model_path: YOLO目标检测模型路径
        """
        # 加载页面分类模型
        try:
            from train_page_classifier import PageClassifier
            self.page_classifier = PageClassifier()
            self.page_classifier.load_model(page_classifier_path)
            print(f"✓ 页面分类模型加载成功: {page_classifier_path}")
        except Exception as e:
            print(f"✗ 页面分类模型加载失败: {e}")
            self.page_classifier = None
        
        # 加载YOLO模型
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print(f"✓ YOLO模型加载成功: {yolo_model_path}")
        except Exception as e:
            print(f"✗ YOLO模型加载失败: {e}")
            self.yolo_model = None
        
        # 加载页面映射（如果存在）
        self.page_mapping = {}
        mapping_file = Path("yolo_dataset/page_mapping.json")
        if mapping_file.exists():
            import json
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.page_mapping = json.load(f)
            print(f"✓ 页面映射加载成功: {len(self.page_mapping)} 条记录")
    
    def detect(self, image_path, conf_threshold=0.25):
        """
        检测页面元素
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            dict: 检测结果
        """
        if not self.page_classifier or not self.yolo_model:
            print("模型未加载")
            return None
        
        # 1. 识别页面类型
        page_type, page_conf = self.page_classifier.predict(image_path)
        print(f"\n页面类型: {page_type} (置信度: {page_conf:.2%})")
        
        # 2. 获取该页面应该检测的元素
        expected_elements = get_element_names_for_page(page_type)
        element_ids = get_element_classes_for_page(page_type)
        
        print(f"应检测元素: {expected_elements}")
        
        # 3. 使用YOLO检测元素（只检测该页面的元素）
        if element_ids:
            results = self.yolo_model.predict(
                source=image_path,
                conf=conf_threshold,
                classes=element_ids,  # 只检测指定类别
                verbose=False
            )
        else:
            print("该页面无需检测元素")
            return {
                'page_type': page_type,
                'page_confidence': page_conf,
                'elements': []
            }
        
        # 4. 解析检测结果
        detected_elements = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                class_name = result.names[cls]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detected_elements.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'center': [center_x, center_y]
                })
                
                print(f"  ✓ {class_name}: ({center_x:.0f}, {center_y:.0f}), 置信度: {conf:.2%}")
        
        return {
            'page_type': page_type,
            'page_confidence': page_conf,
            'expected_elements': expected_elements,
            'detected_elements': detected_elements
        }
    
    def detect_and_click(self, image_path, element_name, adb_controller=None):
        """
        检测元素并点击
        
        Args:
            image_path: 图片路径
            element_name: 要点击的元素名称（如"同意按钮"）
            adb_controller: ADB控制器（可选）
            
        Returns:
            tuple: (是否找到, 点击坐标)
        """
        result = self.detect(image_path)
        
        if not result:
            return False, None
        
        # 查找指定元素
        for element in result['detected_elements']:
            if element['class'] == element_name:
                center_x, center_y = element['center']
                print(f"\n找到 {element_name} at ({center_x:.0f}, {center_y:.0f})")
                
                # 如果提供了ADB控制器，执行点击
                if adb_controller:
                    # adb_controller.tap(device_id, int(center_x), int(center_y))
                    print(f"点击: ({int(center_x)}, {int(center_y)})")
                
                return True, (int(center_x), int(center_y))
        
        print(f"\n未找到 {element_name}")
        return False, None


def main():
    """主函数"""
    import sys
    
    # 模型路径
    page_classifier_path = "page_classifier.h5"
    yolo_model_path = "yolo_runs/button_detector/weights/best.pt"
    
    # 创建检测器
    detector = PageElementDetector(page_classifier_path, yolo_model_path)
    
    # 测试图片
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("用法: python detect_page_elements.py <image_path>")
        print("示例: python detect_page_elements.py training_data/个人页_已登录/1.png")
        return
    
    # 检测元素
    result = detector.detect(image_path)
    
    # 示例：查找并点击"同意按钮"
    # found, pos = detector.detect_and_click(image_path, "同意按钮")


if __name__ == "__main__":
    main()
