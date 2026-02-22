"""
YOLO模型管理器
用于加载和管理所有训练完成的YOLO模型
"""
import json
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Optional


class YOLOModelManager:
    """YOLO模型管理器"""
    
    def __init__(self, registry_path: str = "yolo_model_registry.json"):
        """初始化模型管理器
        
        Args:
            registry_path: 模型注册表文件路径
        """
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
        self._models = {}  # 缓存已加载的模型
    
    def _load_registry(self) -> Dict:
        """加载模型注册表"""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"模型注册表不存在: {self.registry_path}")
        
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_models(self) -> List[str]:
        """列出所有可用的模型
        
        Returns:
            模型ID列表
        """
        return list(self.registry['models'].keys())
    
    def get_model_info(self, model_id: str) -> Dict:
        """获取模型信息
        
        Args:
            model_id: 模型ID（如'homepage', 'login'等）
            
        Returns:
            模型信息字典
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"模型不存在: {model_id}")
        
        return self.registry['models'][model_id]
    
    def load_model(self, model_id: str, force_reload: bool = False) -> YOLO:
        """加载YOLO模型
        
        Args:
            model_id: 模型ID
            force_reload: 是否强制重新加载
            
        Returns:
            YOLO模型对象
        """
        # 如果已缓存且不强制重新加载，直接返回
        if model_id in self._models and not force_reload:
            return self._models[model_id]
        
        # 获取模型信息
        model_info = self.get_model_info(model_id)
        model_path = model_info['model_path']
        
        # 检查模型文件是否存在
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 缓存模型
        self._models[model_id] = model
        
        return model
    
    def get_classes(self, model_id: str) -> List[str]:
        """获取模型的类别列表
        
        Args:
            model_id: 模型ID
            
        Returns:
            类别名称列表
        """
        model_info = self.get_model_info(model_id)
        return model_info['classes']
    
    def get_page_type(self, model_id: str) -> str:
        """获取模型对应的页面类型
        
        Args:
            model_id: 模型ID
            
        Returns:
            页面类型
        """
        model_info = self.get_model_info(model_id)
        return model_info['page_type']
    
    def find_model_by_page_type(self, page_type: str) -> Optional[str]:
        """根据页面类型查找模型ID
        
        Args:
            page_type: 页面类型
            
        Returns:
            模型ID，如果未找到返回None
        """
        for model_id, model_info in self.registry['models'].items():
            if model_info['page_type'] == page_type:
                return model_id
        return None
    
    def get_performance(self, model_id: str) -> Dict:
        """获取模型性能指标
        
        Args:
            model_id: 模型ID
            
        Returns:
            性能指标字典
        """
        model_info = self.get_model_info(model_id)
        return model_info.get('performance', {})
    
    def print_summary(self):
        """打印所有模型的摘要信息"""
        print("=" * 80)
        print("YOLO模型注册表")
        print("=" * 80)
        
        for model_id, model_info in self.registry['models'].items():
            print(f"\n模型ID: {model_id}")
            print(f"  名称: {model_info['name']}")
            print(f"  页面类型: {model_info['page_type']}")
            print(f"  类别数: {model_info['num_classes']}")
            print(f"  类别: {', '.join(model_info['classes'])}")
            
            if 'performance' in model_info:
                perf = model_info['performance']
                print(f"  性能: mAP50={perf['mAP50']:.1%}, P={perf['precision']:.1%}, R={perf['recall']:.1%}")
            
            print(f"  模型路径: {model_info['model_path']}")
            
            if 'notes' in model_info:
                print(f"  备注: {model_info['notes']}")
        
        print("\n" + "=" * 80)


# 使用示例
if __name__ == "__main__":
    # 创建模型管理器
    manager = YOLOModelManager()
    
    # 打印所有模型摘要
    manager.print_summary()
    
    print("\n使用示例:")
    print("-" * 80)
    
    # 示例1: 加载首页检测模型
    print("\n1. 加载首页检测模型:")
    homepage_model = manager.load_model('homepage')
    print(f"   模型已加载: {type(homepage_model)}")
    print(f"   类别: {manager.get_classes('homepage')}")
    
    # 示例2: 根据页面类型查找模型
    print("\n2. 根据页面类型查找模型:")
    model_id = manager.find_model_by_page_type('登录页')
    print(f"   登录页对应的模型ID: {model_id}")
    
    # 示例3: 获取模型性能
    print("\n3. 获取模型性能:")
    perf = manager.get_performance('balance')
    print(f"   余额积分模型性能: mAP50={perf['mAP50']:.1%}")
    
    # 示例4: 列出所有模型
    print("\n4. 所有可用模型:")
    models = manager.list_models()
    print(f"   {', '.join(models)}")
