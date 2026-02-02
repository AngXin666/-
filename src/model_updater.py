"""
模型版本管理器 - 本地模型版本信息管理

功能：
1. 读取模型版本信息
2. 生成模型版本文件
3. 验证模型文件完整性（可选）
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class ModelVersionManager:
    """模型版本管理器（纯本地）"""
    
    def __init__(self, models_dir: Path):
        """初始化模型版本管理器
        
        Args:
            models_dir: 模型目录路径
        """
        self.models_dir = Path(models_dir)
        self.version_file = self.models_dir / "model_version.json"
        
        # 确保模型目录存在
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def get_version_info(self) -> Dict:
        """获取模型版本信息
        
        Returns:
            版本信息字典
        """
        if not self.version_file.exists():
            return {
                "version": "未知",
                "update_date": "未知",
                "description": "未找到版本文件",
                "models": {}
            }
        
        try:
            with open(self.version_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取版本文件失败: {e}")
            return {
                "version": "错误",
                "update_date": "未知",
                "description": f"读取失败: {e}",
                "models": {}
            }
    
    def save_version_info(self, version_info: Dict):
        """保存模型版本信息
        
        Args:
            version_info: 版本信息字典
        """
        try:
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(version_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存版本文件失败: {e}")
    
    def calculate_file_md5(self, file_path: Path) -> str:
        """计算文件MD5值（可选功能）
        
        Args:
            file_path: 文件路径
            
        Returns:
            MD5值
        """
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()
    
    def generate_version_file(self, version: str = "1.0.0", description: str = ""):
        """生成版本文件（用于发布新模型）
        
        Args:
            version: 版本号
            description: 版本描述
        """
        version_data = {
            "version": version,
            "update_date": datetime.now().strftime("%Y-%m-%d"),
            "description": description or f"模型版本 {version}",
            "models": {}
        }
        
        # 扫描模型文件
        model_files = {
            "page_classifier": "page_classifier_pytorch_best.pth",
            "page_classes": "page_classes.json",
            "yolo_registry": "yolo_model_registry.json",
            "page_yolo_mapping": "page_yolo_mapping.json"
        }
        
        for model_name, file_name in model_files.items():
            file_path = self.models_dir / file_name
            if file_path.exists():
                version_data['models'][model_name] = {
                    "version": version,
                    "file": file_name,
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    "description": self._get_model_description(model_name)
                }
        
        # 扫描YOLO模型
        yolo_runs_dir = self.models_dir / "yolo_runs"
        if yolo_runs_dir.exists():
            for detector_dir in yolo_runs_dir.iterdir():
                if detector_dir.is_dir():
                    # 查找best.pt文件
                    for best_pt in detector_dir.rglob("best.pt"):
                        relative_path = best_pt.relative_to(self.models_dir)
                        model_name = f"yolo_{detector_dir.name}"
                        
                        version_data['models'][model_name] = {
                            "version": version,
                            "file": str(relative_path).replace('\\', '/'),
                            "size_mb": round(best_pt.stat().st_size / (1024 * 1024), 2),
                            "description": f"YOLO检测器 - {detector_dir.name}"
                        }
        
        # 保存版本文件
        self.save_version_info(version_data)
        print(f"✅ 版本文件已生成: {self.version_file}")
        print(f"   版本号: {version}")
        print(f"   模型数量: {len(version_data['models'])}")
        
        return version_data
    
    def _get_model_description(self, model_name: str) -> str:
        """获取模型描述
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型描述
        """
        descriptions = {
            "page_classifier": "页面分类器（PyTorch）",
            "page_classes": "页面类别映射",
            "yolo_registry": "YOLO模型注册表",
            "page_yolo_mapping": "页面YOLO映射"
        }
        return descriptions.get(model_name, "未知模型")
    
    def get_formatted_info(self) -> str:
        """获取格式化的版本信息（用于显示）
        
        Returns:
            格式化的版本信息字符串
        """
        info = self.get_version_info()
        
        lines = []
        lines.append("=" * 60)
        lines.append("模型版本信息")
        lines.append("=" * 60)
        lines.append(f"版本号: {info['version']}")
        lines.append(f"更新日期: {info['update_date']}")
        lines.append(f"说明: {info.get('description', '无')}")
        lines.append(f"模型数量: {len(info['models'])}")
        lines.append("")
        
        if info['models']:
            lines.append("已安装的模型:")
            lines.append("-" * 60)
            for model_name, model_info in info['models'].items():
                lines.append(f"  • {model_name}")
                lines.append(f"    版本: {model_info.get('version', '未知')}")
                lines.append(f"    文件: {model_info.get('file', '未知')}")
                lines.append(f"    大小: {model_info.get('size_mb', 0):.2f} MB")
                lines.append(f"    说明: {model_info.get('description', '无')}")
                lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


if __name__ == '__main__':
    """测试和工具功能"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        # 生成版本文件
        version = sys.argv[2] if len(sys.argv) > 2 else "1.0.0"
        description = sys.argv[3] if len(sys.argv) > 3 else ""
        
        manager = ModelVersionManager(Path("models"))
        manager.generate_version_file(version, description)
    else:
        # 显示版本信息
        manager = ModelVersionManager(Path("models"))
        print(manager.get_formatted_info())
