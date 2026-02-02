"""
自动模型检测和注册模块
Auto Model Detection and Registration Module

功能：
1. 自动扫描models目录，检测新模型
2. 自动注册新模型到registry和mapping
3. 自动更新版本号
4. 提供GUI接口
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import hashlib


class AutoModelRegistry:
    """自动模型注册器"""
    
    def __init__(self, models_dir: Path = None):
        """初始化自动注册器
        
        Args:
            models_dir: 模型目录路径，默认为 models/
        """
        if models_dir is None:
            # 自动检测models目录
            import sys
            if getattr(sys, 'frozen', False):
                # 打包后的EXE环境
                base_dir = Path(sys.executable).parent
            else:
                # 开发环境
                base_dir = Path(__file__).parent.parent
            models_dir = base_dir / "models"
        
        self.models_dir = Path(models_dir)
        self.registry_path = self.models_dir / "yolo_model_registry.json"
        self.mapping_path = self.models_dir / "page_yolo_mapping.json"
        self.version_path = self.models_dir / "model_version.json"
        
        # 确保目录存在
        if not self.models_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.models_dir}")
    
    def scan_new_models(self) -> List[Dict]:
        """扫描新模型文件
        
        Returns:
            新模型列表，每个元素包含：
            - model_path: 模型文件路径
            - model_name: 模型名称（从路径推断）
            - file_hash: 文件哈希值
            - file_size: 文件大小（MB）
            - modified_time: 修改时间
        """
        new_models = []
        
        # 读取现有注册表
        registry = self._load_registry()
        registered_paths = set()
        
        # 收集已注册的模型路径
        for model_key, model_info in registry.get('models', {}).items():
            model_path = model_info.get('model_path', '')
            if model_path:
                registered_paths.add(model_path)
        
        # 扫描yolo_runs目录
        yolo_runs_dir = self.models_dir / "yolo_runs"
        if yolo_runs_dir.exists():
            for best_pt in yolo_runs_dir.rglob("best.pt"):
                # 计算相对路径
                relative_path = best_pt.relative_to(self.models_dir)
                path_str = str(relative_path).replace('\\', '/')
                
                # 检查是否已注册
                if path_str not in registered_paths:
                    # 从路径推断模型名称
                    model_name = self._infer_model_name(best_pt)
                    
                    new_models.append({
                        'model_path': path_str,
                        'model_name': model_name,
                        'file_hash': self._calculate_file_hash(best_pt),
                        'file_size': round(best_pt.stat().st_size / (1024 * 1024), 2),
                        'modified_time': datetime.fromtimestamp(best_pt.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'full_path': str(best_pt)
                    })
        
        # 扫描runs/detect目录
        runs_dir = self.models_dir / "runs" / "detect"
        if runs_dir.exists():
            for best_pt in runs_dir.rglob("best.pt"):
                relative_path = best_pt.relative_to(self.models_dir)
                path_str = str(relative_path).replace('\\', '/')
                
                if path_str not in registered_paths:
                    model_name = self._infer_model_name(best_pt)
                    
                    new_models.append({
                        'model_path': path_str,
                        'model_name': model_name,
                        'file_hash': self._calculate_file_hash(best_pt),
                        'file_size': round(best_pt.stat().st_size / (1024 * 1024), 2),
                        'modified_time': datetime.fromtimestamp(best_pt.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'full_path': str(best_pt)
                    })
        
        return new_models
    
    def auto_register_models(self, new_models: List[Dict], 
                           auto_increment_version: bool = True) -> Tuple[int, List[str]]:
        """自动注册新模型
        
        Args:
            new_models: 新模型列表（从scan_new_models获取）
            auto_increment_version: 是否自动递增版本号
            
        Returns:
            (成功数量, 错误信息列表)
        """
        success_count = 0
        errors = []
        
        # 读取现有注册表
        registry = self._load_registry()
        mapping = self._load_mapping()
        
        for model_info in new_models:
            try:
                # 生成模型key
                model_key = self._generate_model_key(model_info['model_name'])
                
                # 检查key是否已存在
                if model_key in registry.get('models', {}):
                    model_key = f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # 推断页面类型和类别
                page_type, classes = self._infer_page_type_and_classes(model_info['model_name'])
                
                # 添加到注册表
                registry['models'][model_key] = {
                    "name": model_info['model_name'],
                    "page_type": page_type,
                    "model_path": model_info['model_path'],
                    "classes": classes,
                    "num_classes": len(classes),
                    "performance": {
                        "mAP50": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "mAP50-95": 0.0
                    },
                    "training_date": datetime.now().strftime("%Y-%m-%d"),
                    "dataset_size": {
                        "original": 0,
                        "augmented": 0,
                        "train": 0,
                        "val": 0
                    },
                    "file_size_mb": model_info['file_size'],
                    "file_hash": model_info['file_hash'],
                    "auto_registered": True,
                    "notes": f"自动注册于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
                
                # 添加到映射（如果页面类型不存在）
                if page_type not in mapping.get('mapping', {}):
                    page_state = self._generate_page_state(page_type)
                    mapping['mapping'][page_type] = {
                        "page_state": page_state,
                        "yolo_models": [
                            {
                                "model_key": model_key,
                                "purpose": f"检测{', '.join(classes)}",
                                "priority": 1
                            }
                        ]
                    }
                else:
                    # 页面类型已存在，添加到模型列表
                    existing_models = mapping['mapping'][page_type]['yolo_models']
                    existing_keys = [m['model_key'] for m in existing_models]
                    
                    if model_key not in existing_keys:
                        existing_models.append({
                            "model_key": model_key,
                            "purpose": f"检测{', '.join(classes)}",
                            "priority": len(existing_models) + 1
                        })
                
                success_count += 1
                
            except Exception as e:
                errors.append(f"{model_info['model_name']}: {str(e)}")
        
        # 保存注册表和映射
        if success_count > 0:
            self._save_registry(registry)
            self._save_mapping(mapping)
            
            # 自动递增版本号
            if auto_increment_version:
                self._increment_version(success_count)
        
        return success_count, errors
    
    def get_version_info(self) -> Dict:
        """获取当前版本信息"""
        if not self.version_path.exists():
            return {
                "version": "1.0.0",
                "update_date": "未知",
                "description": "未找到版本文件"
            }
        
        try:
            with open(self.version_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                "version": "1.0.0",
                "update_date": "未知",
                "description": "读取失败"
            }
    
    def _load_registry(self) -> Dict:
        """加载注册表"""
        if not self.registry_path.exists():
            return {"models": {}, "version": "1.0", "last_updated": datetime.now().strftime("%Y-%m-%d")}
        
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"models": {}, "version": "1.0", "last_updated": datetime.now().strftime("%Y-%m-%d")}
    
    def _save_registry(self, registry: Dict):
        """保存注册表"""
        registry['last_updated'] = datetime.now().strftime("%Y-%m-%d")
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
    
    def _load_mapping(self) -> Dict:
        """加载映射"""
        if not self.mapping_path.exists():
            return {"mapping": {}, "version": "1.0", "last_updated": datetime.now().strftime("%Y-%m-%d")}
        
        try:
            with open(self.mapping_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"mapping": {}, "version": "1.0", "last_updated": datetime.now().strftime("%Y-%m-%d")}
    
    def _save_mapping(self, mapping: Dict):
        """保存映射"""
        mapping['last_updated'] = datetime.now().strftime("%Y-%m-%d")
        with open(self.mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    def _increment_version(self, model_count: int):
        """递增版本号
        
        Args:
            model_count: 新增模型数量
        """
        version_info = self.get_version_info()
        current_version = version_info.get('version', '1.0.0')
        
        # 解析版本号
        parts = current_version.split('.')
        if len(parts) == 3:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            
            # 递增修订版本号
            patch += 1
            if patch > 9:
                patch = 0
                minor += 1
            if minor > 9:
                minor = 0
                major += 1
            
            new_version = f"{major}.{minor}.{patch}"
        else:
            new_version = "1.0.1"
        
        # 更新版本文件
        version_info['version'] = new_version
        version_info['update_date'] = datetime.now().strftime("%Y-%m-%d")
        version_info['description'] = f"自动注册 {model_count} 个新模型"
        
        with open(self.version_path, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, ensure_ascii=False, indent=2)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()[:16]  # 只取前16位
    
    def _infer_model_name(self, model_path: Path) -> str:
        """从路径推断模型名称"""
        # 尝试从父目录名称推断
        parent_name = model_path.parent.parent.name
        
        # 清理名称
        if parent_name.endswith('_detector'):
            parent_name = parent_name[:-9]
        
        # 如果是中文页面名称，直接使用
        if any('\u4e00' <= c <= '\u9fff' for c in parent_name):
            return f"{parent_name}检测模型"
        
        # 英文名称转换
        name_map = {
            'login': '登录页',
            'warmtip': '温馨提示',
            'profile': '个人页',
            'balance': '余额积分',
            'transfer': '转账页',
            'checkin': '签到页',
            'coupon': '优惠券页',
            'home': '首页',
            'search': '搜索页',
            'wallet': '钱包页'
        }
        
        for key, value in name_map.items():
            if key in parent_name.lower():
                return f"{value}检测模型"
        
        return f"{parent_name}检测模型"
    
    def _infer_page_type_and_classes(self, model_name: str) -> Tuple[str, List[str]]:
        """推断页面类型和检测类别
        
        Returns:
            (页面类型, 类别列表)
        """
        # 从模型名称提取页面类型
        page_type = model_name.replace('检测模型', '').strip()
        
        # 默认类别（根据常见模式推断）
        default_classes = {
            '登录页': ['登陆按钮', '账号输入框', '密码输入框'],
            '温馨提示': ['确认按钮'],
            '个人页': ['昵称文本', '用户ID'],
            '余额积分': ['余额数字', '积分数字'],
            '转账页': ['转账按钮', '输入框'],
            '签到页': ['签到按钮'],
            '优惠券页': ['返回按钮'],
            '首页': ['我的按钮', '签到按钮'],
            '搜索页': ['返回按钮'],
            '钱包页': ['余额数字', '转增按钮']
        }
        
        classes = default_classes.get(page_type, ['按钮'])
        
        return page_type, classes
    
    def _generate_model_key(self, model_name: str) -> str:
        """生成模型key"""
        # 移除"检测模型"后缀
        key = model_name.replace('检测模型', '').strip()
        
        # 转换为英文key（如果是中文）
        key_map = {
            '登录页': 'login',
            '温馨提示': 'warmtip',
            '个人页': 'profile',
            '余额积分': 'balance',
            '转账页': 'transfer',
            '签到页': 'checkin',
            '优惠券页': 'coupon',
            '首页': 'home',
            '搜索页': 'search',
            '钱包页': 'wallet',
            '分类页': 'category',
            '积分页': 'points',
            '文章页': 'article'
        }
        
        return key_map.get(key, key.lower().replace(' ', '_'))
    
    def _generate_page_state(self, page_type: str) -> str:
        """生成页面状态枚举"""
        # 转换为大写下划线格式
        state_map = {
            '登录页': 'LOGIN',
            '温馨提示': 'WARMTIP',
            '个人页': 'PROFILE',
            '余额积分': 'BALANCE',
            '转账页': 'TRANSFER',
            '签到页': 'CHECKIN',
            '优惠券页': 'COUPON',
            '首页': 'HOME',
            '搜索页': 'SEARCH',
            '钱包页': 'WALLET',
            '分类页': 'CATEGORY',
            '积分页': 'POINTS',
            '文章页': 'ARTICLE'
        }
        
        return state_map.get(page_type, page_type.upper().replace(' ', '_'))


def check_and_register_new_models(log_callback=None) -> Dict:
    """检查并注册新模型（GUI调用接口）
    
    Args:
        log_callback: 日志回调函数
        
    Returns:
        结果字典：
        - new_models_count: 新模型数量
        - registered_count: 成功注册数量
        - errors: 错误列表
        - version: 新版本号
    """
    try:
        registry = AutoModelRegistry()
        
        if log_callback:
            log_callback("正在扫描新模型...")
        
        # 扫描新模型
        new_models = registry.scan_new_models()
        
        if not new_models:
            if log_callback:
                log_callback("✅ 未发现新模型")
            return {
                'new_models_count': 0,
                'registered_count': 0,
                'errors': [],
                'version': registry.get_version_info().get('version', '1.0.0')
            }
        
        if log_callback:
            log_callback(f"发现 {len(new_models)} 个新模型:")
            for model in new_models:
                log_callback(f"  - {model['model_name']} ({model['file_size']}MB)")
        
        # 自动注册
        if log_callback:
            log_callback("正在自动注册...")
        
        success_count, errors = registry.auto_register_models(new_models, auto_increment_version=True)
        
        if log_callback:
            if success_count > 0:
                log_callback(f"✅ 成功注册 {success_count} 个模型")
                version_info = registry.get_version_info()
                log_callback(f"📦 版本已更新: {version_info.get('version', '1.0.0')}")
            
            if errors:
                log_callback(f"⚠️ {len(errors)} 个模型注册失败:")
                for error in errors:
                    log_callback(f"  - {error}")
        
        return {
            'new_models_count': len(new_models),
            'registered_count': success_count,
            'errors': errors,
            'version': registry.get_version_info().get('version', '1.0.0')
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"❌ 自动注册失败: {str(e)}")
        return {
            'new_models_count': 0,
            'registered_count': 0,
            'errors': [str(e)],
            'version': '1.0.0'
        }
