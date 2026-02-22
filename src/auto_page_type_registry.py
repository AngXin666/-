"""
自动页面类型注册模块
Auto Page Type Registration Module

功能：
1. 扫描 page_classes.json 中的新类别
2. 自动生成页面状态映射配置
3. 更新 page_state_mapping.json
4. 提供GUI调用接口
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class AutoPageTypeRegistry:
    """自动页面类型注册器"""
    
    def __init__(self, models_dir: Path = None, config_dir: Path = None):
        """初始化注册器
        
        Args:
            models_dir: 模型目录路径
            config_dir: 配置目录路径
        """
        if models_dir is None:
            import sys
            if getattr(sys, 'frozen', False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).parent.parent
            models_dir = base_dir / "models"
            config_dir = base_dir / "config"
        
        self.models_dir = Path(models_dir)
        self.config_dir = Path(config_dir)
        self.classes_path = self.models_dir / "page_classes.json"
        self.mapping_path = self.config_dir / "page_state_mapping.json"
    
    def scan_new_page_types(self) -> List[str]:
        """扫描新的页面类型
        
        Returns:
            未映射的页面类型列表
        """
        # 加载页面类别
        if not self.classes_path.exists():
            return []
        
        with open(self.classes_path, 'r', encoding='utf-8') as f:
            page_classes = json.load(f)
        
        # 加载现有映射
        if not self.mapping_path.exists():
            return page_classes  # 如果映射文件不存在，所有类别都是新的
        
        with open(self.mapping_path, 'r', encoding='utf-8') as f:
            mapping_config = json.load(f)
        
        # 查找未映射的类别
        mapped_classes = set(mapping_config.get('mappings', {}).keys())
        unmapped = [cls for cls in page_classes if cls not in mapped_classes]
        
        return unmapped
    
    def generate_state_config(self, class_name: str) -> Dict:
        """生成页面状态配置
        
        Args:
            class_name: 类别名称（中文）
            
        Returns:
            配置字典
        """
        # 生成状态名称
        state, state_value = self._generate_state_name(class_name)
        
        return {
            "state": state,
            "state_value": state_value,
            "chinese_name": class_name,
            "description": f"{class_name}（自动生成）"
        }
    
    def register_page_types(self, page_types: List[str], 
                          auto_backup: bool = True) -> Tuple[int, List[str]]:
        """注册新的页面类型
        
        Args:
            page_types: 页面类型列表
            auto_backup: 是否自动备份配置文件
            
        Returns:
            (成功数量, 错误列表)
        """
        if not page_types:
            return 0, []
        
        success_count = 0
        errors = []
        
        try:
            # 加载现有配置
            if self.mapping_path.exists():
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    mapping_config = json.load(f)
            else:
                # 创建新配置
                mapping_config = {
                    "version": "1.0.0",
                    "description": "页面类型到PageState的映射配置",
                    "mappings": {},
                    "auto_register": {
                        "enabled": True,
                        "default_state": "UNKNOWN",
                        "default_state_value": "unknown",
                        "description": "未映射的类别自动注册为UNKNOWN状态"
                    }
                }
            
            # 备份原配置
            if auto_backup and self.mapping_path.exists():
                backup_path = self.config_dir / f"page_state_mapping.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                import shutil
                shutil.copy(self.mapping_path, backup_path)
            
            # 添加新映射
            for class_name in page_types:
                try:
                    config = self.generate_state_config(class_name)
                    mapping_config['mappings'][class_name] = config
                    success_count += 1
                except Exception as e:
                    errors.append(f"{class_name}: {str(e)}")
            
            # 保存配置
            if success_count > 0:
                with open(self.mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_config, f, ensure_ascii=False, indent=2)
            
            return success_count, errors
            
        except Exception as e:
            errors.append(f"保存配置失败: {str(e)}")
            return success_count, errors
    
    def get_required_enum_code(self, page_types: List[str]) -> str:
        """获取需要添加的PageState枚举代码
        
        Args:
            page_types: 页面类型列表
            
        Returns:
            需要添加的代码字符串
        """
        lines = []
        lines.append("# 需要在 src/page_detector.py 中添加以下枚举：")
        lines.append("")
        lines.append("class PageState(Enum):")
        lines.append("    # ... 现有的枚举 ...")
        lines.append("")
        
        for class_name in page_types:
            state, state_value = self._generate_state_name(class_name)
            lines.append(f"    {state} = \"{state_value}\"  # {class_name}")
        
        lines.append("")
        lines.append("    @property")
        lines.append("    def chinese_name(self) -> str:")
        lines.append("        name_map = {")
        lines.append("            # ... 现有的映射 ...")
        
        for class_name in page_types:
            _, state_value = self._generate_state_name(class_name)
            lines.append(f"            \"{state_value}\": \"{class_name}\",")
        
        lines.append("        }")
        lines.append("        return name_map.get(self.value, self.value)")
        
        return "\n".join(lines)
    
    def _generate_state_name(self, class_name: str) -> Tuple[str, str]:
        """生成状态名称
        
        Args:
            class_name: 类别名称（中文）
            
        Returns:
            (STATE, state_value) 元组
        """
        # 简单的映射规则
        replacements = {
            '页': '_PAGE',
            '弹窗': '_POPUP',
            '广告': '_AD',
            '流水': '_HISTORY',
            '桌面': '_LAUNCHER',
            '提示': '_TIP',
            '确认': '_CONFIRM',
            '异常': '_ERROR',
            '公告': '_NOTICE',
            '服务': '_SERVICE',
            '启动': 'STARTUP',
            '已登录': '_LOGGED',
            '未登录': '_UNLOGGED',
        }
        
        # 生成英文状态名
        state = class_name
        for cn, en in replacements.items():
            state = state.replace(cn, en)
        
        # 清理状态名
        state = state.replace('__', '_').strip('_').upper()
        if not state or state == '_':
            state = 'UNKNOWN'
        
        # 生成state_value（小写+下划线）
        state_value = state.lower()
        
        return state, state_value


def check_and_register_page_types(log_callback=None) -> Dict:
    """检查并注册新页面类型（GUI调用接口）
    
    Args:
        log_callback: 日志回调函数
        
    Returns:
        结果字典：
        - new_types_count: 新类型数量
        - registered_count: 成功注册数量
        - errors: 错误列表
        - enum_code: 需要添加的枚举代码
    """
    try:
        registry = AutoPageTypeRegistry()
        
        if log_callback:
            log_callback("正在扫描新页面类型...")
        
        # 扫描新类型
        new_types = registry.scan_new_page_types()
        
        if not new_types:
            if log_callback:
                log_callback("✅ 未发现新页面类型")
            return {
                'new_types_count': 0,
                'registered_count': 0,
                'errors': [],
                'enum_code': ''
            }
        
        if log_callback:
            log_callback(f"发现 {len(new_types)} 个新页面类型:")
            for page_type in new_types:
                log_callback(f"  - {page_type}")
        
        # 自动注册
        if log_callback:
            log_callback("正在自动注册...")
        
        success_count, errors = registry.register_page_types(new_types, auto_backup=True)
        
        # 生成枚举代码
        enum_code = registry.get_required_enum_code(new_types)
        
        if log_callback:
            if success_count > 0:
                log_callback(f"✅ 成功注册 {success_count} 个页面类型")
                log_callback(f"📝 配置已更新: {registry.mapping_path}")
            
            if errors:
                log_callback(f"⚠️ {len(errors)} 个类型注册失败:")
                for error in errors:
                    log_callback(f"  - {error}")
        
        return {
            'new_types_count': len(new_types),
            'registered_count': success_count,
            'errors': errors,
            'enum_code': enum_code,
            'new_types': new_types
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"❌ 自动注册失败: {str(e)}")
        return {
            'new_types_count': 0,
            'registered_count': 0,
            'errors': [str(e)],
            'enum_code': ''
        }
